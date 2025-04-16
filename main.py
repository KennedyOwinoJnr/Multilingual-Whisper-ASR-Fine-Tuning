import os
import argparse
import getpass
from datasets import load_dataset, DatasetDict, Audio, concatenate_datasets
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    GenerationConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import logging
import wandb
from huggingface_hub import login as hf_login

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def login_to_platforms():
    """Prompt for and set up authentication tokens for HF and W&B."""
    # Hugging Face login
    if os.environ.get("HUGGINGFACE_TOKEN") is None:
        hf_token = getpass.getpass("Enter your Hugging Face token (or press enter to skip): ")
        if hf_token.strip():
            hf_login(token=hf_token)
            logger.info("Successfully logged in to Hugging Face Hub")
        else:
            logger.warning("No Hugging Face token provided. Push to hub functionality may be limited.")
    
    # Weights & Biases login
    if os.environ.get("WANDB_API_KEY") is None:
        wandb_token = getpass.getpass("Enter your Weights & Biases API token (or press enter to skip): ")
        if wandb_token.strip():
            os.environ["WANDB_API_KEY"] = wandb_token
            logger.info("Successfully set Weights & Biases API token")
        else:
            logger.warning("No W&B token provided. Training metrics will not be logged to W&B.")

def load_and_prepare_datasets(languages=None, cache_dir=None):
    """Load and prepare datasets more efficiently.
    
    Args:
        languages: List of language codes to load
        cache_dir: Directory to cache the datasets
    """
    if languages is None:
        languages = ["sw", "en"]  # Default to Swahili and English
    
    logger.info(f"Loading datasets for languages: {languages}")
    
    datasets = {}
    for lang in languages:
        logger.info(f"Loading {lang} dataset...")
        # Load both train+validation and test in one call to avoid redundant downloads
        ds = load_dataset(
            "mozilla-foundation/common_voice_11_0", 
            lang, 
            split=["train+validation", "test"],
            cache_dir=cache_dir
        )
        datasets[lang] = {"train": ds[0], "test": ds[1]}
    
    # Combine training datasets
    train_datasets = [ds["train"] for ds in datasets.values()]
    test_datasets = [ds["test"] for ds in datasets.values()]
    
    combined_train = concatenate_datasets(train_datasets).shuffle(seed=42)
    combined_test = concatenate_datasets(test_datasets).shuffle(seed=42)
    
    common_voice = DatasetDict({
        "train": combined_train,
        "test": combined_test
    })
    
    # Remove unwanted columns
    columns_to_remove = [
        "accent", "age", "client_id", "down_votes", 
        "gender", "path", "segment", "up_votes"
    ]
    common_voice = common_voice.remove_columns(columns_to_remove)
    
    # Prepare audio data
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    
    return common_voice

def load_model_and_processors(model_name="openai/whisper-large", languages=None, device_map="auto"):
    """Load model and processors more efficiently.
    
    Args:
        model_name: Name of the Whisper model to load
        languages: List of languages to use
        device_map: Device mapping strategy for model loading
    """
    if languages is None:
        languages = ["Swahili", "English"]
        
    logger.info(f"Loading feature extractor, tokenizer, and processor from {model_name}")
    
    # Load components with a single API call when possible
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)
    
    logger.info(f"Loading model from {model_name}")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.float16  # Use mixed precision by default
    )
    
    # Configure generation settings
    model.generation_config.task = "transcribe"
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    
    return model, feature_extractor, tokenizer, processor

def prepare_dataset_features(batch, feature_extractor, tokenizer):
    """Process a single batch to prepare features and labels."""
    audio = batch["audio"]
    
    # Compute input features from audio array
    batch["input_features"] = feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    #get lan codes from locale column
    lan_code = batch['locale']
    lang_mapping = {
        'sw': 'Swahili',
        'en': 'English'
    }

    language  = lang_mapping.get(lan_code, 'English') # we default to English if not found
    
    # Tokenize text with language and task
    batch["labels"] = tokenizer(
        batch["sentence"], 
        language=language, 
        task="transcribe"
    ).input_ids

    return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def get_compute_metrics_fn(tokenizer, metric_name="wer"):
    """Create a compute_metrics function with the given tokenizer and metric."""
    metric = evaluate.load(metric_name)
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # We do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    return compute_metrics

def main(args):
    # Login to platforms if needed
    login_to_platforms()
    
    # Load and prepare datasets
    logger.info("Loading and preparing datasets...")
    common_voice = load_and_prepare_datasets(
        languages=args.languages,
        cache_dir=args.cache_dir
    )
    
    # Load model and processors
    logger.info(f"Loading model {args.model_name}...")
    model, feature_extractor, tokenizer, processor = load_model_and_processors(
        model_name=args.model_name,
        languages=[lang.capitalize() for lang in args.languages],
        device_map=args.device_map
    )
    
    # Prepare datasets with features
    logger.info("Preparing datasets with features...")
    prepare_fn = lambda batch: prepare_dataset_features(batch, feature_extractor, tokenizer)
    common_voice = common_voice.map(
        prepare_fn,
        remove_columns=common_voice.column_names["train"],
        num_proc=args.num_proc,
        desc="Processing audio and generating features"
    )
    
    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # Set up compute metrics function
    compute_metrics = get_compute_metrics_fn(tokenizer)
    
    # Configure training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=True,
        generation_max_length=100,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=["tensorboard", "wandb"] if "WANDB_API_KEY" in os.environ else ["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy=args.hub_strategy,
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    
    # Save processor for later use
    logger.info(f"Saving processor to {args.output_dir}")
    processor.save_pretrained(args.output_dir)
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Push model to the hub if requested
    if args.push_to_hub:
        logger.info("Pushing model to the Hugging Face Hub...")
        languages_str = "-".join(args.languages)
        kwargs = {
            "dataset_tags": "mozilla-foundation/common_voice_11_0",
            "dataset": "Common Voice 11.0",
            "dataset_args": f"config: {languages_str}, split: test",
            "language": languages_str,
            "model_name": f"ASR Whisper {args.model_size.capitalize()} - {args.model_name_suffix}",
            "finetuned_from": args.model_name,
            "tasks": "automatic-speech-recognition",
        }
        trainer.push_to_hub(**kwargs)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Whisper model for multilingual ASR")
    
    # Model and dataset parameters
    parser.add_argument("--model_name", type=str, default="openai/whisper-large", 
                        help="Name or path of the model to fine-tune")
    parser.add_argument("--model_size", type=str, default="large", 
                        help="Size of the model (tiny, base, small, medium, large)")
    parser.add_argument("--model_name_suffix", type=str, default="Kenn", 
                        help="Suffix to add to the model name when pushing to hub")
    parser.add_argument("--languages", type=str, nargs="+", default=["sw", "en"], 
                        help="Languages to train on")
    parser.add_argument("--cache_dir", type=str, default=None, 
                        help="Directory to cache the datasets")
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, default="./whisper-large-mult", 
                        help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size per GPU for training")
    parser.add_argument("--eval_batch_size", type=int, default=8, 
                        help="Batch size per GPU for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=1e-5, 
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, 
                        help="Number of warmup steps")
    parser.add_argument("--max_steps", type=int, default=3000, 
                        help="Number of training steps")
    parser.add_argument("--save_steps", type=int, default=1000, 
                        help="Number of steps between saving checkpoints")
    parser.add_argument("--eval_steps", type=int, default=1000, 
                        help="Number of steps between evaluations")
    parser.add_argument("--logging_steps", type=int, default=100, 
                        help="Number of steps between logging")
    
    # Processing parameters
    parser.add_argument("--num_proc", type=int, default=4, 
                        help="Number of processes to use for preprocessing")
    parser.add_argument("--device_map", type=str, default="auto", 
                        help="Device map for model loading")
    
    # Hugging Face Hub parameters
    parser.add_argument("--push_to_hub", action="store_true", default=True, 
                        help="Whether to push the model to the Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, 
                        help="Model ID to use when pushing to the Hugging Face Hub")
    parser.add_argument("--hub_strategy", type=str, default="every_save", 
                        choices=["end", "every_save", "checkpoint"], 
                        help="When to push to the hub")
    
    args = parser.parse_args()
    main(args)
