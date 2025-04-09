import os
import argparse
import getpass
import json
import sqlite3
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

# Replace checkpoint management functions with SQLite-based versions
def init_checkpoint_db(db_path):
    """Initialize the SQLite checkpoint database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS process_state (
        id INTEGER PRIMARY KEY,
        stage TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        details TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS language_progress (
        language TEXT PRIMARY KEY,
        status TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feature_extraction (
        id INTEGER PRIMARY KEY,
        progress INTEGER,
        total INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    logger.info(f"Initialized checkpoint database at {db_path}")

def save_checkpoint(db_path, stage, details=None):
    """Save checkpoint state to SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Convert details dict to JSON string if it exists
    details_json = json.dumps(details) if details else None
    
    # Insert new state
    cursor.execute(
        "INSERT INTO process_state (stage, details) VALUES (?, ?)",
        (stage, details_json)
    )
    
    conn.commit()
    conn.close()
    logger.info(f"Checkpoint saved: {stage}")

def get_latest_checkpoint(db_path):
    """Get the latest checkpoint state from SQLite database."""
    if not os.path.exists(db_path):
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get the latest state
        cursor.execute(
            "SELECT stage, details FROM process_state ORDER BY id DESC LIMIT 1"
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            stage, details_json = result
            details = json.loads(details_json) if details_json else None
            return {"stage": stage, "details": details}
        return None
    except sqlite3.Error as e:
        logger.error(f"Error reading checkpoint database: {str(e)}")
        return None

def save_language_progress(db_path, language, status):
    """Save language dataset loading progress."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Insert or replace language status
    cursor.execute(
        "INSERT OR REPLACE INTO language_progress (language, status) VALUES (?, ?)",
        (language, status)
    )
    
    conn.commit()
    conn.close()

def get_completed_languages(db_path):
    """Get list of languages that have been successfully loaded."""
    if not os.path.exists(db_path):
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT language FROM language_progress WHERE status = 'completed'"
        )
        results = cursor.fetchall()
        conn.close()
        
        return [lang[0] for lang in results]
    except sqlite3.Error as e:
        logger.error(f"Error reading language progress: {str(e)}")
        return []

def save_feature_extraction_progress(db_path, progress, total):
    """Save feature extraction progress."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Insert new progress record
    cursor.execute(
        "INSERT INTO feature_extraction (progress, total) VALUES (?, ?)",
        (progress, total)
    )
    
    conn.commit()
    conn.close()

def get_latest_feature_extraction_progress(db_path):
    """Get the latest feature extraction progress."""
    if not os.path.exists(db_path):
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT progress, total FROM feature_extraction ORDER BY id DESC LIMIT 1"
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            progress, total = result
            return {"progress": progress, "total": total}
        return None
    except sqlite3.Error as e:
        logger.error(f"Error reading feature extraction progress: {str(e)}")
        return None

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

def load_and_prepare_datasets(languages=None, cache_dir=None, checkpoint_dir=None):
    """Load and prepare datasets more efficiently with SQLite-based checkpointing.
    
    Args:
        languages: List of language codes to load
        cache_dir: Directory to cache the datasets
        checkpoint_dir: Directory to store checkpoints
    """
    if languages is None:
        languages = ["sw", "en"]  # Default to Swahili and English
    
    db_path = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        db_path = os.path.join(checkpoint_dir, "checkpoints.db")
        init_checkpoint_db(db_path)
        
        # Check if we've already loaded the datasets
        checkpoint = get_latest_checkpoint(db_path)
        if checkpoint and checkpoint['stage'] == 'datasets_loaded':
            logger.info(f"Resuming from previous dataset loading checkpoint")
            if checkpoint['details'] and 'dataset_path' in checkpoint['details']:
                return checkpoint['details']['dataset_path']
    
    logger.info(f"Loading datasets for languages: {languages}")
    
    try:
        datasets = {}
        completed_languages = get_completed_languages(db_path) if db_path else []
        
        for i, lang in enumerate(languages):
            # Skip languages that have already been loaded
            if lang in completed_languages:
                logger.info(f"Skipping {lang} dataset (already loaded)")
                continue
                
            logger.info(f"Loading {lang} dataset ({i+1}/{len(languages)})...")
            # Load both train+validation and test in one call to avoid redundant downloads
            ds = load_dataset(
                "mozilla-foundation/common_voice_11_0", 
                lang, 
                split=["train+validation", "test"],
                cache_dir=cache_dir
            )
            datasets[lang] = {"train": ds[0], "test": ds[1]}
            
            # Save intermediate checkpoint after each language dataset is loaded
            if db_path:
                save_language_progress(db_path, lang, "completed")
        
        # Get any previously completed languages that weren't in the current run
        for lang in completed_languages:
            if lang not in datasets and lang in languages:
                logger.info(f"Loading previously completed {lang} dataset...")
                ds = load_dataset(
                    "mozilla-foundation/common_voice_11_0", 
                    lang, 
                    split=["train+validation", "test"],
                    cache_dir=cache_dir
                )
                datasets[lang] = {"train": ds[0], "test": ds[1]}
        
        # Combine training datasets
        logger.info("Combining datasets...")
        train_datasets = [ds["train"] for ds in datasets.values()]
        test_datasets = [ds["test"] for ds in datasets.values()]
        
        combined_train = concatenate_datasets(train_datasets).shuffle(seed=42)
        combined_test = concatenate_datasets(test_datasets).shuffle(seed=42)
        
        common_voice = DatasetDict({
            "train": combined_train,
            "test": combined_test
        })
        
        # Remove unwanted columns
        logger.info("Removing unwanted columns...")
        columns_to_remove = [
            "accent", "age", "client_id", "down_votes", 
            "gender", "locale", "path", "segment", "up_votes"
        ]
        common_voice = common_voice.remove_columns(columns_to_remove)
        
        # Prepare audio data
        logger.info("Preparing audio data...")
        common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
        
        # Save dataset checkpoint
        if checkpoint_dir:
            # Save the dataset to disk to resume later
            dataset_path = os.path.join(checkpoint_dir, "processed_dataset")
            common_voice.save_to_disk(dataset_path)
            save_checkpoint(db_path, "datasets_loaded", {"dataset_path": dataset_path})
        
        return common_voice
    
    except Exception as e:
        logger.error(f"Error during dataset loading: {str(e)}")
        if db_path:
            logger.info("Saving error checkpoint before exiting...")
            save_checkpoint(db_path, "error", {
                "error": str(e),
                "last_state": "dataset_loading"
            })
        raise

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
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=languages, task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, language=languages, task="transcribe")
    
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

def prepare_dataset_features(batch, feature_extractor, tokenizer, counter=None, total=None, db_path=None):
    """Process a single batch to prepare features and labels with progress tracking using SQLite."""
    if counter is not None and counter.value % 100 == 0 and total is not None:
        progress = counter.value
        progress_percentage = (progress / total) * 100
        logger.info(f"Processing features: {progress}/{total} ({progress_percentage:.2f}%)")
        
        # Save checkpoint periodically during feature extraction
        if db_path:
            save_feature_extraction_progress(db_path, progress, total)
    
    audio = batch["audio"]
    
    # Compute input features from audio array
    batch["input_features"] = feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    
    if counter is not None:
        counter.value += 1
        
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
    # Set up checkpoint directory and database
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    db_path = os.path.join(checkpoint_dir, "checkpoints.db")
    init_checkpoint_db(db_path)
    
    # Check for existing process checkpoint
    checkpoint = get_latest_checkpoint(db_path)
    current_stage = checkpoint.get('stage') if checkpoint else None
    
    # Login to platforms if needed and checkpoint doesn't exist
    if current_stage is None:
        login_to_platforms()
        save_checkpoint(db_path, "login_completed")
    
    # Load and prepare datasets
    common_voice = None
    dataset_path = None
    
    if current_stage is None or current_stage in ['login_completed', 'error']:
        logger.info("Loading and preparing datasets...")
        try:
            result = load_and_prepare_datasets(
                languages=args.languages,
                cache_dir=args.cache_dir,
                checkpoint_dir=checkpoint_dir
            )
            
            # Check if the result is a dataset or a path to a saved dataset
            if isinstance(result, str) and os.path.exists(result):
                dataset_path = result
                save_checkpoint(db_path, "datasets_loaded", {"dataset_path": dataset_path})
            else:
                common_voice = result
                save_checkpoint(db_path, "datasets_loaded")
        except Exception as e:
            logger.error(f"Error during dataset loading: {str(e)}")
            save_checkpoint(db_path, "error", {
                "error": str(e),
                "last_state": "dataset_loading"
            })
            raise
    
    # Load dataset from disk if path is provided
    if dataset_path and common_voice is None:
        logger.info(f"Loading processed dataset from {dataset_path}...")
        common_voice = DatasetDict.load_from_disk(dataset_path)
    
    # Load model and processors if not already done
    model, feature_extractor, tokenizer, processor = None, None, None, None
    if current_stage is None or current_stage in ['login_completed', 'datasets_loaded', 'error']:
        logger.info(f"Loading model {args.model_name}...")
        try:
            model, feature_extractor, tokenizer, processor = load_model_and_processors(
                model_name=args.model_name,
                languages=[lang.capitalize() for lang in args.languages],
                device_map=args.device_map
            )
            save_checkpoint(db_path, "model_loaded")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            save_checkpoint(db_path, "error", {
                "error": str(e),
                "last_state": "model_loading"
            })
            raise
    
    # Process features if not already done
    processed_dataset_path = os.path.join(checkpoint_dir, "processed_features_dataset")
    
    if os.path.exists(processed_dataset_path):
        logger.info(f"Loading processed features dataset from {processed_dataset_path}...")
        common_voice = DatasetDict.load_from_disk(processed_dataset_path)
    elif current_stage is None or current_stage in ['datasets_loaded', 'model_loaded', 'error']:
        logger.info("Preparing datasets with features...")
        try:
            # Create a shared counter for progress tracking
            from multiprocessing import Value
            counter = Value('i', 0)
            total = len(common_voice["train"]) + len(common_voice["test"])
            
            # Create a dataset processing function with progress tracking
            prepare_fn = lambda batch: prepare_dataset_features(
                batch, feature_extractor, tokenizer, counter, total, db_path
            )
            
            # Process the dataset
            common_voice = common_voice.map(
                prepare_fn,
                remove_columns=common_voice.column_names["train"],
                num_proc=args.num_proc,
                desc="Processing audio and generating features"
            )
            
            # Save processed dataset
            logger.info(f"Saving processed features dataset to {processed_dataset_path}...")
            common_voice.save_to_disk(processed_dataset_path)
            save_checkpoint(db_path, "features_processed")
        except Exception as e:
            logger.error(f"Error processing features: {str(e)}")
            save_checkpoint(db_path, "error", {
                "error": str(e),
                "last_state": "feature_processing"
            })
            raise
    
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
    processor_path = os.path.join(args.output_dir, "processor")
    if not os.path.exists(processor_path):
        logger.info(f"Saving processor to {processor_path}")
        processor.save_pretrained(processor_path)
    
    # Train the model
    logger.info("Starting training...")
    save_checkpoint(db_path, "training_started")
    
    # Training will automatically resume from the last checkpoint if it exists
    trainer.train(resume_from_checkpoint=checkpoint is not None)
    
    save_checkpoint(db_path, "training_completed")
    
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
        save_checkpoint(db_path, "pushed_to_hub")
    
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