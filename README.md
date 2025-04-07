# Multilingual Whisper ASR Fine-Tuning

This repository contains code for fine-tuning OpenAI's Whisper model for multilingual Automatic Speech Recognition (ASR), with a focus on low-resource languages.

## Overview

Whisper is a powerful pre-trained model for ASR published by OpenAI. This project provides tools to fine-tune Whisper on multilingual datasets, specifically designed for Swahili and English using the Common Voice dataset, but adaptable to other languages.

## Features

- Fine-tune Whisper models of different sizes (tiny, base, small, medium, large)
- Support for multiple languages
- Integration with Weights & Biases for experiment tracking
- Automatic model uploading to Hugging Face Hub
- Efficient data processing and training

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM (more for larger models)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

The simplest way to run the fine-tuning process:

```bash
python main.py
```

This will use the default configuration:
- Model: openai/whisper-large
- Languages: Swahili (sw) and English (en)
- Output directory: ./whisper-large-mult

### Command-line Arguments

The script supports numerous command-line arguments for customization:

#### Model and Dataset Parameters

- `--model_name`: Model to fine-tune (default: "openai/whisper-large")
- `--model_size`: Size of the model - tiny, base, small, medium, large (default: "large")
- `--model_name_suffix`: Suffix for model name when pushing to hub (default: "Kenn")
- `--languages`: Languages to train on (default: ["sw", "en"])
- `--cache_dir`: Directory to cache datasets

#### Training Parameters

- `--output_dir`: Directory to save the model (default: "./whisper-large-mult")
- `--batch_size`: Batch size per GPU for training (default: 16)
- `--eval_batch_size`: Batch size per GPU for evaluation (default: 8)
- `--gradient_accumulation_steps`: Steps to accumulate gradients (default: 1)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--warmup_steps`: Number of warmup steps (default: 500)
- `--max_steps`: Number of training steps (default: 3000)
- `--save_steps`: Steps between saving checkpoints (default: 1000)
- `--eval_steps`: Steps between evaluations (default: 1000)
- `--logging_steps`: Steps between logging (default: 100)

#### Processing Parameters

- `--num_proc`: Number of processes for preprocessing (default: 4)
- `--device_map`: Device mapping strategy (default: "auto")


#### Hugging Face Hub Parameters

- `--push_to_hub`: Push model to Hugging Face Hub (default: True)
- `--hub_model_id`: Model ID for Hugging Face Hub
- `--hub_strategy`: When to push to hub - "end", "every_save", "checkpoint" (default: "every_save")

### Examples

#### Fine-tune a smaller model

```bash
python main.py --model_name "openai/whisper-small" --model_size "small" --batch_size 32
```

#### Train on different languages

```bash
python main.py --languages fr de es --model_name_suffix "Multilingual-European"
```

#### Custom training configuration

```bash
python main.py --learning_rate 5e-5 --warmup_steps 1000 --max_steps 5000 --gradient_accumulation_steps 2
```

## Authentication

The script will prompt for:

1. **Hugging Face Token**: Required to push models to the Hugging Face Hub
   - Alternatively, set the `HUGGINGFACE_TOKEN` environment variable
   - Get your token from: https://huggingface.co/settings/tokens

2. **Weights & Biases API Key**: For experiment tracking (optional)
   - Alternatively, set the `WANDB_API_KEY` environment variable
   - Get your key from: https://wandb.ai/settings


Alternatively you can run the notebook line by line.

## Resource Requirements

Resource requirements vary based on the model size:

| Model Size | GPU Memory | RAM       | Training Time    |
|------------|------------|-----------|------------------|
| tiny       | 2GB+       | 8GB+      | ~2-4 hours       |
| base       | 4GB+       | 8GB+      | ~4-8 hours       |
| small      | 8GB+       | 16GB+     | ~10-20 hours     |
| medium     | 16GB+      | 32GB+     | ~24-48 hours     |
| large      | 24GB+      | 64GB+     | ~48-96 hours     |

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM) errors**:
   - Reduce batch size (`--batch_size`)
   - Increase gradient accumulation steps (`--gradient_accumulation_steps`)
   - Use a smaller model size

2. **Slow training**:
   - Ensure you're using a CUDA-compatible GPU
   - Increase the number of preprocessing workers (`--num_proc`)
   - Check if mixed precision (fp16) is enabled

3. **Authentication errors**:
   - Ensure your Hugging Face token has write permissions
   - Check network connectivity to Hugging Face/Weights & Biases servers
