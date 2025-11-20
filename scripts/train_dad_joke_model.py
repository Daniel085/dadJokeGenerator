#!/usr/bin/env python3
"""
Fine-tune TinyLlama on dad jokes using LoRA
Optimized for Mac mini training with MPS (Metal Performance Shaders)
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import json
from pathlib import Path


# Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./dad-joke-model"
TRAINING_DATA = "./training_data/dad_jokes_train.jsonl"
VALIDATION_DATA = "./training_data/dad_jokes_validation.jsonl"

# Training hyperparameters
EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4  # Effective batch size: 16
LEARNING_RATE = 2e-4
MAX_LENGTH = 256


def check_device():
    """Check available compute device"""
    if torch.backends.mps.is_available():
        device = "mps"
        print("🎮 Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("🎮 Using CUDA GPU")
    else:
        device = "cpu"
        print("💻 Using CPU (slower)")

    return device


def load_and_prepare_data(tokenizer, train_file, val_file=None):
    """Load and tokenize training data"""
    print("📂 Loading training data...")

    # Load datasets
    data_files = {"train": train_file}
    if val_file and os.path.exists(val_file):
        data_files["validation"] = val_file

    dataset = load_dataset("json", data_files=data_files)

    print(f"   Training samples: {len(dataset['train'])}")
    if "validation" in dataset:
        print(f"   Validation samples: {len(dataset['validation'])}")

    def format_chat(example):
        """Format messages as chat template"""
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    # Format conversations
    dataset = dataset.map(format_chat, remove_columns=["messages"])

    def tokenize_function(examples):
        """Tokenize texts"""
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )
        result["labels"] = result["input_ids"].copy()
        return result

    # Tokenize
    print("🔤 Tokenizing...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    return tokenized_dataset


def setup_lora_model(model):
    """Configure LoRA for fine-tuning"""
    print("🔧 Configuring LoRA...")

    lora_config = LoraConfig(
        r=16,                              # Rank of LoRA matrices
        lora_alpha=32,                     # Scaling parameter
        target_modules=[                    # Which layers to adapt
            "q_proj", "k_proj",
            "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,                 # Dropout for regularization
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / all_params

    print(f"   Trainable params: {trainable_params:,}")
    print(f"   All params: {all_params:,}")
    print(f"   Trainable: {trainable_percent:.2f}%")

    return model


def main():
    print("=" * 80)
    print("🎓 Dad Joke Model Training")
    print("=" * 80)

    # Check device
    device = check_device()

    # Check if training data exists
    train_file = Path(TRAINING_DATA)
    if not train_file.exists():
        print(f"\n❌ Training data not found: {train_file}")
        print("Please run generate_training_data.py first to create training data.")
        return

    # 1. Load tokenizer
    print("\n📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load base model
    print("🧠 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map="auto"
    )

    # 3. Setup LoRA
    model = setup_lora_model(model)

    # 4. Load and prepare data
    val_file = Path(VALIDATION_DATA)
    dataset = load_and_prepare_data(
        tokenizer,
        str(train_file),
        str(val_file) if val_file.exists() else None
    )

    # 5. Training arguments
    print("\n⚙️  Configuring training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        save_steps=500,
        eval_steps=500 if "validation" in dataset else None,
        evaluation_strategy="steps" if "validation" in dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if "validation" in dataset else False,
        bf16=True if device != "cpu" else False,
        fp16=False,
        report_to="none",  # Change to "wandb" for experiment tracking
        remove_unused_columns=False,
    )

    # 6. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # 7. Initialize trainer
    print("🏋️  Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 8. Train!
    print("\n" + "=" * 80)
    print("🎓 Starting training...")
    print("=" * 80)
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION})")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Device: {device}")
    print("=" * 80 + "\n")

    trainer.train()

    # 9. Save final model
    print("\n💾 Saving final model...")
    final_dir = f"{OUTPUT_DIR}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print("=" * 80)
    print(f"📁 Model saved to: {final_dir}")
    print("\nNext steps:")
    print("1. Test the model: python scripts/test_model.py")
    print("2. Evaluate quality: python scripts/evaluate_model.py")
    print("3. Merge & quantize: python scripts/merge_and_quantize.py")
    print("4. Upload to Hugging Face: huggingface-cli upload")
    print("=" * 80)


if __name__ == "__main__":
    main()
