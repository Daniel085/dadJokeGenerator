#!/usr/bin/env python3
"""
Fine-tune DistilGPT-2 on dad jokes.

Unlike train_from_scratch.py, this starts from a pretrained model that
already knows English, so it only needs to learn the joke style — not the
language itself. This is the fix for incoherent punchlines: a from-scratch
model trained on ~150K tokens memorizes joke *shape* but can't generalize
meaning.

Usage:
    python scripts/finetune_gpt2.py

Expects training data in:
    training_data/dad_jokes_train.jsonl
    training_data/dad_jokes_validation.jsonl

Outputs:
    dad-joke-model/finetuned_best.pt   (state dict + config for export)
"""

import os
import json
import math
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

TRAINING_DATA = "training_data/dad_jokes_train.jsonl"
VALIDATION_DATA = "training_data/dad_jokes_validation.jsonl"
OUTPUT_DIR = "dad-joke-model"
CHECKPOINT = os.path.join(OUTPUT_DIR, "finetuned_best.pt")

BASE_MODEL = "distilgpt2"   # 82M params, same GPT-2 tokenizer as the browser
MAX_LEN = 128
EPOCHS = 8
BATCH_SIZE = 16
LEARNING_RATE = 5e-5        # small LR: we're nudging a pretrained model
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PATIENCE = 2                # fine-tuning overfits fast; stop early
WARMUP_STEPS = 100
LOG_EVERY = 50


class JokeDataset(Dataset):
    """JSONL jokes -> (input_ids, labels) for HF causal LM fine-tuning.

    Note: GPT2LMHeadModel shifts labels internally, so labels are NOT
    pre-shifted here — they mirror input_ids, with -100 on padding.
    """

    def __init__(self, jsonl_path, tokenizer, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []

        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "text" in data:
                    text = data["text"]
                elif "messages" in data:
                    for msg in data["messages"]:
                        if msg.get("role") == "assistant":
                            text = msg["content"]
                            break
                    else:
                        continue
                else:
                    continue
                self.examples.append(text)

        print(f"  Loaded {len(self.examples)} jokes from {jsonl_path}", flush=True)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        eos_id = self.tokenizer.eos_token_id
        encoded = self.tokenizer.encode(self.examples[idx]) + [eos_id]

        if len(encoded) > self.max_len:
            encoded = encoded[: self.max_len - 1] + [eos_id]

        pad_len = self.max_len - len(encoded)
        input_ids = encoded + [eos_id] * pad_len
        labels = encoded + [-100] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)", flush=True)
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using CUDA GPU", flush=True)
        return torch.device("cuda")
    print("Using CPU (this will be slow)", flush=True)
    return torch.device("cpu")


def validate(model, val_loader, device):
    model.train(False)
    total, n = 0.0, 0
    with torch.no_grad():
        for input_ids, labels in val_loader:
            out = model(input_ids.to(device), labels=labels.to(device))
            total += out.loss.item()
            n += 1
    model.train(True)
    return total / max(n, 1)


def train():
    print("=" * 70, flush=True)
    print(f"Dad Joke Fine-Tuning — base model: {BASE_MODEL}", flush=True)
    print("=" * 70, flush=True)

    device = get_device()

    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}", flush=True)

    train_ds = JokeDataset(TRAINING_DATA, tokenizer)
    val_ds = JokeDataset(VALIDATION_DATA, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = float("inf")
    patience_counter = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nTraining config:", flush=True)
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}", flush=True)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Steps/epoch: {len(train_loader)}", flush=True)
    print("=" * 70 + "\n", flush=True)

    start = time.time()

    for epoch in range(EPOCHS):
        model.train(True)
        epoch_loss, n_batches = 0.0, 0

        for step, (input_ids, labels) in enumerate(train_loader):
            out = model(input_ids.to(device), labels=labels.to(device))
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

            global_step = epoch * len(train_loader) + step + 1
            if global_step % LOG_EVERY == 0:
                print(
                    f"  Step {global_step}/{total_steps} | Loss: {epoch_loss / n_batches:.4f} "
                    f"| LR: {scheduler.get_last_lr()[0]:.6f} | Time: {time.time() - start:.0f}s",
                    flush=True,
                )

        val_loss = validate(model, val_loader, device)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}", flush=True)
        print(f"  Train loss: {epoch_loss / n_batches:.4f}", flush=True)
        print(f"  Val loss:   {val_loss:.4f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "base_model": BASE_MODEL,
                    "tokenizer_name": BASE_MODEL,
                },
                CHECKPOINT,
            )
            print(f"  Saved best model (val_loss: {val_loss:.4f})", flush=True)
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})", flush=True)
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping after {epoch + 1} epochs", flush=True)
                break

    print("=" * 70, flush=True)
    print("Fine-tuning complete!", flush=True)
    print(f"  Total time: {(time.time() - start) / 60:.1f} minutes", flush=True)
    print(f"  Best val loss: {best_val:.4f}", flush=True)
    print(f"  Checkpoint: {CHECKPOINT}", flush=True)
    print("\nNext: python scripts/export_finetuned_to_onnx.py", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    train()
