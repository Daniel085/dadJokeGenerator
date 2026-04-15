#!/usr/bin/env python3
"""
Train custom dad joke transformer from scratch on M4 Mac Mini.
Uses GPT-2 tokenizer + custom ~25M param transformer.

Usage:
    python scripts/train_from_scratch.py

Expects training data in:
    training_data/dad_jokes_train.jsonl
    training_data/dad_jokes_validation.jsonl
"""

import os
import json
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from pathlib import Path

from model import DadJokeTransformer, DadJokeConfig


# ─── Configuration ───────────────────────────────────────────────────────────

TRAINING_DATA = "training_data/dad_jokes_train.jsonl"
VALIDATION_DATA = "training_data/dad_jokes_validation.jsonl"
OUTPUT_DIR = "dad-joke-model"

EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PATIENCE = 5          # Early stopping patience (epochs)
LOG_EVERY = 50        # Log training loss every N steps


# ─── Dataset ─────────────────────────────────────────────────────────────────

class DadJokeDataset(Dataset):
    """
    Load dad jokes from JSONL, tokenize with GPT-2 tokenizer.

    Each line in the JSONL is either:
        {"text": "Q: ... A: ..."}
    or:
        {"messages": [..., {"role": "assistant", "content": "Q: ... A: ..."}]}
    """

    def __init__(self, jsonl_path, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []

        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                # Handle both formats
                if 'text' in data:
                    text = data['text']
                elif 'messages' in data:
                    # Extract assistant's response (the joke)
                    for msg in data['messages']:
                        if msg.get('role') == 'assistant':
                            text = msg['content']
                            break
                    else:
                        continue
                else:
                    continue

                self.examples.append(text)

        print(f"  Loaded {len(self.examples)} jokes from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]

        # Tokenize: add EOS token so model learns when to stop
        encoded = self.tokenizer.encode(text + self.tokenizer.eos_token)

        # Truncate if too long
        if len(encoded) > self.max_len:
            encoded = encoded[:self.max_len]

        # Pad if too short
        padding_len = self.max_len - len(encoded)
        input_ids = encoded + [self.tokenizer.pad_token_id] * padding_len

        # Targets: same as input shifted by 1 (handled by cross-entropy)
        # Use -100 for padding so loss ignores it
        targets = encoded[1:] + [-100] * (padding_len + 1)
        # Also mask the padding in input
        input_ids_tensor = torch.tensor(input_ids[:-1], dtype=torch.long)
        targets_tensor = torch.tensor(targets[:self.max_len - 1], dtype=torch.long)

        # Ensure same length
        if len(input_ids_tensor) < self.max_len - 1:
            pad = self.max_len - 1 - len(input_ids_tensor)
            input_ids_tensor = torch.cat([input_ids_tensor, torch.full((pad,), self.tokenizer.pad_token_id)])
            targets_tensor = torch.cat([targets_tensor, torch.full((pad,), -100)])

        return input_ids_tensor, targets_tensor


# ─── Training ────────────────────────────────────────────────────────────────

def get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU (this will be slow)")
        return torch.device("cpu")


def validate(model, val_loader, device):
    """Run validation and return average loss"""
    model.train(False)
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for input_ids, targets in val_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            _, loss = model(input_ids, targets)
            total_loss += loss.item()
            n_batches += 1

    model.train(True)
    return total_loss / max(n_batches, 1)


def train():
    """Main training loop"""
    print("=" * 70)
    print("Dad Joke Transformer - Training from Scratch")
    print("=" * 70)

    # Device
    device = get_device()

    # Tokenizer
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Model
    config = DadJokeConfig()
    model = DadJokeTransformer(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Data
    print("\nLoading training data...")
    train_dataset = DadJokeDataset(TRAINING_DATA, tokenizer, config.max_seq_len)
    val_dataset = DadJokeDataset(VALIDATION_DATA, tokenizer, config.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nTraining config:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Steps per epoch: {len(train_loader)}")
    print(f"  Total steps: {total_steps}")
    print("=" * 70 + "\n")

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for step, (input_ids, targets) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Forward
            _, loss = model(input_ids, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

            global_step = epoch * len(train_loader) + step + 1
            if global_step % LOG_EVERY == 0:
                avg = epoch_loss / n_batches
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(f"  Step {global_step}/{total_steps} | Loss: {avg:.4f} | LR: {lr:.6f} | Time: {elapsed:.0f}s")

        # Epoch complete
        avg_train_loss = epoch_loss / n_batches
        val_loss = validate(model, val_loader, device)
        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  Time:       {elapsed:.0f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': avg_train_loss,
                'config': config.to_dict(),
                'tokenizer_name': 'gpt2'
            }, checkpoint_path)
            print(f"  Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping after {epoch + 1} epochs (no improvement for {PATIENCE} epochs)")
            break

        print()

    # Done
    total_time = time.time() - start_time
    print("=" * 70)
    print("Training complete!")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {OUTPUT_DIR}/best_model.pt")
    print()
    print("Next steps:")
    print("  1. Test:     python scripts/evaluate_custom_model.py")
    print("  2. Export:   python scripts/export_to_onnx.py")
    print("=" * 70)


if __name__ == "__main__":
    train()
