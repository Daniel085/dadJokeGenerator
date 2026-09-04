#!/usr/bin/env python3
"""
LoRA fine-tune Qwen2.5-0.5B on dad jokes (v2 data).

Why LoRA: a full fine-tune of a 0.5B model needs ~8 GB for weights, grads
and Adam state before activations, which is too tight on a 16 GB Mac. LoRA
trains ~1% of the parameters and fits comfortably; the adapter is merged
back into the base weights at the end so export sees a plain model.

Usage:
    python scripts/prepare_data_v2.py     # once
    python scripts/finetune_qwen.py [--base Qwen/Qwen2.5-0.5B] [--data v2] [--name qwen]
                                    [--batch 8] [--epochs 4]

    e.g. the 1.5B model on the judge-filtered v3 data:
    python scripts/finetune_qwen.py --base Qwen/Qwen2.5-1.5B --data v3 --name qwen15 --batch 4

Outputs:
    dad-joke-model/<name>_finetuned_hf/   merged model + tokenizer (for export)
    dad-joke-model/<name>_lora_adapter/   the adapter alone (small, for reference)
"""

import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

_ap = argparse.ArgumentParser()
_ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
_ap.add_argument("--data", default="v2", help="training_data/<data>_train.jsonl / _validation.jsonl")
_ap.add_argument("--name", default="qwen", help="output folder prefix under dad-joke-model/")
_ap.add_argument("--batch", type=int, default=8)
_ap.add_argument("--epochs", type=int, default=4)
ARGS = _ap.parse_args()

BASE_MODEL = ARGS.base
TRAIN = f"training_data/{ARGS.data}_train.jsonl"
VAL = f"training_data/{ARGS.data}_validation.jsonl"
OUT_DIR = "dad-joke-model"
MERGED_DIR = os.path.join(OUT_DIR, f"{ARGS.name}_finetuned_hf")
ADAPTER_DIR = os.path.join(OUT_DIR, f"{ARGS.name}_lora_adapter")

MAX_LEN = 80          # jokes are ~25-40 Qwen tokens; 80 leaves headroom
EPOCHS = ARGS.epochs
BATCH_SIZE = ARGS.batch
LEARNING_RATE = 2e-4  # LoRA uses a higher LR than full fine-tuning
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
PATIENCE = 2
WARMUP_STEPS = 50
LOG_EVERY = 50

LORA = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)


class JokeDataset(Dataset):
    """JSONL {"text": "Q: ... A: ..."} -> (input_ids, labels); labels mirror
    input_ids with -100 on padding (HF shifts internally)."""

    def __init__(self, path, tokenizer, max_len=MAX_LEN):
        self.tok = tokenizer
        self.max_len = max_len
        self.texts = [json.loads(l)["text"] for l in open(path) if l.strip()]
        print(f"  Loaded {len(self.texts)} jokes from {path}", flush=True)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        eos = self.tok.eos_token_id
        ids = self.tok.encode(self.texts[i]) + [eos]
        if len(ids) > self.max_len:
            ids = ids[: self.max_len - 1] + [eos]
        pad = self.max_len - len(ids)
        return (
            torch.tensor(ids + [eos] * pad, dtype=torch.long),
            torch.tensor(ids + [-100] * pad, dtype=torch.long),
        )


def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)", flush=True)
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("Using CPU (this will be slow)", flush=True)
    return torch.device("cpu")


def validate(model, loader, device):
    model.train(False)
    total, n = 0.0, 0
    with torch.no_grad():
        for ids, labels in loader:
            total += model(input_ids=ids.to(device), labels=labels.to(device)).loss.item()
            n += 1
    model.train(True)
    return total / max(n, 1)


def train():
    print("=" * 70, flush=True)
    print(f"LoRA fine-tuning {BASE_MODEL} on dad jokes (v2 data)", flush=True)
    print("=" * 70, flush=True)
    device = get_device()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    model = get_peft_model(model, LORA).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    train_ds, val_ds = JokeDataset(TRAIN, tok), JokeDataset(VAL, tok)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        p = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f"\n  Epochs {EPOCHS} | Batch {BATCH_SIZE} | LR {LEARNING_RATE} | "
          f"Steps/epoch {len(train_loader)} | Total {total_steps}", flush=True)
    print("=" * 70 + "\n", flush=True)

    best, bad_epochs, start = float("inf"), 0, time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train(True)
        run_loss, n = 0.0, 0
        for step, (ids, labels) in enumerate(train_loader):
            loss = model(input_ids=ids.to(device), labels=labels.to(device)).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            run_loss += loss.item()
            n += 1
            g = epoch * len(train_loader) + step + 1
            if g % LOG_EVERY == 0:
                print(f"  Step {g}/{total_steps} | Loss {run_loss / n:.4f} | "
                      f"LR {sched.get_last_lr()[0]:.6f} | {time.time() - start:.0f}s", flush=True)

        val = validate(model, val_loader, device)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}  train {run_loss / n:.4f}  val {val:.4f}", flush=True)

        if val < best:
            best, bad_epochs = val, 0
            model.save_pretrained(ADAPTER_DIR)
            print(f"  Saved adapter (val {val:.4f})", flush=True)
        else:
            bad_epochs += 1
            print(f"  No improvement ({bad_epochs}/{PATIENCE})", flush=True)
            if bad_epochs >= PATIENCE:
                print("\nEarly stopping", flush=True)
                break

    # Merge the best adapter into the base weights for export
    print("\nMerging best adapter into base model...", flush=True)
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload()
    merged.save_pretrained(MERGED_DIR)
    tok.save_pretrained(MERGED_DIR)
    with open(os.path.join(MERGED_DIR, "training_summary.json"), "w") as f:
        json.dump({"base_model": BASE_MODEL, "best_val_loss": best,
                   "minutes": round((time.time() - start) / 60, 1)}, f, indent=2)

    print("=" * 70, flush=True)
    print(f"Done in {(time.time() - start) / 60:.1f} min | best val loss {best:.4f}", flush=True)
    print(f"Merged model: {MERGED_DIR}", flush=True)
    print(f"Next: python scripts/export_qwen_to_onnx.py {MERGED_DIR} dad-joke-model/{ARGS.name}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    train()
