#!/usr/bin/env python3
"""
Export the fine-tuned DistilGPT-2 dad joke model to ONNX for the browser.

Uses HuggingFace Optimum for the export: torch.onnx.export (both the dynamo
and legacy exporters) fails on transformers>=4.5x GPT-2 graphs, while
Optimum maintains working export configs for them.

The exported graph takes input_ids, attention_mask and position_ids and
returns logits (GPT-2 vocab, 50257) — matching what index.html feeds it.

Usage:
    python scripts/finetune_gpt2.py            # first, produce the checkpoint
    python scripts/export_finetuned_to_onnx.py
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from optimum.exporters.onnx import main_export
from onnxruntime.quantization import quantize_dynamic, QuantType

CHECKPOINT = "dad-joke-model/finetuned_best.pt"
QUANTIZED_OUTPUT = "dad-joke-model/dad_joke_finetuned_quantized.onnx"


def main():
    if not Path(CHECKPOINT).exists():
        print(f"Checkpoint not found: {CHECKPOINT}")
        print("Train first: python scripts/finetune_gpt2.py")
        sys.exit(1)

    print("=" * 70)
    print("ONNX Export - Fine-tuned Dad Joke Model (via Optimum)")
    print("=" * 70)

    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    base = ckpt.get("base_model", "distilgpt2")
    print(f"\nLoading {base} and applying fine-tuned weights "
          f"(epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})...")

    model = GPT2LMHeadModel.from_pretrained(base)
    model.load_state_dict(ckpt["model_state_dict"])

    with tempfile.TemporaryDirectory() as tmp:
        hf_dir = os.path.join(tmp, "hf")
        onnx_dir = os.path.join(tmp, "onnx")

        model.save_pretrained(hf_dir)
        GPT2Tokenizer.from_pretrained(base).save_pretrained(hf_dir)

        print("Exporting with Optimum (task: text-generation)...")
        main_export(hf_dir, output=onnx_dir, task="text-generation")

        full = os.path.join(onnx_dir, "model.onnx")
        full_mb = os.path.getsize(full) / (1024 * 1024)
        print(f"  Full model size: {full_mb:.1f} MB")

        print(f"\nQuantizing to {QUANTIZED_OUTPUT}...")
        os.makedirs(os.path.dirname(QUANTIZED_OUTPUT), exist_ok=True)
        quantize_dynamic(full, QUANTIZED_OUTPUT, weight_type=QuantType.QUInt8)

    q_mb = os.path.getsize(QUANTIZED_OUTPUT) / (1024 * 1024)
    print(f"  Quantized size: {q_mb:.1f} MB")

    print("\nTesting ONNX inference (quantized)...")
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(QUANTIZED_OUTPUT)
    n = 10
    feeds = {
        "input_ids": np.random.randint(0, 50257, (1, n)).astype(np.int64),
        "attention_mask": np.ones((1, n), dtype=np.int64),
        "position_ids": np.arange(n, dtype=np.int64)[None, :],
    }
    out = sess.run(None, feeds)
    print(f"  Output shape: {out[0].shape}")
    assert out[0].shape == (1, n, 50257)
    print("  Inference OK!")

    print("\n" + "=" * 70)
    print("Export complete!")
    print(f"  Quantized: {QUANTIZED_OUTPUT} ({q_mb:.1f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
