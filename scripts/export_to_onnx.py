#!/usr/bin/env python3
"""
Export trained custom transformer to ONNX for browser deployment.

Usage:
    python scripts/export_to_onnx.py [checkpoint_path] [output_path]

Examples:
    python scripts/export_to_onnx.py
    python scripts/export_to_onnx.py dad-joke-model/best_model.pt dad-joke-model/model.onnx
"""

import sys
import os
import torch
import onnx
from pathlib import Path

from model import DadJokeTransformer, DadJokeConfig


DEFAULT_CHECKPOINT = "dad-joke-model/best_model.pt"
DEFAULT_OUTPUT = "dad-joke-model/dad_joke_model.onnx"


class ExportWrapper(torch.nn.Module):
    """Wrapper that returns only logits (no loss) for ONNX export"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        logits, _ = self.model(input_ids, targets=None)
        return logits


def export_to_onnx(checkpoint_path, output_path):
    """Export model to ONNX format"""
    print("=" * 70)
    print("ONNX Export - Dad Joke Transformer")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    config = DadJokeConfig()
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        for k, v in saved_config.items():
            if hasattr(config, k):
                setattr(config, k, v)

    model = DadJokeTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(False)

    # Wrap for export (logits only)
    wrapper = ExportWrapper(model)

    # Create output directory
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Dummy input
    dummy_input = torch.randint(0, config.vocab_size, (1, 64))

    # Export
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq_len'},
            'logits': {0: 'batch', 1: 'seq_len'}
        },
        opset_version=17,
        do_constant_folding=True
    )

    # Verify
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Model size: {file_size_mb:.1f} MB")

    # Quantize
    quantized_path = output_path.replace('.onnx', '_quantized.onnx')
    print(f"\nQuantizing to {quantized_path}...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            output_path,
            quantized_path,
            weight_type=QuantType.QUInt8
        )
        q_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
        print(f"  Quantized size: {q_size_mb:.1f} MB ({q_size_mb / file_size_mb * 100:.0f}% of original)")
    except ImportError:
        print("  Skipping quantization (onnxruntime.quantization not available)")
        quantized_path = None

    # Test inference
    print("\nTesting ONNX inference...")
    try:
        import onnxruntime as ort
        import numpy as np

        test_path = quantized_path if quantized_path and os.path.exists(quantized_path) else output_path
        sess = ort.InferenceSession(test_path)
        test_input = np.random.randint(0, config.vocab_size, (1, 10)).astype(np.int64)
        result = sess.run(None, {'input_ids': test_input})
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {result[0].shape}")
        print("  Inference OK!")
    except ImportError:
        print("  Skipping test (onnxruntime not available)")

    print("\n" + "=" * 70)
    print("Export complete!")
    print(f"  Full model:  {output_path} ({file_size_mb:.1f} MB)")
    if quantized_path and os.path.exists(quantized_path):
        print(f"  Quantized:   {quantized_path} ({q_size_mb:.1f} MB)")
    print("\nNext: copy the model to model/ and update index.html")
    print("=" * 70)


def main():
    checkpoint_path = DEFAULT_CHECKPOINT
    output_path = DEFAULT_OUTPUT

    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    if not Path(checkpoint_path).exists():
        print(f"Model not found: {checkpoint_path}")
        print("\nTrain first: python scripts/train_from_scratch.py")
        sys.exit(1)

    export_to_onnx(checkpoint_path, output_path)


if __name__ == "__main__":
    main()
