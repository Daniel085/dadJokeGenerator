#!/usr/bin/env python3
"""
Prepare the fine-tuned model for WebLLM deployment

This script documents the steps needed to convert the model to WebLLM format.
Actual conversion requires MLC-LLM tools which should be installed separately.

Installation:
    pip install mlc-llm mlc-ai-nightly

Documentation:
    https://llm.mlc.ai/docs/compilation/compile_models.html
"""

import os
import json
from pathlib import Path


def check_requirements():
    """Check if required tools are installed"""
    print("🔍 Checking requirements...\n")

    try:
        import mlc_llm
        print("✅ mlc-llm installed")
    except ImportError:
        print("❌ mlc-llm not installed")
        print("   Install: pip install mlc-llm mlc-ai-nightly")
        return False

    return True


def create_mlc_config(model_path, output_path):
    """Create MLC configuration for the model"""
    config = {
        "model_type": "llama",
        "quantization": "q4f16_1",
        "model_name": "dad-joke-tinyllama",
        "conv_template": "chatml",
        "context_window_size": 2048,
        "prefill_chunk_size": 2048,
        "tensor_parallel_shards": 1,
        "temperature": 0.7,
        "top_p": 0.9,
        "mean_gen_len": 128,
        "max_gen_len": 256,
        "shift_fill_factor": 0.3
    }

    config_path = Path(output_path) / "mlc-chat-config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Created MLC config: {config_path}")
    return str(config_path)


def main():
    print("=" * 80)
    print("🚀 WebLLM Preparation Guide")
    print("=" * 80 + "\n")

    model_path = Path("./dad-joke-model/final")
    output_path = Path("./dad-joke-model-webllm")

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("\nPlease train the model first:")
        print("  python scripts/train_dad_joke_model.py")
        return

    print(f"📁 Input model: {model_path}")
    print(f"📁 Output directory: {output_path}\n")

    # Check requirements
    if not check_requirements():
        print("\n⚠️  Install required tools first, then re-run this script.")
        return

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("STEP 1: Convert Model Weights")
    print("=" * 80 + "\n")

    print("Run this command:")
    print(f"""
mlc_llm convert_weight \\
    {model_path} \\
    --quantization q4f16_1 \\
    --output {output_path}
    """)

    print("\n" + "=" * 80)
    print("STEP 2: Compile for WebGPU")
    print("=" * 80 + "\n")

    # Create MLC config
    config_path = create_mlc_config(model_path, output_path)

    print("\nRun this command:")
    print(f"""
mlc_llm compile \\
    {output_path} \\
    --target webgpu \\
    --quantization q4f16_1 \\
    --use-cache 0
    """)

    print("\n" + "=" * 80)
    print("STEP 3: Test Locally")
    print("=" * 80 + "\n")

    print("Test the compiled model:")
    print(f"""
mlc_llm chat \\
    {output_path} \\
    --model-lib {output_path}/dad-joke-tinyllama-webgpu.wasm
    """)

    print("\n" + "=" * 80)
    print("STEP 4: Prepare for Upload")
    print("=" * 80 + "\n")

    print("Expected file structure for Hugging Face:")
    print(f"""
{output_path}/
├── params_shard_*.bin        # Model weights (~600MB)
├── ndarray-cache.json         # Weight manifest
├── mlc-chat-config.json       # Configuration
├── tokenizer.json             # Tokenizer
├── tokenizer_config.json      # Tokenizer config
├── *.wasm                     # WebGPU runtime
└── MODEL_CARD.md              # Documentation
    """)

    print("\n" + "=" * 80)
    print("STEP 5: Upload to Hugging Face")
    print("=" * 80 + "\n")

    print("Commands:")
    print(f"""
# Login
huggingface-cli login

# Create repository
huggingface-cli repo create dad-joke-tinyllama-webllm --type model

# Upload
cd {output_path}
huggingface-cli upload \\
    YOUR_USERNAME/dad-joke-tinyllama-webllm \\
    . \\
    --repo-type model \\
    --commit-message "Add WebLLM-compatible dad joke model"
    """)

    print("\n" + "=" * 80)
    print("STEP 6: Integration with Web App")
    print("=" * 80 + "\n")

    print("Update index.html:")
    print("""
// Change model ID
const MODEL_ID = "YOUR_USERNAME/dad-joke-tinyllama-webllm-q4f16_1-MLC";

// Or use full URL
const customModel = {
    model: "YOUR_USERNAME/dad-joke-tinyllama-webllm-q4f16_1-MLC",
    model_id: "dad-joke-tinyllama",
    model_lib: "https://huggingface.co/YOUR_USERNAME/dad-joke-tinyllama-webllm/resolve/main/dad-joke-tinyllama-webgpu.wasm"
};
    """)

    print("\n✅ Ready for WebLLM conversion!")
    print("\nFollow the steps above to convert and deploy your model.")
    print("=" * 80)


if __name__ == "__main__":
    main()
