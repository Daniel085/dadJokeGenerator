# Training Scripts - DadJokeTransformer

Scripts for training, evaluating, and exporting a custom ~25M parameter transformer that generates dad jokes.

## Prerequisites

- Python 3.10+
- Mac mini (M1/M2/M3/M4) recommended, or any machine with 8GB+ RAM

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r scripts/requirements.txt

# Verify PyTorch + MPS
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

---

## Scripts

### `model.py` - Model Architecture

Defines the DadJokeTransformer: a decoder-only transformer (~25M parameters) with:

- 6 transformer layers, 8 attention heads, 512 hidden dim
- Pre-norm architecture (LayerNorm before attention/FFN)
- Weight tying between token embedding and output head
- GPT-2 tokenizer (50,257 vocab)

```bash
# Verify model builds correctly
python3 -c "from scripts.model import DadJokeTransformer, DadJokeConfig; m = DadJokeTransformer(DadJokeConfig()); print(f'Params: {sum(p.numel() for p in m.parameters()):,}')"
```

### `joke_validator.py` - Quality Validation

Validates generated jokes against quality criteria:

- Format check (Q: ... ? A: ...)
- Length check (20-200 characters)
- Profanity filter
- Wordplay/pun detection
- Meta-commentary removal

```bash
python3 scripts/joke_validator.py
```

### `train_from_scratch.py` - Training

Trains the DadJokeTransformer on curated dad jokes.

```bash
python3 scripts/train_from_scratch.py
```

**Configuration** (edit at top of script):
- Epochs: 20 (with early stopping, patience=5)
- Batch size: 32
- Learning rate: 3e-4 (cosine annealing)
- Gradient clipping: 1.0

**Input:** `training_data/dad_jokes_train.jsonl` and `training_data/dad_jokes_validation.jsonl`

**Output:** `dad-joke-model/best_model.pt`

**Estimated time:** 15-30 minutes on M4 Mac Mini

### `evaluate_custom_model.py` - Evaluation

Generates N jokes and measures validation pass rate.

```bash
# Default: 100 samples from dad-joke-model/best_model.pt
python3 scripts/evaluate_custom_model.py

# Custom checkpoint and sample count
python3 scripts/evaluate_custom_model.py dad-joke-model/best_model.pt 200
```

**Target:** 80%+ validation pass rate

### `export_to_onnx.py` - ONNX Export

Exports trained model to ONNX format for browser deployment.

```bash
# Default paths
python3 scripts/export_to_onnx.py

# Custom paths
python3 scripts/export_to_onnx.py dad-joke-model/best_model.pt dad-joke-model/model.onnx
```

**What it does:**
1. Loads PyTorch checkpoint
2. Exports to ONNX (opset 17, dynamic axes)
3. Quantizes to uint8 (reduces size ~60-75%)
4. Verifies with ONNX Runtime

**Output:** `dad-joke-model/dad_joke_model.onnx` and `dad-joke-model/dad_joke_model_quantized.onnx`

---

## Complete Workflow

```bash
# 1. Train the model
python3 scripts/train_from_scratch.py

# 2. Evaluate quality
python3 scripts/evaluate_custom_model.py

# 3. Export to ONNX
python3 scripts/export_to_onnx.py

# 4. Copy quantized model to web directory
cp dad-joke-model/dad_joke_model_quantized.onnx model/dad_joke_model.onnx

# 5. Test in browser
python3 -m http.server 8000
# Visit http://localhost:8000 and select AI mode
```

---

## Training Data

Located in `training_data/`:

| File | Description |
|------|-------------|
| `all_jokes.jsonl` | 4,566 unique dad jokes |
| `dad_jokes_train.jsonl` | Training split (90%, 4,109 jokes) |
| `dad_jokes_validation.jsonl` | Validation split (10%, 457 jokes) |
| `batch_*.json` | Raw generated batches |
| `validated_batch_*.jsonl` | Validated batches in JSONL format |

**Format:** Each line is `{"text": "Q: Why...? A: Because...!"}`

---

## Troubleshooting

**MPS not available:** Reinstall PyTorch: `pip install torch torchvision torchaudio`

**Out of memory:** Reduce `BATCH_SIZE` to 16 in `train_from_scratch.py`

**Low pass rate (<80%):** Try more training data, more epochs, or lower learning rate

**ONNX export fails:** Ensure `onnx` and `onnxruntime` are installed: `pip install onnx onnxruntime`
