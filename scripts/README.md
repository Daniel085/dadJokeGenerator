# 🛠️ Training Scripts Documentation

This directory contains all the scripts needed to fine-tune a custom dad joke model and deploy it to Hugging Face.

## 📋 Overview

The training pipeline consists of 6 main phases:

1. **Data Generation** - Generate training jokes using Claude
2. **Environment Setup** - Install dependencies on Mac mini
3. **Model Training** - Fine-tune TinyLlama with LoRA
4. **Evaluation** - Validate model quality
5. **WebLLM Conversion** - Prepare for browser deployment
6. **Deployment** - Upload to Hugging Face

---

## 🚀 Quick Start

### Prerequisites

- Mac mini (M1/M2/M3 recommended) or any machine with 8GB+ RAM
- Python 3.10+
- 50GB free disk space
- Internet connection for initial downloads

### Installation

```bash
# 1. Navigate to project root
cd ~/dadJokeGenerator

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r scripts/requirements.txt

# 4. Verify installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

Expected output:
```
PyTorch: 2.1.0+
MPS: True
```

---

## 📝 Script Reference

### 1. `joke_validator.py`

**Purpose:** Validate dad jokes against quality criteria

**Usage:**
```bash
# Test the validator
python3 scripts/joke_validator.py
```

**What it does:**
- Checks joke format (Q: A: structure)
- Validates length (20-200 characters)
- Filters profanity
- Detects wordplay/puns
- Removes meta-commentary

**Import in other scripts:**
```python
from joke_validator import JokeValidator

validator = JokeValidator()
is_valid, failures = validator.validate("Q: Why? A: Because!")
```

---

### 2. `generate_training_data.py`

**Purpose:** Generate and validate training data using Claude

**Usage:**

**Step 1: See the generation workflow**
```bash
python3 scripts/generate_training_data.py
```

This displays the prompt to use with Claude for joke generation.

**Step 2: Generate jokes with Claude**
- Copy the prompt shown by the script
- Paste into Claude conversation
- Claude will generate 100 jokes as JSON array
- Save response to `batch_001.json`

**Step 3: Validate the batch**
```bash
python3 scripts/generate_training_data.py --validate batch_001.json
```

Output:
```
📊 Validation Results for batch_001.json
Total: 100
✅ Valid: 87 (87.0%)
❌ Invalid: 13

✅ Saved 87 jokes to training_data/validated_batch_001.jsonl
```

**Step 4: Repeat**
- Generate multiple batches (batch_002.json, batch_003.json, etc.)
- Aim for 10,000-50,000 total jokes
- Use different few-shot examples each time for diversity

**Step 5: Combine and split**
```bash
# Combine all validated batches
cat training_data/validated_*.jsonl > training_data/all_jokes.jsonl

# Split into train/validation (90/10)
total=$(wc -l < training_data/all_jokes.jsonl)
train_count=$((total * 9 / 10))

head -n $train_count training_data/all_jokes.jsonl > training_data/dad_jokes_train.jsonl
tail -n +$((train_count + 1)) training_data/all_jokes.jsonl > training_data/dad_jokes_validation.jsonl

echo "✅ Training set: $(wc -l < training_data/dad_jokes_train.jsonl) jokes"
echo "✅ Validation set: $(wc -l < training_data/dad_jokes_validation.jsonl) jokes"
```

---

### 3. `train_dad_joke_model.py`

**Purpose:** Fine-tune TinyLlama on dad jokes using LoRA

**Usage:**
```bash
python3 scripts/train_dad_joke_model.py
```

**What it does:**
1. Loads TinyLlama-1.1B-Chat-v1.0 base model
2. Configures LoRA adapters (trainable: 0.4% of params)
3. Loads training data from `training_data/dad_jokes_train.jsonl`
4. Trains for 3 epochs
5. Saves fine-tuned model to `./dad-joke-model/final`

**Expected output:**
```
🎓 Dad Joke Model Training
============================================================
🎮 Using MPS (Apple Silicon GPU)
📚 Loading tokenizer...
🧠 Loading base model...
🔧 Configuring LoRA...
   Trainable params: 4,718,592
   All params: 1,104,718,592
   Trainable: 0.43%

🎓 Starting training...
============================================================
Epoch 1/3: 100%|████████| 625/625 [45:23<00:00, Loss: 1.234]
Epoch 2/3: 100%|████████| 625/625 [43:12<00:00, Loss: 0.876]
Epoch 3/3: 100%|████████| 625/625 [42:48<00:00, Loss: 0.654]

✅ Training complete!
📁 Model saved to: ./dad-joke-model/final
```

**Estimated training time:**
- M1 Mac mini: 2-3 hours
- M2 Mac mini: 1.5-2.5 hours
- Intel Mac mini: 8-12 hours

**Troubleshooting:**
- **Out of memory:** Reduce `BATCH_SIZE` from 4 to 2
- **Too slow:** Reduce dataset size or epochs
- **Poor quality:** Increase epochs to 5 or add more training data

---

### 4. `test_model.py`

**Purpose:** Interactively test the fine-tuned model

**Usage:**
```bash
python3 scripts/test_model.py
```

**What it does:**
- Loads the fine-tuned model
- Interactive prompt for joke generation
- Manual quality assessment

**Example session:**
```
🤖 Loading dad joke model...
✅ Model loaded!

Interactive Dad Joke Generator
============================================================

🎤 Topic (or Enter for random): computers

🎨 Generating joke...

─────────────────────────────────────────────────────
Q: Why do programmers prefer dark mode?
A: Because light attracts bugs!
─────────────────────────────────────────────────────

🎤 Topic (or Enter for random): [Enter]

🎨 Generating joke...

─────────────────────────────────────────────────────
Q: What do you call a bear with no teeth?
A: A gummy bear!
─────────────────────────────────────────────────────
```

---

### 5. `evaluate_model.py`

**Purpose:** Quantitative evaluation of model quality

**Usage:**
```bash
# Evaluate with 100 samples (default)
python3 scripts/evaluate_model.py

# Evaluate with custom parameters
python3 scripts/evaluate_model.py ./dad-joke-model/final 200 0.7 results.json
```

**Arguments:**
1. Model path (default: `./dad-joke-model/final`)
2. Number of samples (default: 100)
3. Temperature (default: 0.7)
4. Output file (optional)

**What it does:**
1. Generates N jokes with the model
2. Validates each joke with JokeValidator
3. Calculates pass rate
4. Shows sample outputs and failure analysis

**Example output:**
```
📊 EVALUATION RESULTS
============================================================
Total jokes generated: 100
✅ Passed validation: 96 (96.0%)
❌ Failed validation: 4 (4.0%)

📋 Failure Breakdown:
  - Missing wordplay setup: 2 (50.0% of failures)
  - Invalid length: 1 (25.0% of failures)
  - Missing Q&A format: 1 (25.0% of failures)

✅ Sample VALID jokes:
============================================================
1. Q: Why don't scientists trust atoms?
   A: Because they make up everything!

2. Q: What do you call a factory that sells good products?
   A: A satisfactory!

📈 QUALITY ASSESSMENT
============================================================
🎉 EXCELLENT! Model quality is production-ready.
   ✓ Pass rate exceeds 95% threshold
   ✓ Ready for deployment to Hugging Face
```

**Success criteria:**
- **≥95%:** Excellent, ready for deployment
- **85-94%:** Good, acceptable quality
- **<85%:** Insufficient, needs more training

---

### 6. `prepare_for_webllm.py`

**Purpose:** Guide for converting model to WebLLM format

**Usage:**
```bash
python3 scripts/prepare_for_webllm.py
```

**What it does:**
- Checks if MLC-LLM is installed
- Shows step-by-step conversion commands
- Creates MLC configuration
- Provides upload instructions

**Steps shown:**

1. **Convert weights to 4-bit quantization**
2. **Compile for WebGPU**
3. **Test locally**
4. **Upload to Hugging Face**
5. **Integration instructions**

**Note:** Actual conversion requires MLC-LLM tools:
```bash
pip install mlc-llm mlc-ai-nightly
```

---

## 📊 Complete Workflow

### Phase 1: Data Generation (2-3 hours)

```bash
# 1. Generate first batch with Claude
python3 scripts/generate_training_data.py
# Copy prompt, use with Claude, save to batch_001.json

# 2. Validate batch
python3 scripts/generate_training_data.py --validate batch_001.json

# 3. Repeat for ~100 batches (10,000+ jokes)
# Generate batch_002.json, batch_003.json, etc.

# 4. Combine all validated jokes
cat training_data/validated_*.jsonl > training_data/all_jokes.jsonl

# 5. Split into train/validation
total=$(wc -l < training_data/all_jokes.jsonl)
train_count=$((total * 9 / 10))
head -n $train_count training_data/all_jokes.jsonl > training_data/dad_jokes_train.jsonl
tail -n +$((train_count + 1)) training_data/all_jokes.jsonl > training_data/dad_jokes_validation.jsonl
```

### Phase 2: Training (2-4 hours)

```bash
# Train the model
python3 scripts/train_dad_joke_model.py

# Expected: 2-4 hours depending on hardware
# Output: ./dad-joke-model/final/
```

### Phase 3: Evaluation (10-15 minutes)

```bash
# Quick test (interactive)
python3 scripts/test_model.py

# Quantitative evaluation
python3 scripts/evaluate_model.py ./dad-joke-model/final 100 0.7 evaluation_results.json

# Check pass rate - should be ≥95%
```

### Phase 4: WebLLM Conversion (30 minutes)

```bash
# Install MLC-LLM
pip install mlc-llm mlc-ai-nightly

# Run conversion guide
python3 scripts/prepare_for_webllm.py

# Follow the displayed steps to convert and test
```

### Phase 5: Deployment (15 minutes)

```bash
# Login to Hugging Face
huggingface-cli login

# Create repository
huggingface-cli repo create dad-joke-tinyllama-webllm --type model

# Upload model
cd dad-joke-model-webllm
huggingface-cli upload YOUR_USERNAME/dad-joke-tinyllama-webllm . --repo-type model
```

### Phase 6: Integration (30 minutes)

Update `index.html`:
```javascript
// Change model ID to your custom model
const MODEL_ID = "YOUR_USERNAME/dad-joke-tinyllama-webllm-q4f16_1-MLC";
```

Test on GitHub Pages and verify quality!

---

## 🐛 Troubleshooting

### Issue: "MPS not available"

**Solution:**
```bash
# Check PyTorch installation
python3 -c "import torch; print(torch.__version__)"

# Reinstall PyTorch with MPS support
pip3 uninstall torch
pip3 install torch torchvision torchaudio
```

### Issue: "Out of memory during training"

**Solution:** Reduce batch size in `train_dad_joke_model.py`:
```python
BATCH_SIZE = 2  # Instead of 4
GRADIENT_ACCUMULATION = 8  # Instead of 4
```

### Issue: "Low validation pass rate (<85%)"

**Possible causes:**
1. Insufficient training data → Generate more jokes
2. Too few epochs → Increase to 5 epochs
3. Poor quality training data → Review rejected samples
4. Need better prompts → Refine Claude generation prompts

**Solution:**
```bash
# Review rejected jokes for patterns
python3 -c "
import json
with open('training_data/rejected.jsonl') as f:
    for line in f:
        print(json.loads(line))
"

# Train for more epochs
# Edit train_dad_joke_model.py: EPOCHS = 5
python3 scripts/train_dad_joke_model.py
```

### Issue: "Model generates gibberish"

**Possible causes:**
1. Training diverged → Use lower learning rate
2. Overfitting → Use more training data
3. Wrong model format → Check tokenizer

**Solution:**
```python
# In train_dad_joke_model.py, reduce learning rate:
LEARNING_RATE = 1e-4  # Instead of 2e-4
```

---

## 📚 Additional Resources

- [TinyLlama Model Card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Training](https://huggingface.co/docs/transformers/training)
- [MLC-LLM Docs](https://llm.mlc.ai/docs/)
- [WebLLM GitHub](https://github.com/mlc-ai/web-llm)

---

## ❓ FAQ

**Q: How long does the entire process take?**
A: 8-12 hours total (2-3 hours data generation, 2-4 hours training, 1-2 hours conversion, rest is setup/testing)

**Q: Can I train on Linux or Windows?**
A: Yes! The scripts work on any platform with Python 3.10+. GPU acceleration varies (CUDA on Linux/Windows, MPS on macOS)

**Q: How much does this cost?**
A: $0 if using Mac mini locally + Hugging Face free tier. All tools are free and open source.

**Q: What if I don't have a Mac?**
A: Any computer with 8GB+ RAM works. Intel Macs use CPU (slower). Linux/Windows can use CUDA GPU if available.

**Q: Can I use a different base model?**
A: Yes! Edit `MODEL_NAME` in `train_dad_joke_model.py`. Other options: Llama-3.2-1B, Phi-3.5-mini, etc.

**Q: How do I improve quality beyond 95%?**
A: More training data (50K+ jokes), train longer (5+ epochs), manual review and filtering, hyperparameter tuning

---

**Last Updated:** 2025-11-20
**Maintainer:** Daniel (@Daniel085)
