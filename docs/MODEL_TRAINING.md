# Model Training: What Worked, What Didn't, and Why

This document records the two approaches tried for the in-browser AI joke
model, why the first one failed, and how to reproduce the one that works.

---

## TL;DR

| | v1: From scratch | v2: Fine-tuned (current) |
|---|---|---|
| Base | Random init, custom 8-layer transformer | Pretrained DistilGPT-2 |
| Parameters | 51M | 82M |
| Training data | 4,109 jokes (~150K tokens) | same |
| Best val loss | 3.02 | **2.05** |
| Perplexity | ~20.6 | **~7.7** |
| Training time (M4 Mac mini, MPS) | 38 min | 21 min |
| Quantized ONNX size | 49 MB | 115 MB |
| Sample punchline | "To reach the daily answers!" | "He was outstanding in his field!" |

**Lesson:** a model trained from scratch on a few thousand examples learns the
*shape* of a joke but not the language it's written in. Start from a pretrained
model whenever the domain is natural language.

---

## Why the from-scratch model failed

### The data was ~1000× too small for the job

A randomly-initialised language model knows nothing: not what "beach" means,
not that questions get answers, not which words sound alike. All of that has to
come from the training text.

GPT-2 learned English from roughly 40 GB of text (~10 billion tokens). Our
training set is 4,109 jokes, about 150,000 tokens. That's enough to memorise a
few thousand examples. It is nowhere near enough to learn what words mean.

### What it actually learned

Looking at generations from the from-scratch model, three layers are visible:

1. **The format** — learned perfectly. Every output was `Q: ... ? A: ... !`
   with a clean EOS stop. This pattern appears 4,109 times, so it's trivial.
2. **Local grammar** — learned partially. "Why did the tourist bring a map to
   the beach?" is a real sentence because "why did the X bring a Y to the Z"
   appears in dozens of training jokes and the model can fill the slots with
   nouns seen in similar positions.
3. **Meaning** — never learned. A punchline like "To reach the daily answers!"
   is assembled from tokens that commonly follow `A: To` in the training set.
   Nothing in 150K tokens can teach what a map is *for* or what would be funny
   about it.

### The loss curves prove it

From `train_from_scratch.py`, run on 2026-06-09:

```
Epoch  8   train 2.03   val 3.02   <- best
Epoch 12   train 1.45   val 3.18
Epoch 18   train 0.97   val 3.34   <- early stop
```

Train loss kept falling while validation loss *rose* after epoch 8. That gap
is the signature of memorisation: the model got very good at reproducing the
exact jokes it had seen and worse at anything new.

Every "improvement" we tried before switching approaches — deeper model (6→8
layers), more dropout, longer training, LR warmup, a fixed EOS tokenizer bug —
pushed on the wrong lever. They made memorisation more efficient rather than
adding understanding.

### Why fine-tuning worked immediately

DistilGPT-2 arrives already knowing English, common idioms, and even the
rough shape of a pun. The 4,109 jokes then only have to teach *style*: short,
Q/A, groan-worthy wordplay. That's a small adjustment to an existing skill
rather than building the skill from nothing.

It shows in the very first log line: after 50 steps the fine-tuned model was
at loss 3.84 — already close to the from-scratch model's best-ever score after
38 minutes.

### The general rule

Train from scratch only when you have hundreds of millions of tokens or more,
or when the domain is genuinely alien to existing models (protein sequences, a
novel programming language). For anything expressed in natural language, start
from a pretrained checkpoint.

---

## Reproducing the current model

All commands run from the repo root with the project venv active.

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r scripts/requirements.txt
```

### 2. Fine-tune

```bash
python scripts/finetune_gpt2.py
```

- Base model: `distilgpt2` (downloaded from the Hugging Face Hub on first run)
- Data: `training_data/dad_jokes_train.jsonl` / `dad_jokes_validation.jsonl`
- LR 5e-5, 100 warmup steps then cosine decay, batch 16, max 8 epochs, early
  stopping with patience 2
- Output: `dad-joke-model/finetuned_best.pt` (~312 MB, gitignored)
- Expect ~20 minutes on Apple Silicon; best val loss around 2.05

### 3. Export to ONNX

```bash
python scripts/export_finetuned_to_onnx.py
```

- Exports via **Hugging Face Optimum**. Plain `torch.onnx.export` (both the
  dynamo and legacy TorchScript exporters) fails on GPT-2 graphs with
  transformers ≥ 4.5x because the causal-mask helper isn't traceable.
- Dynamic-quantizes weights to QUInt8
- Output: `dad-joke-model/dad_joke_finetuned_quantized.onnx` (~115 MB)
- The graph takes `input_ids`, `attention_mask`, `position_ids` (all int64,
  shape `[1, seq_len]`) and returns `logits` `[1, seq_len, 50257]`.
  `index.html` feeds all three.

### 4. Sanity-check the output

Quick sample from the PyTorch checkpoint:

```bash
python - <<'EOF'
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
ckpt = torch.load("dad-joke-model/finetuned_best.pt", map_location="cpu")
m = GPT2LMHeadModel.from_pretrained(ckpt["base_model"])
m.load_state_dict(ckpt["model_state_dict"]); m.train(False)
tok = GPT2Tokenizer.from_pretrained(ckpt["base_model"])
ids = tok.encode("Q:", return_tensors="pt")
for _ in range(5):
    out = m.generate(ids, max_new_tokens=60, do_sample=True, temperature=0.8,
                     top_k=50, top_p=0.9, eos_token_id=tok.eos_token_id,
                     pad_token_id=tok.eos_token_id)
    print(tok.decode(out[0], skip_special_tokens=True))
EOF
```

Then serve the app (`python -m http.server 8765`), switch to AI mode, and
generate a few jokes in the browser.

---

## Known limitations and next steps

The current model is *coherent* but not consistently *funny*. Roughly half of
generations are well-formed but flat ("Why did the basketball player love
elevators? A: He loved elevators!"). Options to push quality further, in
rough order of cost:

1. **Bigger base model** — full GPT-2 (124M) quantizes to ~175 MB and has
   noticeably better world knowledge. GPT-2 Medium (355M) is better still but
   ~400 MB quantized is a heavy browser download.
2. **More and better data** — the 4,109 jokes were synthetically generated.
   Filtering to the strongest few thousand, or adding a few thousand
   human-written jokes, would likely help more than any hyperparameter.
3. **Rejection sampling in the browser** — generate 3–5 candidates and keep the
   one that passes a stricter validator (e.g. requires the answer to share a
   sound or root with a word in the question).
4. **Lower temperature** — the browser currently samples at 0.7–0.9. Dropping
   to ~0.6 trades variety for coherence.

---

## File map

| File | Purpose |
|---|---|
| `scripts/finetune_gpt2.py` | Fine-tune DistilGPT-2 (current approach) |
| `scripts/export_finetuned_to_onnx.py` | Optimum export + quantization (current) |
| `scripts/model.py` | Custom transformer architecture (v1, kept for reference) |
| `scripts/train_from_scratch.py` | From-scratch trainer (v1, kept for reference) |
| `scripts/export_to_onnx.py` | Export for the v1 custom model |
| `docs/plans/2026-06-09-training-improvements.md` | The v1 improvement plan that led to this analysis |
