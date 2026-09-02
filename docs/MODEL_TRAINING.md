# Model Training: What Worked, What Didn't, and Why

This document records the three approaches tried for the in-browser AI joke
model, why the first two fell short, and how to reproduce the current one.

---

## TL;DR

| | v1: From scratch | v2: DistilGPT-2 fine-tune | v3: Qwen2.5-0.5B LoRA (current) |
|---|---|---|---|
| Base | Random init, 8-layer transformer | Pretrained DistilGPT-2 | Pretrained Qwen2.5-0.5B |
| Parameters | 51M | 82M | 494M |
| Training data | 4,109 synthetic jokes | same | 3,676 (synthetic + 766 curated, deduped) |
| Best val loss | 3.02 | 2.05 | 1.84 |
| Training time (M4 Mac mini) | 38 min | 21 min | 28 min |
| Browser download | 49 MB | 115 MB | 426 MB |
| Browser latency per candidate | ~11 s | ~11 s | 1.5–3 s (WebGPU) |
| Jokes that land (hand-scored) | ~5% | ~5% | **~30%** |
| Nonsense | ~60% | ~60% | ~27% |

**Lessons:**

1. A model trained from scratch on a few thousand examples learns the *shape*
   of a joke but not the language it's written in.
2. A small pretrained model (82M) learns the language but still cannot do
   puns: it produces pun-*shaped* output ("za-stra performer") with no sound
   overlap.
3. A 0.5B model fine-tuned on the same data is the first one where a
   meaningful share of jokes actually work. The cost is a 4x larger download.

---

## v1: Why the from-scratch model failed

### The data was ~1000x too small for the job

A randomly-initialised language model knows nothing: not what "beach" means,
not that questions get answers, not which words sound alike. All of that has
to come from the training text. GPT-2 learned English from roughly 40 GB of
text (~10 billion tokens). Our training set is ~150,000 tokens. That's enough
to memorise a few thousand examples, nowhere near enough to learn what words
mean.

### What it actually learned

1. **The format**: learned perfectly. Every output was `Q: ... ? A: ... !`.
2. **Local grammar**: learned partially, by filling slots in templates seen
   dozens of times.
3. **Meaning**: never. "To reach the daily answers!" is assembled from tokens
   that commonly follow `A: To` in the training set.

### The loss curves prove it

```
Epoch  8   train 2.03   val 3.02   <- best
Epoch 18   train 0.97   val 3.34   <- early stop
```

Train loss kept falling while validation loss rose: memorisation. Every
"improvement" tried (deeper model, more dropout, longer training, LR warmup)
pushed on the wrong lever.

---

## v2: Why DistilGPT-2 was better but still not good

Fine-tuning DistilGPT-2 fixed coherence immediately (val loss 3.02 -> 2.05,
first log line already better than v1's best). Every output was a grammatical
Q/A pair. But hand-scoring 75 samples gave ~1 in 20 that worked, and a live
test in Chrome gave 0 of 8.

The failure mode changed from "word salad" to two new ones:

- **Tautology**: "What do you call a lazy duck? A: A lazy duck!"
- **Pun-shaped nonsense**: hyphenated coinages with no phonetic basis,
  "bee-glades bee", "bison-ucated".

A pun needs to know that "lei" sounds like "lazy". An 82M model has no such
knowledge. Lower temperature made it worse (more tautologies). Retraining
reproduced the result exactly (val 2.0448 vs 2.0455), so it was the ceiling of
the approach, not a bad seed.

Two experiments ruled out the obvious alternatives:

- **Few-shot prompting a bigger model without fine-tuning** (Qwen2.5-0.5B and
  1.5B, five curated examples in the prompt): more grammatical, but a large
  share of output was famous jokes recited from memory ("atoms make up
  everything", three times in ten). Sub-2B models recall jokes; they don't
  invent them unless taught the style.
- **A surface-level wordplay validator as a hard gate**: rejected 17 of 24
  human-written jokes. Most real puns play on a word *associated* with the
  setup (scarecrow -> field, orange -> juice), which no string heuristic can
  see. Wordplay is now used only to rank candidates, never to reject.

---

## v3: The current model

### Data (`scripts/prepare_data_v2.py`)

- 4,566 synthetic jokes + 766 curated jokes from `jokes.json` (converted to
  Q/A form; 7 one-liners without a question skipped)
- exact and question-level dedupe: 653 removed
- the dominant opener ("Why did the...", 43% of the set) capped at 30%
- 90/10 split: `training_data/v2_train.jsonl` (3,676) and
  `v2_validation.jsonl` (408)

### Training (`scripts/finetune_qwen.py`)

LoRA rather than a full fine-tune: a full fine-tune of a 0.5B model needs
~8 GB for weights, gradients and Adam state before activations, too tight on
a 16 GB Mac. LoRA (r=16, alpha=32, all attention and MLP projections) trains
8.8M parameters (1.75%) and fits comfortably.

- LR 2e-4 with 50 warm-up steps and cosine decay, batch 8, max length 80
- early stopping with patience 2: best epoch 2 (val 1.8422); epochs 3-4
  overfit (val 1.96, 2.14)
- 27.6 minutes on MPS; adapter merged into the base weights for export

### Export (`scripts/export_qwen_to_onnx.py`)

- Optimum `text-generation-with-past`: the graph takes `input_ids`,
  `attention_mask`, `position_ids` and 24 pairs of `past_key_values.N.key/
  value`; returns `logits` and `present.N.*`. Each generated token is one
  small forward pass instead of re-processing the whole sequence.
- **Untie the output head**: Optimum emits it as
  `Transpose(embed_tokens) -> MatMul`, which the MatMulNBits quantizer skips.
  The transposed weight is materialised as its own initializer first.
- **MatMulNBits** 4-bit, block size 32, for every MatMul (168 nodes).
- **Int8 embedding table**: MatMulNBits leaves the 152k x 896 `Gather` as
  fp32 (519 MB on its own). It is replaced by a per-row symmetric int8 table
  with `Gather -> Cast -> Mul(row scale)`: 130 MB.
- **Tokenizer fix**: `tokenizers >= 0.20` writes BPE merges as pair arrays;
  Transformers.js 2.x expects `"a b"` strings and throws
  `e.split is not a function` otherwise. Rewritten in place.
- Result: 426 MB, self-contained, plus the tokenizer files in the same folder.

### Browser (`index.html`)

- `AutoTokenizer.from_pretrained('dad-joke-model/qwen')` with
  `env.allowRemoteModels = false`, so the tokenizer is served from the same
  folder as the model. The tokenizer object doesn't expose `eos_token_id`
  for Qwen2; it is read from `config.json` (151643).
- ONNX Runtime Web 1.20 `ort.webgpu.min.mjs` (includes the wasm backend).
  WebGPU is tried first with `preferredOutputLocation` set to `gpu-buffer`
  for every `present.*` output, so the KV cache never round-trips through
  the CPU between steps (this halved per-token time: ~270 ms -> ~120 ms).
  Stale GPU buffers are disposed each step.
- wasm fallback uses `ort.env.wasm.proxy = true` so inference runs in a
  worker and the page doesn't freeze. (Proxy mode must be off for WebGPU;
  it forbids GPU-resident outputs.)
- Top-k sampling over the 152k vocab uses a single linear pass keeping the
  k best, instead of sorting the whole array per token.
- A short warm-up generation at init pays the WebGPU shader-compilation
  cost (~1.5-5 s) behind the progress bar rather than on the first joke.
- Validator: hard gates only (format, length, profanity, degenerate
  repetition, tautology). Wordplay evidence ranks candidates in a best-of-2
  loop; the best valid candidate is shown, and the local vault is used only
  if every candidate is broken.

### Quality (hand-scored, 30 samples from the quantized model)

- ~30% land: "Why did the car salesman win an award? A: He had an
  outstanding drive!", "What do you call a computer that sneezes? A: A
  virus!", "What did the scallop say to the shellfish? A: You're a little
  shell-fish about me!"
- ~40% coherent but flat: "Why did the computer go to jail? A: It stole the
  data!"
- ~27% nonsense: "What do you call a car that does yoga? A: A Honda-Mi Yoga!"
- 0 of 30 were verbatim training jokes; 3 reused a training question with a
  new answer.

---

## Reproducing the current model

All commands run from the repo root with the project venv active.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r scripts/requirements.txt

python scripts/prepare_data_v2.py        # ~1 s
python scripts/finetune_qwen.py          # ~30 min on Apple Silicon
python scripts/export_qwen_to_onnx.py    # ~5 min; writes dad-joke-model/qwen/
python -m http.server 8765               # then open http://localhost:8765
```

`dad-joke-model/` is gitignored; the exported folder must be produced (or
copied) wherever the site is served from. Add `?debug=true` to the URL to
see per-attempt timing, validator decisions and candidates in the console.

---

## Known limitations and next steps

- **Download size.** 426 MB is a lot for a joke site. The int8 embedding and
  the 4-bit head are already in; the remaining lever is vocabulary pruning
  (the model only ever needs to *emit* English tokens, so the 152k table
  could be cut to ~20-30k rows, saving ~150 MB) or a smaller base such as
  SmolLM2-360M.
- **Per-token speed.** ~120 ms/token on WebGPU is slower than the hardware
  should allow; an fp16 graph or Transformers.js v3's generation pipeline
  would likely help.
- **Quality ceiling.** A third landing is a good outcome for 0.5B. The next
  real step up is a 1.5B model, which is ~1 GB in the browser, i.e. the
  WebLLM-scale download this project moved away from.
- **Data.** The synthetic set still contains weak puns. An LLM-judged filter
  keeping only jokes whose pun is phonetically real would likely help more
  than any hyperparameter.

---

## File map

| File | Purpose |
|---|---|
| `scripts/prepare_data_v2.py` | Build the v2 dataset (curated + synthetic, deduped, balanced) |
| `scripts/finetune_qwen.py` | LoRA fine-tune Qwen2.5-0.5B (current) |
| `scripts/export_qwen_to_onnx.py` | KV-cache export, untie head, 4-bit + int8, tokenizer fix (current) |
| `scripts/finetune_gpt2.py` | v2 DistilGPT-2 fine-tune (kept for reference) |
| `scripts/export_finetuned_to_onnx.py` | v2 export (kept for reference) |
| `scripts/model.py`, `scripts/train_from_scratch.py`, `scripts/export_to_onnx.py` | v1 from-scratch model (kept for reference) |
