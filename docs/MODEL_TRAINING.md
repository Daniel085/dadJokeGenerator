# Model Training: What Worked, What Didn't, and Why

This document records the four approaches tried for the in-browser AI joke
model, why the first three fell short, and how to reproduce the current one.

---

## TL;DR

| | v1: From scratch | v2: DistilGPT-2 | v3: Qwen2.5-0.5B LoRA | v4: Qwen2.5-1.5B LoRA (current) |
|---|---|---|---|---|
| Base | Random init, 8 layers | Pretrained DistilGPT-2 | Pretrained Qwen2.5-0.5B | Pretrained Qwen2.5-1.5B |
| Parameters | 51M | 82M | 494M | 1.54B |
| Training data | 4,109 synthetic | same | 3,676 (synthetic + curated) | 2,307 (LLM-judged synthetic + curated) |
| Best val loss | 3.02 | 2.05 | 1.84 (v2 val set) | 1.62 (v3 val set) |
| Training time (M4 Mac mini) | 38 min | 21 min | 28 min | 13 min (1 epoch) |
| Browser download | 49 MB | 115 MB | 426 MB | 1.1 GB |
| Browser latency per candidate | ~11 s | ~11 s | 0.5–1.5 s (Chrome, WebGPU) | 1–3 s (Chrome, WebGPU) |
| Jokes that land (hand-scored) | ~5% | ~5% | ~30% | **~30-35%, none copied** |
| Nonsense | ~60% | ~60% | ~25% | ~20% |

**Lessons:**

1. A model trained from scratch on a few thousand examples learns the *shape*
   of a joke but not the language it's written in.
2. A small pretrained model (82M) learns the language but still cannot do
   puns: it produces pun-*shaped* output ("za-stra performer") with no sound
   overlap.
3. A 0.5B model fine-tuned on the same data is the first one where a
   meaningful share of jokes actually work.
4. Going to 1.5B is what moves coherence again. Cleaning the data with an
   LLM judge did *not* help the 0.5B on its own (see the ablation in v4).
   The bigger model memorises training jokes at 2 epochs (5 of 30 verbatim);
   at 1 epoch it produced none in 30 with the same original-joke hit rate,
   and the browser additionally filters anything from the training set.

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

## v3: Qwen2.5-0.5B (superseded by v4, kept as the small option)

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

## v4: The current model (Qwen2.5-1.5B on LLM-judged data)

Two levers were pulled after v3, and only one of them turned out to matter.

### Lever 1: LLM-judged training data (`scripts/judge_jokes.py`, `prepare_data_v3.py`)

Every synthetic joke was scored 1-5 by Claude via the `claude` CLI in
headless mode (`claude -p`, no API key), 40 jokes per call, 8 calls in
parallel, ~20 minutes for 4,503 unique jokes:

| Score | Meaning | Count |
|---|---|---|
| 5 | real pun that clearly works | 369 |
| 4 | real pun, weak or well-worn | 2,402 |
| 3 | joke-shaped, wordplay is a stretch | 1,130 |
| 2 | grammatical, no actual joke | 567 |
| 1 | nonsense / broken | 35 |

Keeping 4 and above, plus all 766 curated human jokes, deduping and capping
the "Why did the" opener at 30% gives **2,307 train / 256 validation**.
Spot checks agree with the judge: dropped examples look like "What do you
call a happy hospital bed? A resting place of joy!"; 5s look like "What do
you call a furnace that loves music? A heavy metal fan!".

### Lever 2: Qwen2.5-1.5B

LoRA r=16 on all projections (18.5M trainable, 1.2%), batch 4, LR 2e-4.
A dry run showed 6 GB allocated / 10.6 GB driver memory on MPS, so it fits
on a 16 GB Mac. Val loss: epoch 1 **1.6238**, epoch 2 1.6216, then 1.77 and
1.98 (overfitting). One epoch is the sweet spot: identical validation loss
to epoch 2 with a much higher train loss, i.e. much less memorisation.

### Ablation: which lever mattered?

Same 30-sample hand-scoring, same sampling seed, all three from the
quantized ONNX exports:

| Model | Data | Land | Nonsense | Verbatim training copies |
|---|---|---|---|---|
| 0.5B | v2 | ~33% | ~23% | 0 |
| 0.5B | v3 (judged) | ~25% | ~35% | 1 |
| 1.5B | v3 (judged), 2 epochs | ~43% | ~17% | **5** |
| **1.5B, 1 epoch (shipped)** | v3 (judged) | ~30-35% | ~20% | **0** |

The judged data did nothing for the 0.5B (if anything it hurt, likely
because the set is 40% smaller). The 1.5B is clearly more coherent, and its
hits are better ("Why did the sailor love music? It was full of sea-sharps
and sea-flats!", "What do you call a sleeping bag at the airport? A terminal
nap sack!"). But it recited training jokes: "outstanding in his field" three
times in 30, plus "dino-snore" and "night-stand". It also emitted a Chinese
character once.

### The regurgitation filter (`scripts/build_known_jokes.py`, `known_jokes.json`)

Because the bigger model memorises, the browser now loads a 100 KB set of
FNV-1a hashes of every training joke and every training *question* and
rejects any candidate that matches either. Normalisation and hash are
implemented identically in Python and JS (parity verified). Combined with a
non-English check, AI mode can only show material that is not in the
training set. In practice the best-of-2 loop absorbs the rejections.

### Export notes specific to 1.5B

- Optimum's post-processing (tied-weight dedupe) serialises the fp32 graph
  into one protobuf and fails above 2 GB; `no_post_process=True` skips it.
  The head then arrives as its own initializer and MatMulNBits quantizes it
  directly.
- Result: 4-bit MatMuls + int8 embedding (890 MB -> 223 MB) = **1,145 MB**.
- The KV cache is 28 layers x 2 heads x head_dim 128; the page reads those
  from `config.json` rather than assuming the 0.5B's shape.

---

## Reproducing the current model

All commands run from the repo root with the project venv active. The
`claude` CLI must be installed and logged in for the judging step.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r scripts/requirements.txt

python scripts/prepare_data_v2.py                       # base pool
python scripts/judge_jokes.py --workers 8               # ~20 min, resumable
python scripts/prepare_data_v3.py --min-score 4
python scripts/finetune_qwen.py --base Qwen/Qwen2.5-1.5B --data v3 --name qwen15 --batch 4 --epochs 1   # ~15 min
python scripts/export_qwen_to_onnx.py dad-joke-model/qwen15_finetuned_hf dad-joke-model/qwen15         # ~10 min
python scripts/build_known_jokes.py                     # known_jokes.json (committed)
python -m http.server 8765                              # then open http://localhost:8765
```

`dad-joke-model/` is gitignored; the exported folder must be produced (or
copied) wherever the site is served from. Add `?debug=true` to the URL to
see per-attempt timing, validator decisions and candidates in the console,
and `?model=<folder>` to load a different export (e.g. `?model=qwen` for
the 0.5B).

---

## Known limitations and next steps

- **Download size.** 1.1 GB is the WebLLM-scale download this project once
  moved away from. Vocabulary pruning (the model only needs to *emit*
  English tokens) could cut ~200 MB; a smaller vocabulary model would cut
  more. The 0.5B export (426 MB) remains available via `?model=qwen`.
- **Memorisation.** Even at one epoch the 1.5B leans on well-known jokes.
  The hash filter blocks exact copies and reused setups, not near-copies
  ("Why did the music teacher get promoted? She was outstanding in her
  field!"). A fuzzy filter (e.g. shared 4-gram ratio against the training
  set) would catch those.
- **Judged data didn't help the small model.** Either the filtered set is
  too small, or the judge's threshold removes stylistic variety. Worth
  re-testing with score >= 3 or with more synthetic data generated to
  replace what was dropped.
- **Per-token speed.** In the in-app preview pane the 1.5B takes 1.7-5 s per
  candidate; in Chrome proper it's roughly 3x faster. An fp16 graph would
  likely help further.

---

## File map

| File | Purpose |
|---|---|
| `scripts/prepare_data_v2.py` | Build the v2 dataset (curated + synthetic, deduped, balanced) |
| `scripts/judge_jokes.py` | Score synthetic jokes 1-5 with the claude CLI (headless, resumable) |
| `scripts/prepare_data_v3.py` | v2 pipeline with judge filtering (current data) |
| `scripts/finetune_qwen.py` | LoRA fine-tune any Qwen2.5 base (`--base/--data/--name`) (current) |
| `scripts/export_qwen_to_onnx.py` | KV-cache export, untie head, 4-bit + int8, tokenizer fix (current) |
| `scripts/build_known_jokes.py` | Hash set of training jokes/questions for the browser filter |
| `scripts/finetune_gpt2.py` | v2 DistilGPT-2 fine-tune (kept for reference) |
| `scripts/export_finetuned_to_onnx.py` | v2 export (kept for reference) |
| `scripts/model.py`, `scripts/train_from_scratch.py`, `scripts/export_to_onnx.py` | v1 from-scratch model (kept for reference) |
