# Custom Dad Joke Model — Design Document

**Date:** 2026-04-14
**Branch:** `custom-joke-model-v2`
**Status:** Approved

---

## Goal

Replace the generic Qwen2.5-3B model (~2GB, poor joke quality) with a custom-built small transformer trained entirely on high-quality dad jokes. The result: a ~50-80MB model that loads fast, runs in any modern browser, and generates genuinely funny dad jokes.

## Decisions

| Decision | Choice |
|----------|--------|
| Model type | Custom transformer, trained from scratch |
| Tokenizer | GPT-2 (pre-trained, 50,257 vocab) |
| Training data | 5,000-10,000 Q&A format jokes |
| Data source | Existing 971 jokes + Claude Code generates the rest |
| Data format | `Q: [setup]? A: [punchline]!` |
| Humor validation | Claude self-scoring (4+/5) + wordplay detection |
| Runtime | ONNX Runtime Web (browser, WebAssembly) |
| Training hardware | M4 Mac Mini (MPS/Metal) |
| Load time priority | Quality over speed (progress bar is fine) |

## Model Architecture

| Component | Spec |
|-----------|------|
| Type | Decoder-only transformer (GPT-2 style) |
| Tokenizer | GPT-2 (50,257 vocab, pre-trained) |
| Layers | 6 transformer blocks |
| Embedding dim | 512 |
| Attention heads | 8 |
| Context length | 128 tokens |
| Parameters | ~25M |
| Exported size | ~50-80MB (ONNX, float16) |

## Training Data Pipeline

### Generation

- Claude Code generates jokes in batches of ~200
- Each joke rated on: groan factor (1-5), pun quality (1-5), surprise (1-5)
- Only jokes scoring 4+ out of 5 included
- Mechanical wordplay detection as a safety net
- Variety across categories: science, food, animals, professions, household, sports, technology, holidays, etc.
- De-duplicated against existing jokes and across batches

### Format

```json
{"text": "Q: Why don't scientists trust atoms? A: Because they make up everything!"}
```

### Split

- 90% training (~4,500-9,000 jokes)
- 10% validation (~500-1,000 jokes)

### Validation Rules

- Must have `Q:` and `A:` format
- Must contain `?` in the setup
- 20-200 characters total
- Family-friendly only
- Must contain wordplay/pun
- No meta-commentary
- No duplicates

## Training Pipeline

1. **Generate data** — Claude Code writes 5,000-10,000 jokes to `training_data/*.jsonl`
2. **Train model** — PyTorch + MPS, ~10-20 epochs, ~1-2 hours on M4 Mac Mini
3. **Export to ONNX** — Convert PyTorch model, quantize to float16
4. **Integrate into browser** — ONNX Runtime Web replaces WebLLM + Qwen2.5-3B
5. **Host model** — Upload to Hugging Face or serve from repo, browser caches in IndexedDB

## Browser Integration

### What changes

- Replace WebLLM + Qwen2.5-3B (~2GB) with ONNX Runtime Web + custom model (~50-80MB)
- No WebGPU requirement — WebAssembly works on more browsers
- Could re-enable AI mode on mobile (currently disabled due to 2GB model)

### What stays the same

- AI mode UI (button, progress bar, loading states)
- JokeValidator (format, content, quality scoring)
- Retry logic (up to 3 attempts with increasing temperature)
- Fallback to Local Vault on failure
- Session tracking and anti-repeat

## Error Handling

- **Model fails to load** — fall back to Local Vault
- **Invalid joke generated** — retry up to 3 times
- **All retries fail** — silent fallback to Local Vault
- **No WebAssembly** — hide AI button (rare — near-universal support)

## Testing Strategy

- Generate 100 sample jokes, run through validator — target 80%+ pass rate
- Manual browser testing across AI, Local Vault, and API modes
- Measure model load time and inference time
- Regression: ensure existing modes unchanged

## Success Metrics

- **Validation pass rate:** 80%+ of generated jokes pass the validator
- **Humor quality:** Noticeably better than Qwen2.5-3B baseline
- **Model size:** <100MB (vs current 2GB)
- **Load time:** <10 seconds on typical connection (vs 30+ seconds for Qwen)
- **Inference time:** <2 seconds per joke
