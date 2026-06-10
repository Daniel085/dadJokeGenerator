# AI Dad Joke Generator - Technical Architecture

**Document Version:** 2.0
**Last Updated:** 2026-04-15
**Implementation Branch:** `custom-joke-model-v2`

---

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │  API Mode   │  │ Local Mode  │  │   AI Mode        │    │
│  │  Button     │  │  Button     │  │   Button         │    │
│  └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘    │
└─────────┼─────────────────┼──────────────────┼──────────────┘
          │                 │                  │
          ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  Joke Management Layer                       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           JokeSourceManager                          │   │
│  │  - selectSource(source)                              │   │
│  │  - getJoke()                                         │   │
│  │  - trackSeenJoke(id)                                 │   │
│  └────┬──────────────┬───────────────────┬──────────────┘   │
└───────┼──────────────┼───────────────────┼──────────────────┘
        │              │                   │
        ▼              ▼                   ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────────────────┐
│ API Source  │ │Local Source │ │   AI Source              │
│             │ │             │ │                          │
│ - fetch()   │ │ - random()  │ │  ┌────────────────────┐  │
│ - parse()   │ │ - filter()  │ │  │  ONNX Runtime Web  │  │
└─────────────┘ └─────────────┘ │  │                    │  │
                                │  │ - loadModel()      │  │
                                │  │ - inference()      │  │
                                │  └──────────┬─────────┘  │
                                │             │            │
                                │             ▼            │
                                │  ┌────────────────────┐  │
                                │  │  GPT-2 Tokenizer   │  │
                                │  │  (Transformers.js) │  │
                                │  └──────────┬─────────┘  │
                                │             │            │
                                │             ▼            │
                                │  ┌────────────────────┐  │
                                │  │ Quality Layer      │  │
                                │  │                    │  │
                                │  │ - JokeValidator    │  │
                                │  │ - RetryLogic       │  │
                                │  │ - Fallback         │  │
                                │  └────────────────────┘  │
                                └──────────────────────────┘
```

---

## Model Architecture

### DadJokeTransformer (~25M Parameters)

A custom decoder-only transformer trained from scratch on 4,500+ curated dad jokes.

**Configuration:**
```python
class DadJokeConfig:
    vocab_size = 50257      # GPT-2 tokenizer vocabulary
    n_layers = 6            # Transformer blocks
    n_heads = 8             # Attention heads
    d_model = 512           # Hidden dimension
    d_ff = 2048             # Feed-forward dimension
    max_seq_len = 128       # Maximum sequence length
    dropout = 0.1           # Dropout rate
```

**Architecture Details:**
- Pre-norm (LayerNorm before attention/FFN)
- Weight tying between token embedding and output head
- Multi-head self-attention with causal masking
- GELU activation in feed-forward layers
- Learned positional embeddings

**Training:**
- Optimizer: AdamW (lr=3e-4, weight_decay=0.01)
- Scheduler: Cosine annealing
- Early stopping: patience=5 epochs
- Hardware: M4 Mac Mini (MPS backend)
- Data: 4,109 train / 457 validation jokes

---

## Core Classes & Responsibilities

### 1. `JokeValidator`

**Purpose:** Validate AI-generated jokes meet quality standards

**Validation Rules:**
1. **Format Check:** Must match `Q: ... ? A: ...` pattern
2. **Length Check:** 20-200 characters total
3. **Profanity Check:** No inappropriate language
4. **Wordplay Detection:** Must contain puns/homophones
5. **Meta Check:** No "Here's a joke..." or explanations
6. **Question Check:** Must contain "?"

**Validation Flow:**
```
Input Joke
    │
    ├─► Format Check ────► FAIL ────► Reject
    │                  └─► PASS
    ├─► Length Check ────► FAIL ────► Reject
    │                  └─► PASS
    ├─► Profanity Check ─► FAIL ────► Reject
    │                  └─► PASS
    ├─► Wordplay Check ──► FAIL ────► Reject
    │                  └─► PASS
    └─► Meta Check ──────► FAIL ────► Reject
                      └─► PASS ────► Accept
```

---

### 2. Browser-Side AI Generation

**State:**
```javascript
{
  aiSession: ort.InferenceSession | null,  // ONNX Runtime session
  aiTokenizer: AutoTokenizer | null,        // GPT-2 tokenizer
  isInitialized: boolean,
  isGenerating: boolean,
  modelConfig: {
    name: "DadJokeTransformer",
    size: "~50-80MB",
    format: "ONNX (quantized uint8)"
  },
  stats: {
    generated: number,
    accepted: number,
    rejected: number,
    avgGenerationTime: number,
    fallbackCount: number
  }
}
```

**Generation Flow:**
```
generateWithONNX(temperature)
    │
    ├─► Tokenize "Q:" prompt with GPT-2 tokenizer
    │
    ├─► Autoregressive generation loop:
    │   ├─ Run ONNX inference (input_ids → logits)
    │   ├─ Apply temperature scaling
    │   ├─ Top-k filtering (k=50)
    │   ├─ Softmax → sample next token
    │   ├─ Append to sequence
    │   └─ Stop at EOS or max_length
    │
    ├─► Decode token IDs back to text
    │
    ├─► Validate with JokeValidator
    │   ├─► PASS → Return joke
    │   └─► FAIL → Retry (up to 3x, increasing temperature)
    │
    └─► All failed → Fallback to Local DB
```

**Key Function:**
```javascript
async function generateWithONNX(temperature = 0.8) {
    const inputText = "Q:";
    const inputIds = aiTokenizer.encode(inputText);
    let generatedIds = [...inputIds];

    for (let i = 0; i < 80; i++) {
        const tensor = new ort.Tensor('int64',
            BigInt64Array.from(generatedIds.map(id => BigInt(id))),
            [1, generatedIds.length]);

        const output = await aiSession.run({ input_ids: tensor });
        const logits = output.logits.data;
        const seqLen = generatedIds.length;
        const vocabSize = 50257;
        const lastLogits = logits.slice((seqLen - 1) * vocabSize, seqLen * vocabSize);

        const nextToken = sampleFromLogits(lastLogits, temperature, 50);

        if (nextToken === eosTokenId) break;
        generatedIds.push(nextToken);
    }

    return aiTokenizer.decode(generatedIds);
}
```

---

## Data Flow

### Complete User Journey

```
User clicks "BUILD A JOKE"
         │
         ▼
Check selected source (API / Local / AI)
         │
         ├─── API ─────► Fetch from icanhazdadjoke.com
         │                    │
         │                    ├─► SUCCESS ──► Display joke
         │                    │
         │                    └─► FAIL ────► Fallback to Local
         │
         ├─── Local ───► Get from jokes.json
         │                    │
         │                    ├─► Filter out seen jokes
         │                    │
         │                    ├─► Random selection
         │                    │
         │                    └─► Display joke
         │
         └─── AI ──────► Check if initialized
                              │
                              ├─► NOT INIT ──► Show init UI
                              │                      │
                              │                      ▼
                              │               Load GPT-2 tokenizer
                              │                      │
                              │                      ▼
                              │               Load ONNX model (~50-80MB)
                              │                      │
                              │                      ▼
                              │               Create InferenceSession
                              │                      │
                              │                      ▼
                              │                  Continue...
                              │
                              └─► INITIALIZED ──► Generate joke
                                                      │
                                                      ├─► Attempt 1 (temp=0.7)
                                                      │      │
                                                      │      ├─► VALID ──► Display
                                                      │      │
                                                      │      └─► INVALID
                                                      │             │
                                                      ├─► Attempt 2 (temp=0.8)
                                                      │      │
                                                      │      ├─► VALID ──► Display
                                                      │      │
                                                      │      └─► INVALID
                                                      │             │
                                                      ├─► Attempt 3 (temp=0.9)
                                                      │      │
                                                      │      ├─► VALID ──► Display
                                                      │      │
                                                      │      └─► INVALID
                                                      │             │
                                                      └─► Fallback to Local DB
```

---

## ONNX Runtime Web Integration

### Model Loading Sequence

```javascript
1. Import ONNX Runtime Web + Transformers.js
   import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"
   import { AutoTokenizer } from "https://cdn.jsdelivr.net/npm/@xenova/transformers"

2. Load GPT-2 Tokenizer
   aiTokenizer = await AutoTokenizer.from_pretrained('gpt2')

3. Create ONNX Inference Session
   aiSession = await ort.InferenceSession.create(
     './model/dad_joke_model.onnx',
     { executionProviders: ['wasm'] }
   )

4. Ready for generation
```

### Inference Loop

```javascript
// Autoregressive token-by-token generation
for (let step = 0; step < maxNewTokens; step++) {
    // Create input tensor
    const tensor = new ort.Tensor('int64',
        BigInt64Array.from(ids.map(id => BigInt(id))),
        [1, ids.length]);

    // Run model
    const output = await aiSession.run({ input_ids: tensor });

    // Get logits for last position
    const lastLogits = extractLastPosition(output.logits);

    // Sample with temperature and top-k
    const nextToken = sampleFromLogits(lastLogits, temperature, topK);

    // Check for end of sequence
    if (nextToken === eosTokenId) break;

    ids.push(nextToken);
}
```

---

## Quality Control System

### Multi-Stage Validation

**Stage 1: Format Validation**
```javascript
function validateFormat(joke) {
  const hasQ = /Q:|Question:/i.test(joke);
  const hasA = /A:|Answer:/i.test(joke);
  const hasQuestion = /\?/.test(joke);
  return hasQ && hasA && hasQuestion;
}
```

**Stage 2: Content Validation**
```javascript
function validateContent(joke) {
  const isShort = joke.length >= 20 && joke.length <= 200;
  const isClean = !containsProfanity(joke);
  const noMeta = !/(here's|I made|joke about|example)/i.test(joke);
  return isShort && isClean && noMeta;
}
```

**Stage 3: Quality Scoring**
- Length sweet spot (50-150 chars): +30
- Has question mark: +20
- Has exclamation: +10
- Common dad joke patterns: +15
- Wordplay detected: +25
- Penalties for excessive length or meta-commentary

**Acceptance Threshold:** Score >= 60

---

## Performance

### Caching Strategy

```javascript
// Model caching (IndexedDB, automatic via browser)
IndexedDB:
  ├─ ONNX model weights (~50-80MB, cached after first download)
  └─ GPT-2 tokenizer files (cached by Transformers.js)

// Seen jokes tracking
sessionStorage: "seenJokes"
  └─ Set of seen joke IDs
```

### Key Metrics

| Metric | Target |
|--------|--------|
| Model download | ~50-80MB (one-time) |
| Model load time | 2-5 seconds |
| Generation time | 2-5 seconds per joke |
| Validation pass rate | 85-95% |
| Memory usage | < 200MB |

---

## Training Pipeline

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/model.py` | DadJokeTransformer architecture definition |
| `scripts/train_from_scratch.py` | Training loop with early stopping |
| `scripts/evaluate_custom_model.py` | Generate N jokes and measure quality |
| `scripts/export_to_onnx.py` | Export to ONNX + uint8 quantization |
| `scripts/joke_validator.py` | Python-side joke validation |

### Training Data

- **Source:** 4,566 unique dad jokes generated by Claude
- **Format:** JSONL with `{"text": "Q: ... A: ..."}` entries
- **Split:** 90% train (4,109) / 10% validation (457)
- **Categories:** 20+ (science, food, animals, tech, music, etc.)

### Training Process

```
training_data/dad_jokes_train.jsonl
         │
         ▼
scripts/train_from_scratch.py
  ├─ GPT-2 tokenizer (pre-trained, frozen)
  ├─ DadJokeTransformer (~25M params, random init)
  ├─ AdamW optimizer + cosine annealing
  ├─ Early stopping (patience=5)
  └─ Saves best model by validation loss
         │
         ▼
dad-joke-model/best_model.pt
         │
         ▼
scripts/export_to_onnx.py
  ├─ ONNX export (opset 17, dynamic axes)
  └─ uint8 quantization
         │
         ▼
model/dad_joke_model.onnx  →  Deployed in browser
```

---

## Error Handling

| Error Type | Detection | Recovery Strategy |
|------------|-----------|-------------------|
| **Model Download Failed** | Network error | Retry, then fallback to Local mode |
| **ONNX Session Error** | Initialization error | Show error, fallback to Local mode |
| **Generation Timeout** | Inference >30s | Cancel, fallback to Local |
| **Validation Failed** | All 3 attempts invalid | Silent fallback to Local DB |
| **Tokenizer Error** | Load failure | Fallback to Local mode |

---

## Security & Privacy

- All processing happens client-side (no server calls except initial model download)
- No user data collected or transmitted
- Custom model is open-source and family-friendly
- Profanity filter prevents inappropriate outputs

---

## Browser Compatibility

| Browser | Support |
|---------|---------|
| Chrome 113+ | Full support (WebAssembly) |
| Edge 113+ | Full support (WebAssembly) |
| Safari 16+ | Full support (WebAssembly) |
| Firefox 100+ | Full support (WebAssembly) |
| Mobile browsers | Supported (~50-80MB download) |

---

## Future Optimizations

- [ ] Fine-tune with more training data (10,000+ jokes)
- [ ] Experiment with model size (larger for quality, smaller for speed)
- [ ] Implement streaming generation (show joke as it's generated)
- [ ] Batch generation (generate 5, cache 4)
- [ ] WebGPU execution provider for faster inference
- [ ] Topic-conditioned generation

---

**Document Status:** Updated for custom DadJokeTransformer
**Implementation Status:** Model architecture + browser integration complete; training pending
