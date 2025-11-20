# AI Dad Joke Generator - Technical Architecture

**Document Version:** 1.0
**Last Updated:** 2025-11-20
**Implementation Branch:** `feature/ai-joke-generation`

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
│  │  - getJoke(topic)                                    │   │
│  │  - trackSeenJoke(id)                                 │   │
│  └────┬──────────────┬───────────────────┬──────────────┘   │
└───────┼──────────────┼───────────────────┼──────────────────┘
        │              │                   │
        ▼              ▼                   ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────────────────┐
│ API Source  │ │Local Source │ │   AI Source              │
│             │ │             │ │                          │
│ - fetch()   │ │ - random()  │ │  ┌────────────────────┐  │
│ - parse()   │ │ - filter()  │ │  │  DadJokeAI         │  │
└─────────────┘ └─────────────┘ │  │                    │  │
                                │  │ - initialize()     │  │
                                │  │ - generate()       │  │
                                │  │ - validate()       │  │
                                │  └──────────┬─────────┘  │
                                │             │            │
                                │             ▼            │
                                │  ┌────────────────────┐  │
                                │  │  WebLLM Engine     │  │
                                │  │                    │  │
                                │  │ - loadModel()      │  │
                                │  │ - inference()      │  │
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

## Core Classes & Responsibilities

### 1. `JokeValidator`

**Purpose:** Validate AI-generated jokes meet quality standards

**Methods:**
```javascript
class JokeValidator {
  validate(joke: string): ValidationResult
  checkFormat(joke: string): boolean
  detectWordplay(joke: string): boolean
  containsProfanity(joke: string): boolean
  hasMetaCommentary(joke: string): boolean
  calculateQualityScore(joke: string): number
}
```

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

### 2. `DadJokeAI`

**Purpose:** Manage WebLLM model and joke generation

**State:**
```javascript
{
  engine: WebLLMEngine | null,
  isInitialized: boolean,
  isGenerating: boolean,
  modelConfig: {
    name: "Qwen2.5-3B-Instruct-q4f16_1-MLC",
    size: "~2GB",
    quantization: "q4f16_1"
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

**Methods:**
```javascript
class DadJokeAI {
  async initialize(onProgress): Promise<void>
  async generateJoke(topic?: string): Promise<Joke>
  async generateWithQuality(topic?: string, maxAttempts: 3): Promise<Joke>
  buildPrompt(topic?: string, attemptNumber: number): string
  cleanJoke(rawOutput: string): string
  getStats(): Stats
  unload(): void
}
```

**Generation Flow:**
```
generateJoke(topic)
    │
    ├─► Build Few-Shot Prompt
    │   ├─ Get 3-6 random examples from static DB
    │   ├─ Build system prompt with rules
    │   └─ Add user prompt with topic
    │
    ├─► Attempt 1 (temp=0.7)
    │   ├─► Model Inference (2-5s)
    │   ├─► Clean Output
    │   ├─► Validate
    │   └─► PASS? ──► Return Joke
    │       │
    │       FAIL
    │       │
    ├─► Attempt 2 (temp=0.8)
    │   ├─► Model Inference
    │   ├─► Clean Output
    │   ├─► Validate
    │   └─► PASS? ──► Return Joke
    │       │
    │       FAIL
    │       │
    ├─► Attempt 3 (temp=0.9)
    │   ├─► Model Inference
    │   ├─► Clean Output
    │   ├─► Validate
    │   └─► PASS? ──► Return Joke
    │       │
    │       FAIL
    │       │
    └─► Fallback to Static Database
```

---

### 3. `PromptBuilder`

**Purpose:** Construct optimal prompts for joke generation

**System Prompt Template:**
```
You are a professional dad joke writer. Generate ONE dad joke following this exact format:

Q: [Setup with question]
A: [Punchline with wordplay]

Requirements:
- Must use wordplay, puns, or homophones
- Must be family-friendly (G-rated)
- Must be groan-worthy but clever
- Maximum 2 sentences total
- No explanations or meta-commentary
```

**Few-Shot Example Selection:**
```javascript
function getRandomExamples(count, previousExamples = []) {
  // Select N examples from static database
  // Ensure diversity (no repeated examples)
  // Prioritize high-quality examples
  // Return in consistent format
}
```

**User Prompt Template:**
```
Generate a dad joke like these examples:

1. Q: Why don't scientists trust atoms?
   A: Because they make up everything!

2. Q: What do you call a factory that makes okay products?
   A: A satisfactory!

3. Q: Why did the scarecrow win an award?
   A: He was outstanding in his field!

Now create a NEW original dad joke about: {topic}
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
                              │               Download model (2GB)
                              │                      │
                              │                      ▼
                              │               Initialize engine
                              │                      │
                              │                      ▼
                              │                  Continue...
                              │
                              └─► INITIALIZED ──► Generate joke
                                                      │
                                                      ├─► Attempt 1
                                                      │      │
                                                      │      ├─► VALID ──► Display
                                                      │      │
                                                      │      └─► INVALID
                                                      │             │
                                                      ├─► Attempt 2
                                                      │      │
                                                      │      ├─► VALID ──► Display
                                                      │      │
                                                      │      └─► INVALID
                                                      │             │
                                                      ├─► Attempt 3
                                                      │      │
                                                      │      ├─► VALID ──► Display
                                                      │      │
                                                      │      └─► INVALID
                                                      │             │
                                                      └─► Fallback to Local DB
```

---

## WebLLM Integration Details

### Model Loading Sequence

```javascript
1. Import WebLLM
   import * as webllm from "https://esm.run/@mlc-ai/web-llm"

2. Check WebGPU Support
   if (!navigator.gpu) throw new Error("WebGPU not supported")

3. Create Engine with Progress Callback
   const engine = await webllm.CreateMLCEngine(
     "Qwen2.5-3B-Instruct-q4f16_1-MLC",
     {
       initProgressCallback: (progress) => {
         // Update UI with download/init progress
         updateProgress(progress.text, progress.progress);
       }
     }
   )

4. Store Engine Reference
   this.engine = engine
   this.isInitialized = true
```

### Progress Events

```javascript
Progress Events:
1. "Fetching param cache[0/1]"
2. "Loading model from cache[0/291]"
3. "Loading model from cache[100/291]"
4. "Loading model from cache[200/291]"
5. "Loading model from cache[291/291]"
6. "Initializing VM"
7. "Initializing WebGPU"
8. "Ready"
```

### Inference Call

```javascript
const response = await engine.chat.completions.create({
  messages: [
    {
      role: "system",
      content: systemPrompt
    },
    {
      role: "user",
      content: userPrompt
    }
  ],
  temperature: 0.7,        // Creativity vs consistency
  max_tokens: 100,         // Limit response length
  top_p: 0.9,             // Nucleus sampling
  frequency_penalty: 0.3   // Reduce repetition
});

const jokeText = response.choices[0].message.content;
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
```javascript
function calculateQualityScore(joke) {
  let score = 0;

  // Length sweet spot (50-150 chars)
  if (joke.length >= 50 && joke.length <= 150) score += 30;

  // Has question mark
  if (/\?/.test(joke)) score += 20;

  // Has exclamation (enthusiasm)
  if (/!/.test(joke)) score += 10;

  // Contains common dad joke patterns
  if (/\b(why|what|how)\b/i.test(joke)) score += 15;
  if (/\b(because|so|they|it)\b/i.test(joke)) score += 10;

  // Wordplay indicators
  if (detectWordplay(joke)) score += 25;

  // Penalties
  if (joke.length > 200) score -= 20;
  if (/(here's|I made)/i.test(joke)) score -= 50;

  return score;
}
```

**Acceptance Threshold:** Score >= 60

---

## Performance Optimization

### Caching Strategy

```javascript
// Model caching (automatic via WebLLM)
IndexedDB: "mlc-chat-config"
  ├─ model weights (2GB)
  ├─ tokenizer config
  └─ runtime config

// Example caching (manual)
sessionStorage: "fewShotExamples"
  └─ Pre-selected high-quality examples

// Seen jokes tracking (existing)
sessionStorage: "seenJokes"
  └─ Set of seen joke IDs
```

### Memory Management

```javascript
// Unload model when switching sources
function switchSource(newSource) {
  if (currentSource === 'ai' && newSource !== 'ai') {
    await dadJokeAI.unload();  // Free VRAM
  }

  if (newSource === 'ai' && !dadJokeAI.isInitialized) {
    await dadJokeAI.initialize();  // Reload if needed
  }

  currentSource = newSource;
}
```

### Lazy Loading

```javascript
// Only load WebLLM when AI mode selected
let webllmModule = null;

async function initializeAIMode() {
  if (!webllmModule) {
    webllmModule = await import("https://esm.run/@mlc-ai/web-llm");
  }

  // ... initialize engine
}
```

---

## Error Handling

### Error Categories & Recovery

| Error Type | Detection | Recovery Strategy |
|------------|-----------|-------------------|
| **WebGPU Not Supported** | `!navigator.gpu` | Hide AI button, show browser requirements |
| **Model Download Failed** | Network error during fetch | Retry with exponential backoff (3 attempts) |
| **Out of Memory** | WebGPU allocation error | Unload model, suggest closing tabs, fallback to Local |
| **Generation Timeout** | Inference >30s | Cancel generation, fallback to Local |
| **Validation Failed** | All 3 attempts invalid | Silent fallback to static DB |
| **Model Corrupted** | Initialization error | Clear IndexedDB cache, re-download |

### Error Logging

```javascript
class ErrorLogger {
  static log(error, context) {
    console.error(`[DadJokeAI] ${context}:`, error);

    // Track for debugging
    if (window.aiErrors) {
      window.aiErrors.push({
        timestamp: Date.now(),
        context,
        error: error.message,
        stack: error.stack
      });
    }
  }

  static getReport() {
    return window.aiErrors || [];
  }
}
```

---

## Testing Strategy

### Unit Tests

```javascript
// JokeValidator tests
describe('JokeValidator', () => {
  test('accepts valid dad joke', () => {
    const joke = "Q: Why don't scientists trust atoms?\nA: Because they make up everything!";
    expect(validator.validate(joke)).toBe(true);
  });

  test('rejects joke without question mark', () => {
    const joke = "Q: Why don't scientists trust atoms\nA: Because they make up everything!";
    expect(validator.validate(joke)).toBe(false);
  });

  test('detects wordplay in "make up"', () => {
    const joke = "They make up everything!";
    expect(validator.detectWordplay(joke)).toBe(true);
  });
});
```

### Integration Tests

```javascript
// AI Generation tests
describe('DadJokeAI', () => {
  test('generates valid joke within 10 seconds', async () => {
    const start = Date.now();
    const joke = await dadJokeAI.generateJoke();
    const duration = Date.now() - start;

    expect(joke).toBeTruthy();
    expect(validator.validate(joke)).toBe(true);
    expect(duration).toBeLessThan(10000);
  });

  test('falls back after 3 failed attempts', async () => {
    // Mock validator to always fail
    validator.validate = jest.fn(() => false);

    const joke = await dadJokeAI.generateJoke();

    // Should have fallen back to static DB
    expect(joke.source).toBe('local');
  });
});
```

### Load Tests

```javascript
// Stress test
test('handles 100 rapid generations without crash', async () => {
  const promises = [];

  for (let i = 0; i < 100; i++) {
    promises.push(dadJokeAI.generateJoke());
  }

  const results = await Promise.all(promises);

  expect(results.length).toBe(100);
  expect(results.every(r => r.joke)).toBe(true);
});
```

---

## Security Considerations

### Privacy

✅ **All processing happens client-side**
- No data sent to external servers (except model download)
- No user tracking
- No analytics

✅ **Model weights are public**
- Qwen2.5 is open-source (Apache 2.0)
- No proprietary data

### Content Safety

✅ **Profanity filter** prevents inappropriate jokes
✅ **Validation layer** ensures family-friendly content
✅ **Fallback system** always available

---

## Deployment Checklist

### Pre-Deployment

- [ ] Test on Chrome 113+
- [ ] Test on Edge 113+
- [ ] Test on Safari 18+ (if available)
- [ ] Verify model loads successfully
- [ ] Verify 20+ generated jokes are quality
- [ ] Test error handling (network offline, etc.)
- [ ] Verify graceful degradation
- [ ] Check memory usage (<3GB total)
- [ ] Test mobile detection/disable

### Post-Deployment

- [ ] Monitor error rates
- [ ] Track validation pass rates
- [ ] Collect user feedback
- [ ] A/B test model selection
- [ ] Optimize prompt engineering

---

## Monitoring & Analytics

### Key Metrics to Track

```javascript
window.aiMetrics = {
  modelLoadTime: number,           // Time to load model
  avgGenerationTime: number,       // Avg time per joke
  validationPassRate: number,      // % passed validation
  fallbackRate: number,            // % fell back to static
  errorRate: number,               // % of errors
  userSatisfaction: number         // Future: thumbs up/down
};
```

### Debug Mode

```javascript
// Enable with ?debug=true
if (new URLSearchParams(location.search).get('debug')) {
  showAIDebugPanel();
}

function showAIDebugPanel() {
  // Display live metrics
  // Show validation details
  // Log prompts and responses
  // Display quality scores
}
```

---

## Future Optimizations

### Model Optimization
- [ ] Fine-tune Qwen2.5 on dad jokes dataset
- [ ] Quantize to 3-bit for smaller size
- [ ] Explore distillation to 1B model

### Prompt Optimization
- [ ] A/B test system prompts
- [ ] Optimize few-shot example selection
- [ ] Dynamic temperature based on topic

### Performance
- [ ] Implement streaming inference (show joke as generated)
- [ ] Batch generation (generate 5, cache 4)
- [ ] Pre-warm model on page load

---

## Conclusion

This architecture balances:
- **Quality:** Multi-layer validation ensures high standards
- **Performance:** Efficient model, caching, lazy loading
- **UX:** Progressive enhancement, clear feedback
- **Robustness:** Fallbacks at every layer
- **Maintainability:** Clear separation of concerns

The result is a production-ready AI feature that showcases modern web ML while maintaining the simplicity and reliability of the core application.

---

**Document Status:** ✅ Complete
**Implementation Status:** 🚧 In Progress
**Next Steps:** Begin implementation of JokeValidator class
