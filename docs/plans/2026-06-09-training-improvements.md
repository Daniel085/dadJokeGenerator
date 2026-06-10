# Training Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Commit in-progress training improvements (tokenization fix, deeper model, longer training, warmup LR schedule, hallucination filtering), re-run training, re-export to ONNX, and merge to main via PR.

**Architecture:** All changes are already written in the working tree as uncommitted modifications. The plan commits them, runs training (spawns a background process — can take 30–90 min on Apple Silicon), exports the new model to ONNX, verifies the browser integration still works, then creates and merges a PR.

**Tech Stack:** Python 3, PyTorch, ONNX Runtime, GPT-2 tokenizer (HuggingFace), GitHub CLI (`gh`)

---

## Task 1: Commit the Training Improvements

**Files:**
- Modify (already done): `scripts/model.py`
- Modify (already done): `scripts/train_from_scratch.py`
- Modify (already done): `index.html`

**Step 1: Verify the diff looks right**

Run from worktree root:
```bash
git diff --stat
```
Expected output: 3 files changed — `index.html`, `scripts/model.py`, `scripts/train_from_scratch.py`

**Step 2: Stage the three files**

```bash
git add scripts/model.py scripts/train_from_scratch.py index.html
```

**Step 3: Verify staging**

```bash
git status
```
Expected: all three files under "Changes to be committed"

**Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat: improve model training — deeper architecture, fixed tokenization, warmup LR

- model.py: increase layers 6→8 and dropout 0.1→0.15 for a deeper,
  slightly more regularized model (~30M params vs ~25M)
- train_from_scratch.py: fix EOS tokenization bug (tokenizer.encode(text +
  eos_token) silently appended the string "<|endoftext|>" rather than the
  integer token id); add linear warmup + cosine decay LR schedule (200 warm-
  up steps); extend training 20→40 epochs with patience 5→10; add flush=True
  to all print() calls so logs appear in real time
- index.html: post-process generated jokes — truncate at first sentence-end
  punctuation to cut hallucinated filler; collapse repeated punctuation

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

Expected: commit created on `custom-joke-model-v2`

---

## Task 2: Run Training

**Step 1: Activate the virtual environment**

```bash
source venv/bin/activate
```

**Step 2: Verify Python dependencies**

```bash
python -c "import torch, transformers, onnx; print('deps OK')"
```
Expected: `deps OK`

**Step 3: Start training (background, log to file)**

Training takes 30–90 minutes on Apple Silicon MPS. Run it in the background so you can continue with other prep:

```bash
cd /Users/daniel/GitHub/dadJokeGenerator/.claude/worktrees/compassionate-banzai && source venv/bin/activate && python scripts/train_from_scratch.py 2>&1 | tee training_run.log &
echo "Training PID: $!"
```

**Step 4: Tail the log to confirm training started correctly**

```bash
sleep 5 && tail -30 training_run.log
```

Expected: you should see lines like:
```
Using MPS (Apple Silicon GPU)
Loading GPT-2 tokenizer...
  Loaded 4109 jokes from training_data/dad_jokes_train.jsonl
  Loaded 457 jokes from training_data/dad_jokes_validation.jsonl
Model parameters: ...
Training config:
  Epochs: 40
  ...
```
And soon after, step-level loss lines:
```
  Step 50/5160 | Loss: 5.xxxx | LR: 0.000075 | Time: ...s
```

**Step 5: Wait for training to finish**

```bash
tail -f training_run.log
```
Training is complete when you see:
```
Training complete!
  Total time: xx.x minutes
  Best val loss: x.xxxx
  Model saved to: dad-joke-model/best_model.pt
```
Press Ctrl-C to exit tail.

**Step 6: Confirm the checkpoint was saved**

```bash
ls -lh dad-joke-model/best_model.pt
```
Expected: file exists, size ~200–700 MB depending on model size

---

## Task 3: Export to ONNX

**Step 1: Run the export script**

```bash
cd /Users/daniel/GitHub/dadJokeGenerator/.claude/worktrees/compassionate-banzai && source venv/bin/activate && python scripts/export_to_onnx.py 2>&1 | tee export_run.log
```

Expected output (key lines):
```
Loading model from dad-joke-model/best_model.pt...
Exporting to dad-joke-model/dad_joke_model.onnx...
Verifying ONNX model...
  Model size: xx.x MB
Quantizing to dad-joke-model/dad_joke_model_quantized.onnx...
  Quantized size: xx.x MB (xx% of original)
Testing ONNX inference...
  Input shape: (1, 10)
  Output shape: (1, 10, 50257)
  Inference OK!
Export complete!
```

**Step 2: Confirm ONNX artefacts**

```bash
ls -lh dad-joke-model/*.onnx*
```
Expected: `dad_joke_model.onnx`, `dad_joke_model.onnx.data` (if external data format), and `dad_joke_model_quantized.onnx`

---

## Task 4: Verify Browser Integration

The `index.html` loads the quantized ONNX model. Confirm the path in `index.html` matches the exported file.

**Step 1: Check the model path referenced in index.html**

```bash
grep -n "onnx" index.html | grep -i "model\|path\|src\|url"
```

The path should point to `dad-joke-model/dad_joke_model_quantized.onnx` (or similar). If it points to a different filename, update it.

**Step 2: Serve locally and test**

```bash
cd /Users/daniel/GitHub/dadJokeGenerator/.claude/worktrees/compassionate-banzai && python -m http.server 8765 &
echo "Server running at http://localhost:8765"
```

Open `http://localhost:8765` in a browser and:
1. Click "Get a Joke" — should work (API / local fallback)
2. Click the "AI Mode" toggle (if present) — model should load and generate a joke
3. Open DevTools console — no errors, especially no ONNX Runtime errors

**Step 3: Stop the server**

```bash
kill $(lsof -ti:8765)
```

---

## Task 5: Commit the New Model Artefacts

> **Note:** `.pt` and large `.onnx` files may be gitignored. Commit only what git will accept. The important commit is the code; the model files live on disk.

**Step 1: Check what's new/changed**

```bash
git status
```

**Step 2: Stage any new script or config changes (not large binary model files if gitignored)**

If `training_run.log` or `export_run.log` exist and are not gitignored, delete them — they're noisy:
```bash
rm -f training_run.log export_run.log
```

**Step 3: Commit if there are stageable changes**

If `git status` shows modified or new tracked files (e.g. an updated `AI_IMPLEMENTATION.md` or `README.md`), stage and commit:
```bash
git add <any tracked changed files>
git commit -m "$(cat <<'EOF'
chore: clean up post-training artefacts

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

If there's nothing to commit beyond what was done in Task 1, skip this task.

---

## Task 6: Create PR and Merge to Main

**Step 1: Push the feature branch**

```bash
git push -u origin custom-joke-model-v2
```

**Step 2: Create the PR**

```bash
gh pr create \
  --base main \
  --head custom-joke-model-v2 \
  --title "feat: improved transformer model — fixed tokenization, deeper architecture, warmup LR" \
  --body "$(cat <<'EOF'
## Summary

- **Fix tokenization bug**: `tokenizer.encode(text + eos_token)` was appending the string `"<|endoftext|>"` rather than the integer EOS token id, meaning the model never learned a clean stop condition. Fixed to `tokenizer.encode(text) + [eos_id]`.
- **Deeper model**: Increased transformer layers 6→8 and dropout 0.1→0.15 for a more expressive, slightly more regularized architecture (~30M params).
- **Better training schedule**: Extended epochs 20→40, early-stopping patience 5→10, replaced plain cosine annealing with linear warmup (200 steps) + cosine decay.
- **Hallucination filtering in browser**: Post-process generated jokes to truncate at first sentence-ending punctuation and collapse repeated punctuation (`!!!` → `!`).

## Test plan

- [ ] Training completed successfully (val loss logged in `training_run.log`)
- [ ] ONNX export produced `dad_joke_model.onnx` and `dad_joke_model_quantized.onnx`
- [ ] ONNX inference test passed (shape: `(1, 10, 50257)`)
- [ ] `http://localhost:8765` loads without console errors
- [ ] AI Mode generates coherent dad jokes (Q/A format, no trailing hallucinations)
- [ ] Regular (API/local) mode still works

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed — note it.

**Step 3: Verify CI (if any)**

```bash
gh pr checks
```
If checks pass (or there are none), proceed.

**Step 4: Merge the PR**

```bash
gh pr merge --merge --delete-branch
```

Use `--merge` (not squash or rebase) to preserve the commit history on main.

**Step 5: Verify merge**

```bash
git checkout main && git pull && git log --oneline -5
```

Expected: the "feat: improve model training" commit appears at the top of `main`.
