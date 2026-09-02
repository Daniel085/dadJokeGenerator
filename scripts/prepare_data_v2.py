#!/usr/bin/env python3
"""
Build the v2 training set for the joke model.

Improvements over v1 (training_data/dad_jokes_{train,validation}.jsonl):
  - adds the 773 human-curated jokes from jokes.json (converted to Q:/A:)
  - removes exact and question-level duplicates
  - caps any single opener ("Why did the", "What do you call") at MAX_OPENER_SHARE
    so the model doesn't collapse onto one template
  - normalises whitespace (single line per joke)

Usage:
    python scripts/prepare_data_v2.py

Outputs:
    training_data/v2_train.jsonl
    training_data/v2_validation.jsonl
"""

import json
import random
import re
from collections import Counter, defaultdict

SEED = 42
VAL_FRACTION = 0.1
MAX_OPENER_SHARE = 0.30   # no single 3-word opener may exceed 30% of the set

V1_FILES = [
    "training_data/dad_jokes_train.jsonl",
    "training_data/dad_jokes_validation.jsonl",
]
CURATED = "jokes.json"
OUT_TRAIN = "training_data/v2_train.jsonl"
OUT_VAL = "training_data/v2_validation.jsonl"


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def clean_ws(s):
    return re.sub(r"\s+", " ", s).strip()


def load_v1():
    out = []
    for path in V1_FILES:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if "text" in d:
                    t = d["text"]
                else:
                    t = next((m["content"] for m in d.get("messages", []) if m.get("role") == "assistant"), None)
                if t:
                    out.append(clean_ws(t))
    return out


def load_curated():
    data = json.load(open(CURATED))
    items = data if isinstance(data, list) else data.get("jokes", [])
    out, skipped = [], 0
    for j in items:
        text = clean_ws(j["joke"] if isinstance(j, dict) else j)
        if text.startswith("Q:"):
            out.append(text)
            continue
        # "Why don't X? Because Y!"  ->  "Q: Why don't X? A: Because Y!"
        m = re.match(r"^(.*?\?)\s*(.+)$", text)
        if m and len(m.group(2)) > 2:
            out.append(f"Q: {m.group(1)} A: {m.group(2)}")
        else:
            skipped += 1   # one-liners without a question don't fit the Q/A format
    print(f"  curated: {len(out)} converted, {skipped} skipped (no question)")
    return out


def question_of(t):
    return norm(t.split("A:")[0])


def main():
    random.seed(SEED)

    v1 = load_v1()
    curated = load_curated()
    print(f"  v1 synthetic: {len(v1)}")

    # Dedupe: exact text, then by question. Curated jokes win ties (they're human-written).
    seen_text, seen_q = set(), set()
    kept = []
    for t in curated + v1:
        nt, nq = norm(t), question_of(t)
        if nt in seen_text or nq in seen_q:
            continue
        seen_text.add(nt)
        seen_q.add(nq)
        kept.append(t)
    print(f"  after dedupe: {len(kept)} (removed {len(curated) + len(v1) - len(kept)})")

    # Cap dominant openers
    by_opener = defaultdict(list)
    for t in kept:
        opener = " ".join(norm(t).split()[1:4])  # skip the "q" token
        by_opener[opener].append(t)
    total = len(kept)
    cap = int(total * MAX_OPENER_SHARE)
    balanced = []
    for opener, items in by_opener.items():
        random.shuffle(items)
        if len(items) > cap:
            print(f"  capping opener '{opener}': {len(items)} -> {cap}")
            items = items[:cap]
        balanced.extend(items)
    random.shuffle(balanced)
    print(f"  after balancing: {len(balanced)}")

    n_val = int(len(balanced) * VAL_FRACTION)
    val, train = balanced[:n_val], balanced[n_val:]

    with open(OUT_TRAIN, "w") as f:
        for t in train:
            f.write(json.dumps({"text": t}) + "\n")
    with open(OUT_VAL, "w") as f:
        for t in val:
            f.write(json.dumps({"text": t}) + "\n")

    print(f"\n  wrote {len(train)} -> {OUT_TRAIN}")
    print(f"  wrote {len(val)} -> {OUT_VAL}")
    openers = Counter(" ".join(norm(t).split()[1:4]) for t in train)
    print("  top openers:", openers.most_common(4))


if __name__ == "__main__":
    print("Preparing v2 training data")
    main()
