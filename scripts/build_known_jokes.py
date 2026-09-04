#!/usr/bin/env python3
"""
Build known_jokes.json: hashes of every joke (and every joke *question*) the
model was trained on, so the browser can reject verbatim regurgitation and
force the AI mode to show only new material.

Larger models memorise: the 1.5B fine-tune reproduced "Why did the
scarecrow win an award? He was outstanding in his field!" verbatim 3 times
in 30 samples. Shipping the training text itself would be ~300 KB; 32-bit
FNV-1a hashes of the normalised text are ~10 bytes each.

The normalisation and hash must match `normalizeJoke` / `fnv1a` in
index.html exactly.

Usage:
    python scripts/build_known_jokes.py

Output:
    known_jokes.json   {"jokes": [..hashes..], "questions": [..hashes..]}
"""

import json
import re

SOURCES = [
    "training_data/v2_train.jsonl", "training_data/v2_validation.jsonl",
    "training_data/v3_train.jsonl", "training_data/v3_validation.jsonl",
    "training_data/dad_jokes_train.jsonl", "training_data/dad_jokes_validation.jsonl",
]
CURATED = "jokes.json"
OUT = "known_jokes.json"


def norm(s):
    # lower-case, strip Q:/A: labels, keep letters/digits/spaces, squeeze spaces
    s = s.lower()
    s = re.sub(r"\b[qa]:", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def fnv1a(s):
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def question_of(text):
    return text.split("A:")[0] if "A:" in text else text.split("?")[0]


def main():
    texts = set()
    for path in SOURCES:
        try:
            for line in open(path):
                if line.strip():
                    d = json.loads(line)
                    t = d.get("text") or next((m["content"] for m in d.get("messages", []) if m.get("role") == "assistant"), None)
                    if t:
                        texts.add(t)
        except FileNotFoundError:
            pass
    data = json.load(open(CURATED))
    for j in (data if isinstance(data, list) else data.get("jokes", [])):
        texts.add(j["joke"] if isinstance(j, dict) else j)

    jokes = sorted({fnv1a(norm(t)) for t in texts})
    questions = sorted({fnv1a(norm(question_of(t))) for t in texts})
    with open(OUT, "w") as f:
        json.dump({"jokes": jokes, "questions": questions}, f, separators=(",", ":"))
    print(f"{len(texts)} source jokes -> {len(jokes)} joke hashes, {len(questions)} question hashes -> {OUT}")


if __name__ == "__main__":
    main()
