#!/usr/bin/env python3
"""
Build the v3 training set: v2's pipeline (curated + synthetic, deduped,
opener-balanced) but with synthetic jokes filtered by the LLM judge scores
from judge_jokes.py. Curated human-written jokes are always kept.

Usage:
    python scripts/judge_jokes.py          # once; writes judge_scores.jsonl
    python scripts/prepare_data_v3.py [--min-score 4]

Outputs:
    training_data/v3_train.jsonl
    training_data/v3_validation.jsonl
"""

import argparse
import json
import random
from collections import Counter, defaultdict

import prepare_data_v2 as v2

SCORES = "training_data/judge_scores.jsonl"
OUT_TRAIN = "training_data/v3_train.jsonl"
OUT_VAL = "training_data/v3_validation.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=int, default=4, help="keep synthetic jokes scored >= this (1-5)")
    args = ap.parse_args()
    random.seed(v2.SEED)

    scores = {}
    for line in open(SCORES):
        if line.strip():
            d = json.loads(line)
            scores[d["text"]] = d["score"]
    dist = Counter(scores.values())
    print(f"  judge scores: {len(scores)} jokes; " + ", ".join(f"{s}:{dist.get(s,0)}" for s in range(5, 0, -1)))

    synthetic = v2.load_v1()
    kept_syn = [t for t in synthetic if scores.get(t, 0) >= args.min_score]
    unscored = sum(1 for t in synthetic if t not in scores)
    print(f"  synthetic: {len(synthetic)} -> {len(kept_syn)} with score >= {args.min_score} ({unscored} unscored, dropped)")

    curated = v2.load_curated()

    seen_text, seen_q, kept = set(), set(), []
    for t in curated + kept_syn:
        nt, nq = v2.norm(t), v2.question_of(t)
        if nt in seen_text or nq in seen_q:
            continue
        seen_text.add(nt); seen_q.add(nq); kept.append(t)
    print(f"  after dedupe: {len(kept)}")

    by_opener = defaultdict(list)
    for t in kept:
        by_opener[" ".join(v2.norm(t).split()[1:4])].append(t)
    cap = int(len(kept) * v2.MAX_OPENER_SHARE)
    balanced = []
    for opener, items in by_opener.items():
        random.shuffle(items)
        if len(items) > cap:
            print(f"  capping opener '{opener}': {len(items)} -> {cap}")
            items = items[:cap]
        balanced.extend(items)
    random.shuffle(balanced)

    n_val = int(len(balanced) * v2.VAL_FRACTION)
    val, train = balanced[:n_val], balanced[n_val:]
    with open(OUT_TRAIN, "w") as f:
        for t in train:
            f.write(json.dumps({"text": t}) + "\n")
    with open(OUT_VAL, "w") as f:
        for t in val:
            f.write(json.dumps({"text": t}) + "\n")
    print(f"\n  wrote {len(train)} -> {OUT_TRAIN}")
    print(f"  wrote {len(val)} -> {OUT_VAL}")


if __name__ == "__main__":
    print("Preparing v3 training data (judge-filtered)")
    main()
