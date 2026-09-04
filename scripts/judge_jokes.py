#!/usr/bin/env python3
"""
Score every synthetic training joke with an LLM judge, so weak puns can be
filtered out of the fine-tuning set.

Uses the `claude` CLI in headless mode (`claude -p`), so no API key is
needed. Curated human-written jokes (jokes.json) are not judged; they are
kept unconditionally by prepare_data_v3.py.

Each joke gets an integer 1-5:
  5 real pun/double meaning that clearly works
  4 real pun, weak or well-worn
  3 coherent and joke-shaped, wordplay is a stretch
  2 grammatical but no actual joke
  1 nonsense / broken / made-up "pun" word

Usage:
    python scripts/judge_jokes.py [--batch 40] [--workers 4] [--model MODEL]

Output (append-only, resumable):
    training_data/judge_scores.jsonl   {"text": ..., "score": n}
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

SOURCES = ["training_data/dad_jokes_train.jsonl", "training_data/dad_jokes_validation.jsonl"]
OUT = "training_data/judge_scores.jsonl"

RUBRIC = """You are grading dad jokes for use as fine-tuning data. For each numbered joke, give an integer score 1-5:
5 = a real pun or double meaning that clearly works, grammatical, funny for a dad joke
4 = a real pun that works but is weak or well-worn
3 = coherent and joke-shaped, but the wordplay is a stretch or barely there
2 = grammatical but no actual joke (tautology, flat statement, logic doesn't connect)
1 = nonsense, broken, or the "pun" is a made-up word with no sound-alike basis

Reply with ONLY a JSON object mapping joke number (as a string) to score, e.g. {"1": 4, "2": 1}. No other text.

"""


def load_jokes():
    seen, out = set(), []
    for path in SOURCES:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("text") or next((m["content"] for m in d.get("messages", []) if m.get("role") == "assistant"), None)
            if t:
                t = re.sub(r"\s+", " ", t).strip()
                if t not in seen:
                    seen.add(t)
                    out.append(t)
    return out


def load_scored():
    if not os.path.exists(OUT):
        return {}
    scored = {}
    for line in open(OUT):
        if line.strip():
            d = json.loads(line)
            scored[d["text"]] = d["score"]
    return scored


def judge_batch(jokes, model):
    prompt = RUBRIC + "\n".join(f"{i + 1}. {j}" for i, j in enumerate(jokes))
    cmd = ["claude", "-p", prompt, "--output-format", "text"]
    if model:
        cmd += ["--model", model]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(f"claude exited {r.returncode}: {r.stderr.strip()[:300]}")
    m = re.search(r"\{[^{}]*\}", r.stdout, re.S)
    if not m:
        raise RuntimeError(f"no JSON in reply: {r.stdout[:200]!r}")
    scores = json.loads(m.group(0))
    out = {}
    for i, j in enumerate(jokes):
        s = scores.get(str(i + 1))
        if isinstance(s, int) and 1 <= s <= 5:
            out[j] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=None, help="claude CLI --model value (default: CLI default)")
    ap.add_argument("--limit", type=int, default=None, help="only judge the first N unscored jokes")
    args = ap.parse_args()

    jokes = load_jokes()
    scored = load_scored()
    todo = [j for j in jokes if j not in scored]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(jokes)} synthetic jokes, {len(scored)} already scored, {len(todo)} to judge", flush=True)
    if not todo:
        return

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    lock = threading.Lock()
    done = 0
    with open(OUT, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(judge_batch, b, args.model): b for b in batches}
        for fut in as_completed(futures):
            b = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"  batch failed ({len(b)} jokes): {e}", file=sys.stderr, flush=True)
                continue
            with lock:
                for j, s in res.items():
                    f.write(json.dumps({"text": j, "score": s}) + "\n")
                f.flush()
                done += 1
                print(f"  batch {done}/{len(batches)} scored {len(res)}/{len(b)}", flush=True)

    scored = load_scored()
    from collections import Counter
    c = Counter(scored.values())
    print(f"\nScored {len(scored)} jokes. Distribution:", flush=True)
    for s in range(5, 0, -1):
        print(f"  {s}: {c.get(s, 0):5d}  {'#' * (c.get(s, 0) // 25)}", flush=True)
    print(f"\nRe-run to retry any failed batches. Next: python scripts/prepare_data_v3.py", flush=True)


if __name__ == "__main__":
    main()
