#!/usr/bin/env python3
"""
Evaluate the custom dad joke transformer.

Generates N jokes, validates each with JokeValidator, and reports:
- Validation pass rate
- Sample valid/invalid jokes
- Failure breakdown

Usage:
    python scripts/evaluate_custom_model.py [checkpoint_path] [n_samples]

Examples:
    python scripts/evaluate_custom_model.py
    python scripts/evaluate_custom_model.py dad-joke-model/best_model.pt 200
"""

import sys
import torch
from transformers import GPT2Tokenizer
from pathlib import Path

from model import DadJokeTransformer, DadJokeConfig
from joke_validator import JokeValidator


DEFAULT_CHECKPOINT = "dad-joke-model/best_model.pt"
DEFAULT_SAMPLES = 100


def load_model(checkpoint_path):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    config = DadJokeConfig()
    # Restore config if saved
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        for k, v in saved_config.items():
            if hasattr(config, k):
                setattr(config, k, v)

    model = DadJokeTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(False)
    return model, config


def generate_joke(model, tokenizer, temperature=0.8, top_k=50, top_p=0.9):
    """Generate a single dad joke starting from 'Q:'"""
    prompt = tokenizer.encode("Q:")
    input_ids = torch.tensor([prompt], dtype=torch.long)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=80,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text.strip()


def run_evaluation(checkpoint_path, n_samples=100):
    """Generate jokes and evaluate quality"""

    model, config = load_model(checkpoint_path)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    validator = JokeValidator()

    print(f"\nGenerating {n_samples} jokes for evaluation...")
    print("=" * 70)

    valid_jokes = []
    invalid_jokes = []
    failure_counts = {}

    for i in range(n_samples):
        joke = generate_joke(model, tokenizer)
        is_valid, failures = validator.validate(joke)

        if is_valid:
            valid_jokes.append(joke)
        else:
            invalid_jokes.append({'joke': joke, 'failures': failures})
            for f in failures:
                failure_counts[f] = failure_counts.get(f, 0) + 1

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_samples}...")

    # Results
    pass_rate = len(valid_jokes) / n_samples * 100

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total generated: {n_samples}")
    print(f"Passed validation: {len(valid_jokes)} ({pass_rate:.1f}%)")
    print(f"Failed validation: {len(invalid_jokes)} ({100 - pass_rate:.1f}%)")

    if failure_counts:
        print("\nFailure breakdown:")
        for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            pct = count / len(invalid_jokes) * 100 if invalid_jokes else 0
            print(f"  {reason}: {count} ({pct:.0f}% of failures)")

    print("\n" + "=" * 70)
    print("SAMPLE VALID JOKES:")
    print("=" * 70)
    for i, joke in enumerate(valid_jokes[:10], 1):
        print(f"\n  {i}. {joke}")

    if invalid_jokes:
        print("\n" + "=" * 70)
        print("SAMPLE INVALID JOKES:")
        print("=" * 70)
        for i, item in enumerate(invalid_jokes[:5], 1):
            print(f"\n  {i}. {item['joke']}")
            print(f"     Reason: {', '.join(item['failures'])}")

    print("\n" + "=" * 70)
    if pass_rate >= 80:
        print("PASS - Model meets quality threshold (80%+)")
    else:
        print("FAIL - Model below quality threshold (80%)")
        print("  Consider: more training data, more epochs, or hyperparameter tuning")
    print("=" * 70)

    return pass_rate, valid_jokes, invalid_jokes


def main():
    checkpoint_path = DEFAULT_CHECKPOINT
    n_samples = DEFAULT_SAMPLES

    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    if len(sys.argv) > 2:
        n_samples = int(sys.argv[2])

    if not Path(checkpoint_path).exists():
        print(f"Model not found: {checkpoint_path}")
        print("\nTrain first: python scripts/train_from_scratch.py")
        sys.exit(1)

    run_evaluation(checkpoint_path, n_samples)


if __name__ == "__main__":
    main()
