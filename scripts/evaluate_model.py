#!/usr/bin/env python3
"""
Evaluate fine-tuned model quality
Tests validation pass rate and generation quality
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from pathlib import Path
from tqdm import tqdm
from joke_validator import JokeValidator


MODEL_PATH = "./dad-joke-model/final"
NUM_SAMPLES = 100
TEMPERATURE = 0.7


def load_model(model_path):
    """Load the fine-tuned model"""
    print(f"📦 Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    return model, tokenizer


def generate_joke(model, tokenizer, prompt="Generate a dad joke.", temperature=0.7):
    """Generate a single dad joke"""

    # Format as chat message
    messages = [
        {"role": "system", "content": "You are a professional dad joke writer. Generate ONE dad joke in Q&A format with wordplay."},
        {"role": "user", "content": prompt}
    ]

    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "assistant" in generated_text.lower():
        parts = generated_text.split("assistant", 1)
        if len(parts) > 1:
            joke = parts[1].strip()
            # Remove any trailing tags
            joke = joke.split("<|")[0].strip()
            return joke

    return generated_text


def evaluate_model(model_path, num_samples=100, temperature=0.7, output_file=None):
    """Evaluate model on multiple generations"""

    # Load model
    model, tokenizer = load_model(model_path)

    # Initialize validator
    validator = JokeValidator()

    # Storage for results
    results = {
        'valid_jokes': [],
        'invalid_jokes': [],
        'all_jokes': []
    }

    stats = {
        'total': num_samples,
        'passed': 0,
        'failed': 0,
        'failures': {}
    }

    print(f"\n🧪 Generating {num_samples} jokes for evaluation...")
    print(f"Temperature: {temperature}")
    print("=" * 80 + "\n")

    # Generate and validate jokes
    for i in tqdm(range(num_samples), desc="Generating jokes"):
        joke = generate_joke(model, tokenizer, temperature=temperature)

        is_valid, failures = validator.validate(joke)

        results['all_jokes'].append({
            'joke': joke,
            'valid': is_valid,
            'failures': failures
        })

        if is_valid:
            stats['passed'] += 1
            results['valid_jokes'].append(joke)
        else:
            stats['failed'] += 1
            results['invalid_jokes'].append({
                'joke': joke,
                'failures': failures
            })

            # Track failure reasons
            for failure in failures:
                stats['failures'][failure] = stats['failures'].get(failure, 0) + 1

    # Calculate pass rate
    pass_rate = (stats['passed'] / stats['total']) * 100

    # Print results
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total jokes generated: {stats['total']}")
    print(f"✅ Passed validation: {stats['passed']} ({pass_rate:.1f}%)")
    print(f"❌ Failed validation: {stats['failed']} ({100-pass_rate:.1f}%)")

    if stats['failures']:
        print("\n📋 Failure Breakdown:")
        for failure, count in sorted(stats['failures'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {failure}: {count} ({count/stats['failed']*100:.1f}% of failures)")

    print("\n" + "=" * 80)
    print("✅ Sample VALID jokes:")
    print("=" * 80)
    for i, joke in enumerate(results['valid_jokes'][:5], 1):
        print(f"\n{i}. {joke}")

    if results['invalid_jokes']:
        print("\n" + "=" * 80)
        print("❌ Sample INVALID jokes:")
        print("=" * 80)
        for i, item in enumerate(results['invalid_jokes'][:5], 1):
            print(f"\n{i}. {item['joke']}")
            print(f"   Failures: {', '.join(item['failures'])}")

    print("\n" + "=" * 80)
    print("📈 QUALITY ASSESSMENT")
    print("=" * 80)

    if pass_rate >= 95:
        print("🎉 EXCELLENT! Model quality is production-ready.")
        print("   ✓ Pass rate exceeds 95% threshold")
        print("   ✓ Ready for deployment to Hugging Face")
    elif pass_rate >= 85:
        print("⚠️  GOOD, but could be better.")
        print("   ✓ Pass rate meets minimum threshold (85%)")
        print("   ⚠ Consider additional training or prompt tuning")
    else:
        print("❌ INSUFFICIENT quality for deployment.")
        print("   ✗ Pass rate below 85% threshold")
        print("   ➜ Recommendations:")
        print("     - Train for more epochs")
        print("     - Add more training data")
        print("     - Adjust LoRA hyperparameters")
        print("     - Review rejected training samples")

    # Save detailed results if requested
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump({
                'stats': stats,
                'results': results
            }, f, indent=2)
        print(f"\n💾 Detailed results saved to: {output_path}")

    return pass_rate, results


def main():
    import sys

    model_path = MODEL_PATH
    num_samples = NUM_SAMPLES
    temperature = TEMPERATURE
    output_file = None

    # Parse command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        num_samples = int(sys.argv[2])
    if len(sys.argv) > 3:
        temperature = float(sys.argv[3])
    if len(sys.argv) > 4:
        output_file = sys.argv[4]

    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("\nPlease train the model first:")
        print("  python scripts/train_dad_joke_model.py")
        return

    # Run evaluation
    pass_rate, results = evaluate_model(
        model_path,
        num_samples=num_samples,
        temperature=temperature,
        output_file=output_file
    )

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
