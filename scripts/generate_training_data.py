#!/usr/bin/env python3
"""
Training Data Generation Script

This script documents the process for generating training data using Claude.
Since this script runs in the Claude conversation, it provides the prompts
and structure needed for Claude to generate high-quality dad jokes.

Usage:
    1. Run this script to see the generation workflow
    2. Use the provided prompts with Claude
    3. Validate and save the generated jokes
"""

import json
import random
from pathlib import Path
from joke_validator import JokeValidator


# Load existing jokes for few-shot examples
def load_existing_jokes():
    """Load jokes from jokes.json for use as examples"""
    jokes_file = Path(__file__).parent.parent / "jokes.json"
    with open(jokes_file, 'r') as f:
        jokes = json.load(f)
    return jokes


def get_generation_prompt(num_jokes=100, examples=None):
    """
    Generate the prompt to use with Claude for joke generation

    Args:
        num_jokes: Number of jokes to generate in this batch
        examples: List of example jokes for few-shot learning
    """
    if examples is None:
        examples = random.sample(load_existing_jokes(), 6)

    prompt = f"""You are a professional dad joke writer. I need you to generate {num_jokes} original dad jokes in the classic Q&A format with wordplay/puns.

**Format Requirements:**
- Each joke MUST follow "Q: [question] A: [punchline]" format
- Question must contain a question mark (?)
- Answer must contain wordplay, pun, or homophone
- Length: 20-200 characters total
- Family-friendly (G-rated) content only
- No meta-commentary (don't say "Here's a joke..." or explain it)

**Style Guidelines:**
- Groan-worthy but clever
- Use wordplay, puns, homophones, double meanings
- Keep it simple and accessible
- Dad joke energy: wholesome and corny

**Examples of high-quality dad jokes:**

{chr(10).join(f'{i+1}. {joke}' for i, joke in enumerate(examples))}

**Your Task:**
Generate {num_jokes} NEW original dad jokes following the exact same style and format. Output them as a JSON array where each element is a string containing one complete joke.

Format:
[
  "Q: Why don't scientists trust atoms?\\nA: Because they make up everything!",
  "Q: What do you call a factory that makes okay products?\\nA: A satisfactory!",
  ...
]

Begin generation:"""

    return prompt


def validate_batch(jokes):
    """
    Validate a batch of generated jokes

    Returns:
        (valid_jokes, invalid_jokes, stats)
    """
    validator = JokeValidator()
    valid = []
    invalid = []

    for joke in jokes:
        is_valid, failures = validator.validate(joke)
        if is_valid:
            valid.append(joke)
        else:
            invalid.append({
                'joke': joke,
                'failures': failures
            })

    stats = {
        'total': len(jokes),
        'valid': len(valid),
        'invalid': len(invalid),
        'pass_rate': (len(valid) / len(jokes) * 100) if jokes else 0
    }

    return valid, invalid, stats


def save_training_data(jokes, output_file):
    """
    Save validated jokes in training format (JSONL)

    Format for fine-tuning:
    {"messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "Generate a dad joke."},
      {"role": "assistant", "content": "Q: ... A: ..."}
    ]}
    """
    output_path = Path(__file__).parent.parent / "training_data" / output_file

    system_message = (
        "You are a professional dad joke writer. Generate ONE dad joke in Q&A format with wordplay. "
        "Requirements: Must use puns/wordplay, family-friendly (G-rated), no meta-commentary, 20-200 characters."
    )

    with open(output_path, 'w') as f:
        for joke in jokes:
            entry = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": "Generate a dad joke."},
                    {"role": "assistant", "content": joke}
                ]
            }
            f.write(json.dumps(entry) + '\n')

    print(f"✅ Saved {len(jokes)} jokes to {output_path}")


def main():
    """Main workflow for data generation"""
    print("=" * 80)
    print("🤖 Dad Joke Training Data Generation Workflow")
    print("=" * 80)

    print("\n📋 STEP 1: Prepare Few-Shot Examples")
    print("-" * 80)
    existing_jokes = load_existing_jokes()
    examples = random.sample(existing_jokes, 6)
    print(f"Loaded {len(existing_jokes)} existing jokes")
    print(f"Selected 6 random examples for few-shot prompting:\n")
    for i, joke in enumerate(examples, 1):
        print(f"{i}. {joke[:60]}...")

    print("\n\n📝 STEP 2: Generation Prompt")
    print("-" * 80)
    prompt = get_generation_prompt(num_jokes=100, examples=examples)
    print("Use this prompt with Claude:\n")
    print("=" * 80)
    print(prompt)
    print("=" * 80)

    print("\n\n🔄 STEP 3: Workflow")
    print("-" * 80)
    print("""
1. Copy the prompt above
2. Send to Claude in a conversation
3. Claude will respond with JSON array of jokes
4. Save Claude's response to a file: batch_001.json
5. Run this script with --validate flag to check quality
6. Repeat for multiple batches until you have 10,000+ jokes

**Target: 10,000-50,000 validated jokes**

Commands:
---------
# Validate a batch
python scripts/generate_training_data.py --validate batch_001.json

# Combine all validated batches into training file
python scripts/generate_training_data.py --combine

# Split into train/validation sets
python scripts/generate_training_data.py --split
    """)

    print("\n\n📊 STEP 4: Expected Results")
    print("-" * 80)
    print("""
Validation Pass Rate: 85-95% (target)
- Valid jokes will be saved to training_data/
- Invalid jokes logged to training_data/rejected.jsonl for analysis

Total Training Data Target:
- Training set: 9,000-45,000 jokes (90%)
- Validation set: 1,000-5,000 jokes (10%)
    """)

    print("\n\n💡 TIPS")
    print("-" * 80)
    print("""
1. Generate in batches of 100-500 jokes
2. Use different few-shot examples for each batch (diversity)
3. Review rejected jokes to improve prompts
4. Monitor validation pass rate - should be 85%+
5. If pass rate drops below 80%, adjust prompt
6. Include existing 750 jokes in final training set
    """)

    print("\n✅ Ready to start generation!")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--validate" and len(sys.argv) > 2:
            # Validate a batch file
            batch_file = sys.argv[2]
            with open(batch_file, 'r') as f:
                jokes = json.load(f)

            valid, invalid, stats = validate_batch(jokes)

            print(f"\n📊 Validation Results for {batch_file}")
            print(f"Total: {stats['total']}")
            print(f"✅ Valid: {stats['valid']} ({stats['pass_rate']:.1f}%)")
            print(f"❌ Invalid: {stats['invalid']}")

            if invalid:
                print("\nSample failures:")
                for item in invalid[:5]:
                    print(f"  - {item['joke'][:50]}...")
                    print(f"    Reasons: {', '.join(item['failures'])}")

            # Save valid jokes
            if valid:
                save_training_data(valid, f"validated_{Path(batch_file).stem}.jsonl")

        elif sys.argv[1] == "--combine":
            # Combine all validated batches
            print("Combining all validated batches...")
            # Implementation: combine all validated_*.jsonl files

        elif sys.argv[1] == "--split":
            # Split into train/validation
            print("Splitting into train/validation sets...")
            # Implementation: 90/10 split

    else:
        main()
