#!/usr/bin/env python3
"""
Quick test script to generate sample jokes with the fine-tuned model
Interactive mode for manual quality assessment
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


MODEL_PATH = "./dad-joke-model/final"


def generate_joke(model, tokenizer, prompt=None, temperature=0.7):
    """Generate a single dad joke"""

    if prompt is None:
        prompt = "Generate a dad joke."

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


def main():
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("\nPlease train the model first:")
        print("  python scripts/train_dad_joke_model.py")
        return

    print("🤖 Loading dad joke model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("✅ Model loaded!\n")
    print("=" * 80)
    print("Interactive Dad Joke Generator")
    print("=" * 80)
    print("Commands:")
    print("  - Press Enter: Generate random dad joke")
    print("  - Type topic: Generate joke about specific topic")
    print("  - 'quit' or 'exit': Exit")
    print("=" * 80 + "\n")

    while True:
        try:
            user_input = input("🎤 Topic (or Enter for random): ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for the laughs!")
                break

            if user_input:
                prompt = f"Generate a dad joke about {user_input}."
            else:
                prompt = "Generate a dad joke."

            print("\n🎨 Generating joke...\n")

            joke = generate_joke(model, tokenizer, prompt, temperature=0.7)

            print("─" * 80)
            print(joke)
            print("─" * 80 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Thanks for the laughs!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
