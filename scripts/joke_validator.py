#!/usr/bin/env python3
"""
JokeValidator - Quality validation for dad jokes
Ported from JavaScript implementation in index.html
"""

import re


class JokeValidator:
    """Validates dad jokes against quality criteria"""

    def __init__(self):
        # Common profanity list (expand as needed)
        self.profanity_list = [
            'damn', 'hell', 'ass', 'bastard', 'bitch', 'shit',
            'fuck', 'crap', 'piss', 'dick', 'cock', 'pussy'
        ]

        # Wordplay indicators (simplified)
        self.wordplay_indicators = [
            'why', 'what', 'how', 'when', 'where',
            'call', 'say', 'difference', 'between'
        ]

    def validate(self, joke):
        """
        Validate a joke against all criteria
        Returns: (is_valid: bool, failures: list)
        """
        failures = []

        # Check 1: Has question mark
        if not self.has_question(joke):
            failures.append("Missing question mark")

        # Check 2: Has Q&A format
        if not self.has_format(joke):
            failures.append("Missing Q&A format")

        # Check 3: Length check
        if not self.is_valid_length(joke):
            failures.append(f"Invalid length: {len(joke)} chars (need 20-200)")

        # Check 4: Profanity check
        if not self.is_clean(joke):
            failures.append("Contains profanity")

        # Check 5: Has wordplay/pun setup
        if not self.has_wordplay_setup(joke):
            failures.append("Missing wordplay setup")

        # Check 6: No meta-commentary
        if not self.no_meta_commentary(joke):
            failures.append("Contains meta-commentary")

        is_valid = len(failures) == 0
        return is_valid, failures

    def has_question(self, joke):
        """Check if joke contains a question mark"""
        return '?' in joke

    def has_format(self, joke):
        """Check if joke has Q: and A: format or question/answer structure"""
        # Explicit Q: A: format
        if re.search(r'Q:', joke, re.IGNORECASE) and re.search(r'A:', joke, re.IGNORECASE):
            return True

        # Or has question/answer structure (question mark followed by content)
        parts = joke.split('?')
        if len(parts) >= 2 and len(parts[1].strip()) > 0:
            return True

        return False

    def is_valid_length(self, joke):
        """Check if joke is appropriate length"""
        length = len(joke)
        return 20 <= length <= 200

    def is_clean(self, joke):
        """Check if joke is family-friendly (no profanity)"""
        joke_lower = joke.lower()
        for word in self.profanity_list:
            # Use word boundaries to avoid false positives
            if re.search(r'\b' + re.escape(word) + r'\b', joke_lower):
                return False
        return True

    def has_wordplay_setup(self, joke):
        """
        Check if joke has typical dad joke setup
        (simplified - real implementation would be more sophisticated)
        """
        joke_lower = joke.lower()

        # Look for common question words
        for indicator in self.wordplay_indicators:
            if indicator in joke_lower:
                return True

        # If it has Q&A format, assume it has wordplay
        if self.has_format(joke):
            return True

        return False

    def no_meta_commentary(self, joke):
        """Check that joke doesn't have meta-commentary"""
        meta_phrases = [
            "here's a joke",
            "here is a joke",
            "i made",
            "i created",
            "joke about",
            "for example",
            "another one",
            "how about",
            "try this"
        ]

        joke_lower = joke.lower()
        for phrase in meta_phrases:
            if phrase in joke_lower:
                return False

        return True

    def clean_joke(self, joke):
        """
        Clean up a joke by removing meta-commentary and fixing format
        Returns: (cleaned_joke: str, was_modified: bool)
        """
        original = joke
        cleaned = joke.strip()

        # Remove common meta prefixes
        meta_prefixes = [
            r'^Here\'s a joke:?\s*',
            r'^Here is a joke:?\s*',
            r'^Joke:?\s*',
            r'^Q:\s*',  # Sometimes duplicated
        ]

        for prefix in meta_prefixes:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)

        # Ensure Q: and A: format if missing
        if '?' in cleaned and 'Q:' not in cleaned:
            parts = cleaned.split('?', 1)
            if len(parts) == 2:
                cleaned = f"Q: {parts[0].strip()}?\nA: {parts[1].strip()}"

        # Remove trailing meta-commentary
        meta_suffixes = [
            r'\s*\(.*?\)$',  # Remove parenthetical notes
            r'\s*Hope you.*$',
            r'\s*Enjoy!.*$',
        ]

        for suffix in meta_suffixes:
            cleaned = re.sub(suffix, '', cleaned, flags=re.IGNORECASE)

        was_modified = (cleaned != original)
        return cleaned.strip(), was_modified


def test_validator():
    """Test the validator with sample jokes"""
    validator = JokeValidator()

    test_cases = [
        # Valid jokes
        ("Q: Why don't scientists trust atoms?\nA: Because they make up everything!", True),
        ("Q: What do you call a factory that makes okay products?\nA: A satisfactory!", True),
        ("Why did the scarecrow win an award? He was outstanding in his field!", True),

        # Invalid jokes
        ("This is not a joke", False),
        ("Q: Why?", False),  # Too short
        ("Q: Why don't scientists trust atoms? A: " + "x" * 200, False),  # Too long
        ("Here's a joke: Why don't scientists trust atoms? Because they make up everything!", False),  # Meta
    ]

    print("🧪 Testing JokeValidator\n")
    for joke, should_pass in test_cases:
        is_valid, failures = validator.validate(joke)
        status = "✅" if is_valid == should_pass else "❌"
        print(f"{status} Expected: {should_pass}, Got: {is_valid}")
        if failures:
            print(f"   Failures: {', '.join(failures)}")
        print(f"   Joke: {joke[:60]}...")
        print()


if __name__ == "__main__":
    test_validator()
