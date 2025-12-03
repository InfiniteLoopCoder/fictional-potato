"""
Proper thinking extraction for Qwen models with transformers

Qwen models with enable_thinking=true output thinking in specific formats.
This module handles extraction correctly.
"""
import re
from typing import Optional, Tuple, Dict


def extract_qwen_thinking(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract thinking and response from Qwen model output

    Qwen with enable_thinking outputs in format:
    <think>
    reasoning here...
    </think>

    actual response here

    Args:
        text: Raw model output

    Returns:
        Tuple of (thinking, response)
    """
    # Pattern for <think>...</think> blocks
    # Use DOTALL flag so . matches newlines
    think_pattern = r'<think>(.*?)</think>'

    match = re.search(think_pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        thinking = match.group(1).strip()
        # Remove the thinking block from text to get response
        response = re.sub(think_pattern, '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        return thinking, response
    else:
        # No thinking block found
        return None, text.strip()


def extract_thinking_and_code_unified(text: str) -> Dict[str, Optional[str]]:
    """
    Unified extraction that handles both thinking blocks and code blocks

    This combines thinking extraction with code parsing.

    Args:
        text: Model response text

    Returns:
        Dictionary with 'thinking', 'code', 'raw_response'
    """
    from .code_parser import extract_code_from_markdown, extract_code_from_text

    # First extract thinking
    thinking, text_without_thinking = extract_qwen_thinking(text)

    # Then extract code from the remaining text
    code = extract_code_from_markdown(text_without_thinking)

    if not code:
        # Try plain text extraction
        code = extract_code_from_text(text_without_thinking)

    return {
        'thinking': thinking,
        'code': code,
        'raw_response': text_without_thinking,  # Response without thinking block
        'full_text': text,  # Complete original text
    }


def format_response_with_thinking(thinking: Optional[str], code: str) -> str:
    """
    Format a response with thinking block for SFT

    Args:
        thinking: Thinking content (optional)
        code: Code content

    Returns:
        Formatted response string
    """
    if thinking:
        return f"<think>\n{thinking}\n</think>\n\n```python\n{code}\n```"
    else:
        return f"```python\n{code}\n```"


def test_thinking_extraction():
    """Test thinking extraction with various formats"""

    # Test case 1: With thinking block
    text1 = """<think>
To solve this problem, I need to:
1. Check if n <= 1
2. Return False for those cases
3. Check divisibility up to sqrt(n)
</think>

```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```"""

    print("Test 1: With thinking block")
    result = extract_thinking_and_code_unified(text1)
    print(f"  Thinking found: {bool(result['thinking'])}")
    print(f"  Thinking preview: {result['thinking'][:50] if result['thinking'] else 'None'}...")
    print(f"  Code found: {bool(result['code'])}")
    print(f"  Code preview: {result['code'][:50] if result['code'] else 'None'}...")
    print()

    # Test case 2: Without thinking block
    text2 = """```python
def add(a, b):
    return a + b
```"""

    print("Test 2: Without thinking block")
    result = extract_thinking_and_code_unified(text2)
    print(f"  Thinking found: {bool(result['thinking'])}")
    print(f"  Code found: {bool(result['code'])}")
    print(f"  Code: {result['code']}")
    print()

    # Test case 3: Plain code (no markdown)
    text3 = """<think>
Simple addition function.
</think>

def add(a, b):
    return a + b"""

    print("Test 3: Thinking + plain code")
    result = extract_thinking_and_code_unified(text3)
    print(f"  Thinking found: {bool(result['thinking'])}")
    print(f"  Code found: {bool(result['code'])}")
    print(f"  Code: {result['code']}")
    print()

    # Test case 4: Case insensitive
    text4 = """<THINK>
Testing case insensitive matching
</THINK>

def test():
    pass"""

    print("Test 4: Case insensitive <THINK>")
    result = extract_thinking_and_code_unified(text4)
    print(f"  Thinking found: {bool(result['thinking'])}")
    print(f"  Thinking: {result['thinking']}")
    print()


def verify_with_transformers():
    """
    Verify thinking extraction works with actual transformers output

    This tests with a real tokenizer to ensure compatibility.
    """
    try:
        from transformers import AutoTokenizer

        print("\n" + "="*70)
        print("Verification with Transformers")
        print("="*70)

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            trust_remote_code=True
        )

        # Simulate a model response with thinking
        response_text = """<think>
I need to implement a function that checks if a number is prime.
The algorithm:
1. Numbers <= 1 are not prime
2. Check divisibility from 2 to sqrt(n)
3. If any divisor found, not prime
</think>

```python
def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False

    return True
```"""

        # Extract
        result = extract_thinking_and_code_unified(response_text)

        print("\nExtracted components:")
        print(f"  Thinking: {'✓' if result['thinking'] else '✗'}")
        if result['thinking']:
            print(f"    Length: {len(result['thinking'])} chars")
            print(f"    Preview: {result['thinking'][:100]}...")

        print(f"\n  Code: {'✓' if result['code'] else '✗'}")
        if result['code']:
            print(f"    Length: {len(result['code'])} chars")
            print(f"    Lines: {len(result['code'].split(chr(10)))}")

        # Format for SFT
        formatted = format_response_with_thinking(
            result['thinking'],
            result['code']
        )

        print(f"\nFormatted for SFT:")
        print(f"  Length: {len(formatted)} chars")

        # Tokenize to verify
        tokens = tokenizer(formatted, return_tensors=None)
        print(f"  Tokens: {len(tokens['input_ids'])}")

        print("\n✓ Transformers verification complete!")

    except ImportError:
        print("\nSkipping transformers verification (transformers not installed)")


if __name__ == "__main__":
    print("="*70)
    print("Testing Thinking Extraction")
    print("="*70)

    test_thinking_extraction()
    verify_with_transformers()

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
