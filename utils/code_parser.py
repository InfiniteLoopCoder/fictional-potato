"""
Robust code parsing and extraction utilities
"""
import re
from typing import Optional, Tuple


def extract_code_from_markdown(text: str) -> Optional[str]:
    """
    Extract code from markdown code blocks

    Args:
        text: Text potentially containing markdown code blocks

    Returns:
        Extracted code or None
    """
    # Pattern for ```python ... ``` or ``` ... ```
    patterns = [
        r'```python\s*\n(.*?)\n```',
        r'```\s*\n(.*?)\n```',
        r'```python(.*?)```',
        r'```(.*?)```',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Return the longest match (most likely the actual code)
            return max(matches, key=len).strip()

    return None


def extract_thinking_and_code(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract thinking blocks and code from response

    Args:
        text: Model response text

    Returns:
        Tuple of (thinking, code)
    """
    thinking = None
    code = None

    # Extract thinking block
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL | re.IGNORECASE)
    if think_match:
        thinking = think_match.group(1).strip()
        # Remove thinking block from text for code extraction
        text_without_thinking = re.sub(think_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    else:
        text_without_thinking = text

    # Try to extract code from markdown first
    code = extract_code_from_markdown(text_without_thinking)

    # If no markdown, try to find function/class definitions
    if not code:
        code = extract_code_from_text(text_without_thinking)

    return thinking, code


def extract_code_from_text(text: str) -> Optional[str]:
    """
    Extract code from plain text by finding function/class definitions

    Args:
        text: Plain text potentially containing code

    Returns:
        Extracted code or None
    """
    lines = text.split('\n')
    code_lines = []
    in_code = False
    min_indent = float('inf')

    for line in lines:
        stripped = line.strip()

        # Start of code block
        if stripped.startswith(('def ', 'class ', 'import ', 'from ')):
            in_code = True
            if line and not line[0].isspace():
                min_indent = 0
            else:
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)

        # Collect code lines
        if in_code:
            # Skip empty lines at the start
            if not code_lines and not stripped:
                continue

            code_lines.append(line)

            # Stop at obvious non-code
            if stripped and not any([
                stripped.startswith(('def ', 'class ', 'import ', 'from ', '@', '#')),
                stripped[0] in ' \t',  # Indented line
                stripped in ['"""', "'''"],  # Docstring markers
                any(c.isalnum() or c in '()[]{}=+-*/<>!&|,:.' for c in stripped)  # Code-like characters
            ]):
                break

    if not code_lines:
        return None

    # Remove common indentation
    if min_indent != float('inf') and min_indent > 0:
        cleaned_lines = []
        for line in code_lines:
            if line.strip():
                cleaned_lines.append(line[min_indent:] if len(line) > min_indent else line)
            else:
                cleaned_lines.append('')
        code_lines = cleaned_lines

    # Remove trailing empty lines
    while code_lines and not code_lines[-1].strip():
        code_lines.pop()

    return '\n'.join(code_lines) if code_lines else None


def clean_code(code: str) -> str:
    """
    Clean and normalize extracted code

    Args:
        code: Raw code string

    Returns:
        Cleaned code
    """
    if not code:
        return ""

    # Remove common artifacts
    code = code.strip()

    # Remove leading/trailing quotes if present
    if code.startswith(('"""', "'''")) and code.endswith(('"""', "'''")):
        code = code[3:-3].strip()

    # Remove explanatory text before code
    lines = code.split('\n')
    code_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(('def ', 'class ', 'import ', 'from ', '@')):
            code_start = i
            break

    if code_start > 0:
        code = '\n'.join(lines[code_start:])

    return code.strip()


def parse_model_response(response: str) -> dict:
    """
    Parse model response to extract all components

    Args:
        response: Raw model response text

    Returns:
        Dictionary with 'thinking', 'code', and 'raw' keys
    """
    thinking, code = extract_thinking_and_code(response)

    # If no code found yet, try just markdown extraction
    if not code:
        code = extract_code_from_markdown(response)

    # Clean the code
    if code:
        code = clean_code(code)

    # If still no code, try to extract from raw text
    if not code:
        code = extract_code_from_text(response)
        if code:
            code = clean_code(code)

    return {
        'thinking': thinking,
        'code': code,
        'raw': response
    }


def validate_python_code(code: str) -> Tuple[bool, str]:
    """
    Validate that code is syntactically correct Python

    Args:
        code: Python code string

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not code or not code.strip():
        return False, "Empty code"

    try:
        compile(code, '<string>', 'exec')
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error: {e.msg} at line {e.lineno}"
    except Exception as e:
        return False, f"Compilation error: {str(e)}"


# Test cases
if __name__ == "__main__":
    # Test markdown extraction
    test1 = """Here's the solution:

```python
def add(a, b):
    return a + b
```

This function adds two numbers."""

    result = parse_model_response(test1)
    print("Test 1 - Markdown:")
    print(f"  Code: {result['code']}")
    print()

    # Test thinking + code
    test2 = """<think>
I need to add two numbers. This is simple arithmetic.
</think>

```python
def add(a, b):
    return a + b
```"""

    result = parse_model_response(test2)
    print("Test 2 - Thinking + Code:")
    print(f"  Thinking: {result['thinking'][:50]}...")
    print(f"  Code: {result['code']}")
    print()

    # Test plain code
    test3 = """def add(a, b):
    return a + b"""

    result = parse_model_response(test3)
    print("Test 3 - Plain Code:")
    print(f"  Code: {result['code']}")
    print()

    # Test validation
    valid, msg = validate_python_code(result['code'])
    print(f"Test 4 - Validation: {valid}, {msg}")
