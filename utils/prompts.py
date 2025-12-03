"""
Unified prompting strategies for code generation

IMPORTANT: All prompts should be consistent across phases:
- Phase 2 (Teacher synthesis)
- Phase 3 (GRPO training)
- Phase 4 (Evaluation)

This ensures the model learns and uses the same format throughout.
"""


# Unified format that works across all phases
UNIFIED_PROMPT_TEMPLATE = """Solve the following problem:

{problem}

Think through the solution step-by-step in <think> tags, then provide the Python code in markdown format.

Format:
<think>
Your step-by-step reasoning here...
</think>

```python
# Your code here
```"""


def get_unified_prompt(task_description: str, dataset_type: str = "mbpp") -> str:
    """
    Create unified prompt for code generation (RECOMMENDED)

    This prompt:
    - Requests thinking in <think> blocks
    - Requests code in markdown ```python```
    - Works consistently across teacher/student/eval

    Args:
        task_description: The problem description
        dataset_type: Type of dataset ("mbpp" or "apps")

    Returns:
        Formatted prompt string
    """
    return UNIFIED_PROMPT_TEMPLATE.format(problem=task_description)


def get_code_only_prompt(task_description: str, dataset_type: str = "mbpp") -> str:
    """
    Create prompt for code-only output (NO thinking blocks)

    Use this ONLY if you don't want thinking blocks.
    Otherwise, use get_unified_prompt() for consistency.

    Args:
        task_description: The problem description
        dataset_type: Type of dataset ("mbpp" or "apps")

    Returns:
        Formatted prompt string
    """
    return f"""Solve the following problem:

{task_description}

Provide the Python code solution in markdown format:

```python
# Your code here
```

Important: Output ONLY the code in markdown format, no explanations."""


def get_thinking_prompt(task_description: str, dataset_type: str = "mbpp") -> str:
    """
    DEPRECATED: Use get_unified_prompt() instead for consistency.

    This is kept for backward compatibility but should not be used in new code.
    """
    return get_unified_prompt(task_description, dataset_type)


# System prompts - UNIFIED across all phases
SYSTEM_PROMPT_UNIFIED = """You are an expert programmer. When solving problems:
1. Think through the solution step-by-step inside <think></think> tags
2. Provide the Python code in markdown format: ```python ... ```

Be clear and thorough in your reasoning."""

# Legacy system prompts (deprecated, use SYSTEM_PROMPT_UNIFIED)
SYSTEM_PROMPT_CODE_ONLY = """You are an expert programmer. Provide Python code solutions in markdown format: ```python ... ```"""

SYSTEM_PROMPT_WITH_THINKING = SYSTEM_PROMPT_UNIFIED  # Use unified version

SYSTEM_PROMPT_APPS = SYSTEM_PROMPT_UNIFIED  # Use unified version
