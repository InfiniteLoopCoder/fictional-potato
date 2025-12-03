"""
SFT Data Preparation - Correctly construct ground truth for Teacher-SFT and Self-SFT

Key concepts:
1. For SFT, we want to predict the RESPONSE given the PROMPT
2. Labels should be -100 for prompt tokens (no loss computed)
3. Labels should be token_ids for response tokens (compute loss)
4. When using chat templates, the assistant response is what we want to learn
"""
import torch
from typing import Dict, List, Optional, Tuple


def prepare_teacher_sft_batch(
    teacher_examples: List[Dict],
    tokenizer,
    max_length: int = 2048,
    include_thinking: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Prepare batch for Teacher-SFT loss

    This creates proper ground truth labels where:
    - Prompt tokens have label -100 (ignored in loss)
    - Response tokens (thinking + code) have actual token IDs (used in loss)

    Args:
        teacher_examples: List of teacher trace examples
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        include_thinking: Whether to include thinking blocks in SFT

    Returns:
        Dictionary with input_ids, attention_mask, labels
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for example in teacher_examples:
        # Get prompt and response
        prompt = example.get("prompt", "")
        thinking = example.get("thinking", "")
        code = example.get("code", "")

        # Construct full response
        if include_thinking and thinking:
            # Include thinking in the response we want to learn
            response = f"<think>\n{thinking}\n</think>\n\n```python\n{code}\n```"
        else:
            # Only include code
            response = f"```python\n{code}\n```"

        # Format as chat messages
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]

        # Apply chat template
        # This creates: <prompt tokens> <assistant_start_token> <response tokens> <eos>
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False  # We provide the assistant response
        )

        # Tokenize the full sequence
        full_tokens = tokenizer(
            formatted,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        # Create labels: -100 for prompt, actual IDs for response
        labels = create_sft_labels(
            tokenizer=tokenizer,
            prompt=prompt,
            response=response,
            input_ids=input_ids,
        )

        input_ids_list.append(torch.tensor(input_ids))
        attention_mask_list.append(torch.tensor(attention_mask))
        labels_list.append(torch.tensor(labels))

    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask_list,
        batch_first=True,
        padding_value=0
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        labels_list,
        batch_first=True,
        padding_value=-100  # Padding is ignored in loss
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def create_sft_labels(
    tokenizer,
    prompt: str,
    response: str,
    input_ids: List[int],
) -> List[int]:
    """
    Create SFT labels where prompt tokens are -100 and response tokens are actual IDs

    The key insight: We want to compute loss ONLY on the assistant's response,
    not on the user's prompt.

    Args:
        tokenizer: Tokenizer
        prompt: User prompt text
        response: Assistant response text
        input_ids: Full sequence token IDs (prompt + response)

    Returns:
        List of label IDs (-100 for prompt, token_id for response)
    """
    # Tokenize prompt only to find where it ends
    messages_prompt_only = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages_prompt_only,
        tokenize=False,
        add_generation_prompt=True  # Add assistant start token
    )

    prompt_tokens = tokenizer(
        formatted_prompt,
        truncation=False,
        padding=False,
        return_tensors=None,
    )

    prompt_length = len(prompt_tokens["input_ids"])

    # Create labels: -100 for prompt, actual IDs for response
    labels = []
    for i, token_id in enumerate(input_ids):
        if i < prompt_length:
            # Prompt tokens: ignore in loss
            labels.append(-100)
        else:
            # Response tokens: compute loss
            labels.append(token_id)

    return labels


def prepare_self_sft_batch(
    successful_samples: List[Dict],
    tokenizer,
    prompts: List[str],
    top_k: int = 2,
    min_reward: float = 0.8,
    max_length: int = 1024,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Prepare batch for Self-SFT loss from successful rollout samples

    This takes the BEST successful samples from the current policy and
    creates SFT targets to reinforce good behavior.

    Args:
        successful_samples: Samples with rewards >= min_reward
        tokenizer: Tokenizer
        prompts: Original prompts for these samples
        top_k: Number of top samples to use per prompt
        min_reward: Minimum reward threshold
        max_length: Max sequence length

    Returns:
        Dictionary with input_ids, attention_mask, labels (or None if no samples)
    """
    if not successful_samples:
        return None

    # Filter by reward
    filtered = [s for s in successful_samples if s.get("reward", 0) >= min_reward]

    if not filtered:
        return None

    # Sort by reward and take top-k
    filtered.sort(key=lambda x: x.get("reward", 0), reverse=True)
    top_samples = filtered[:top_k]

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for sample in top_samples:
        # Get the generated response (code)
        response = sample.get("response", "")
        prompt = sample.get("prompt", "")

        if not response or not prompt:
            continue

        # Format as chat
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        full_tokens = tokenizer(
            formatted,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        # Create labels
        labels = create_sft_labels(
            tokenizer=tokenizer,
            prompt=prompt,
            response=response,
            input_ids=input_ids,
        )

        input_ids_list.append(torch.tensor(input_ids))
        attention_mask_list.append(torch.tensor(attention_mask))
        labels_list.append(torch.tensor(labels))

    if not input_ids_list:
        return None

    # Pad
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask_list,
        batch_first=True,
        padding_value=0
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        labels_list,
        batch_first=True,
        padding_value=-100
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def verify_labels(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    example_idx: int = 0,
):
    """
    Debug function to verify labels are constructed correctly

    Prints the input sequence with labels to visually inspect.
    """
    input_ids_example = input_ids[example_idx]
    labels_example = labels[example_idx]

    print(f"\n{'='*70}")
    print(f"Label Verification (Example {example_idx})")
    print(f"{'='*70}")

    prompt_tokens = []
    response_tokens = []

    for i, (token_id, label_id) in enumerate(zip(input_ids_example, labels_example)):
        token_id = token_id.item()
        label_id = label_id.item()

        token_text = tokenizer.decode([token_id])

        if label_id == -100:
            # Prompt token (no loss)
            prompt_tokens.append(token_text)
        else:
            # Response token (compute loss)
            response_tokens.append(token_text)

    print(f"\nPrompt tokens (label=-100, no loss):")
    print("".join(prompt_tokens))

    print(f"\nResponse tokens (label=token_id, compute loss):")
    print("".join(response_tokens))

    print(f"\n{'='*70}")

    # Verify no data leakage
    num_prompt_tokens = sum(1 for label in labels_example if label == -100)
    num_response_tokens = sum(1 for label in labels_example if label != -100 and label != tokenizer.pad_token_id)

    print(f"Statistics:")
    print(f"  Prompt tokens (ignored): {num_prompt_tokens}")
    print(f"  Response tokens (loss):  {num_response_tokens}")
    print(f"  Total tokens: {len(input_ids_example)}")
    print(f"{'='*70}\n")


# Example usage and test
if __name__ == "__main__":
    from transformers import AutoTokenizer

    print("Testing SFT data preparation...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test Teacher-SFT
    print("\n" + "="*70)
    print("Test 1: Teacher-SFT Data Preparation")
    print("="*70)

    teacher_examples = [
        {
            "prompt": "Write a function to add two numbers.",
            "thinking": "This is a simple addition operation. I'll create a function that takes two parameters and returns their sum.",
            "code": "def add(a, b):\n    return a + b",
        }
    ]

    batch = prepare_teacher_sft_batch(
        teacher_examples,
        tokenizer,
        include_thinking=True
    )

    print(f"Batch shape:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")

    # Verify labels
    verify_labels(batch["input_ids"], batch["labels"], tokenizer)

    # Test Self-SFT
    print("\n" + "="*70)
    print("Test 2: Self-SFT Data Preparation")
    print("="*70)

    successful_samples = [
        {
            "prompt": "Write a function to multiply two numbers.",
            "response": "def multiply(a, b):\n    return a * b",
            "reward": 1.0,
        },
        {
            "prompt": "Write a function to divide two numbers.",
            "response": "def divide(a, b):\n    return a / b",
            "reward": 0.9,
        },
    ]

    batch = prepare_self_sft_batch(
        successful_samples,
        tokenizer,
        prompts=[],
        top_k=2,
        min_reward=0.8,
    )

    if batch:
        print(f"Batch shape:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  labels: {batch['labels'].shape}")

        verify_labels(batch["input_ids"], batch["labels"], tokenizer)
    else:
        print("No successful samples found")

    print("\n✓ Tests complete!")
