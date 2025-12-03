"""
Generate synthetic reasoning traces using teacher model
"""
import asyncio
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from .teacher_query import TeacherModelClient, create_code_generation_prompt


def load_training_data(filepath: str) -> List[Dict]:
    """Load training data from JSONL file"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_teacher_examples(
    task: Dict,
    teacher_responses: List,
    include_thinking: bool = True,
) -> List[Dict]:
    """
    Create training examples from teacher responses

    Args:
        task: MBPP task dictionary
        teacher_responses: List of TeacherResponse objects
        include_thinking: Whether to include thinking blocks

    Returns:
        List of training examples
    """
    examples = []

    for resp in teacher_responses:
        if include_thinking and resp.thinking:
            # Format with thinking block
            formatted_response = f"<think>\n{resp.thinking}\n</think>\n\n{resp.response}"
        else:
            formatted_response = resp.response

        example = {
            "task_id": task["task_id"],
            "prompt": task["prompt"],
            "teacher_response": formatted_response,
            "thinking": resp.thinking,
            "code": resp.response,
            "test_list": task["test_list"],
            "test_setup_code": task.get("test_setup_code", ""),
            "metadata": resp.metadata,
        }

        examples.append(example)

    return examples


async def generate_synthetic_data(
    config,
    train_data_path: str,
    output_path: str,
    max_samples: int = None,
):
    """
    Generate synthetic reasoning traces from teacher model

    Args:
        config: Configuration object
        train_data_path: Path to training data JSONL
        output_path: Path to save synthetic traces
        max_samples: Maximum number of tasks to process (None for all)
    """
    # Load training data
    print(f"Loading training data from {train_data_path}...")
    train_data = load_training_data(train_data_path)

    if max_samples:
        train_data = train_data[:max_samples]

    print(f"Generating traces for {len(train_data)} tasks...")

    # Initialize teacher client
    teacher_client = TeacherModelClient(config.teacher)

    # Prepare prompts
    prompts = [create_code_generation_prompt(task["prompt"]) for task in train_data]

    # Query teacher model in batches
    all_responses = await teacher_client.query_batch(
        prompts=prompts,
        num_samples_per_prompt=config.teacher.num_samples_per_problem,
    )

    # Create training examples
    synthetic_examples = []

    print("Processing teacher responses...")
    for task, responses in tqdm(zip(train_data, all_responses), total=len(train_data)):
        examples = create_teacher_examples(
            task,
            responses,
            include_thinking=config.dual_source.use_teacher_thinking,
        )
        synthetic_examples.extend(examples)

    # Save synthetic data
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(synthetic_examples)} synthetic examples to {output_path}...")
    with open(output_file, 'w') as f:
        for example in synthetic_examples:
            f.write(json.dumps(example) + '\n')

    print(f"✓ Synthetic data generation complete!")
    print(f"Total examples: {len(synthetic_examples)}")
    print(f"Examples per task: {config.teacher.num_samples_per_problem}")

    # Print statistics
    with_thinking = sum(1 for ex in synthetic_examples if ex["thinking"])
    print(f"Examples with thinking: {with_thinking}/{len(synthetic_examples)}")

    return synthetic_examples


def load_synthetic_data(filepath: str) -> List[Dict]:
    """Load synthetic training data from JSONL file"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


async def main():
    """Main function to generate synthetic traces"""
    from config import get_default_config

    config = get_default_config()

    await generate_synthetic_data(
        config=config,
        train_data_path="./data/mbpp_train.jsonl",
        output_path=config.data.synthetic_data_path,
        max_samples=None,  # Process all training samples
    )


if __name__ == "__main__":
    asyncio.run(main())
