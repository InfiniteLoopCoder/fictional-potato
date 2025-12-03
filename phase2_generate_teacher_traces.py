#!/usr/bin/env python
"""
Phase 2: Generate Teacher Traces

Queries teacher model (vLLM) to generate reasoning traces with improved prompting.
Uses proper code parsing to extract clean code from responses.
"""
import asyncio
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from config import get_default_config
from synthesis.teacher_query import TeacherModelClient
from utils.prompts import get_unified_prompt, SYSTEM_PROMPT_UNIFIED
from utils.thinking_extraction import extract_thinking_and_code_unified
from utils.code_parser import validate_python_code
from utils.dataset_loader import load_dataset_from_file


async def generate_traces(
    config,
    train_data_path: str,
    output_path: str,
    num_samples: int = 8,
    max_tasks: int = None,
):
    """
    Generate teacher traces with improved prompting and parsing

    Args:
        config: Configuration
        train_data_path: Path to training data
        output_path: Output path for traces
        num_samples: Samples per problem
        max_tasks: Max tasks to process (None for all)
    """
    # Load training data
    print(f"Loading training data from {train_data_path}...")
    train_data = load_dataset_from_file(train_data_path)

    if max_tasks:
        train_data = train_data[:max_tasks]
        print(f"Limited to {max_tasks} tasks for testing")

    print(f"Generating traces for {len(train_data)} tasks...")

    # Initialize teacher client
    teacher_client = TeacherModelClient(config.teacher)

    # Prepare prompts with UNIFIED format (thinking + markdown)
    prompts = []
    for task in train_data:
        prompt = get_unified_prompt(
            task["prompt"],
            dataset_type=task.get("dataset", "mbpp")
        )
        prompts.append(prompt)

    # Query teacher model with UNIFIED system prompt
    print(f"\nQuerying teacher model ({num_samples} samples per task)...")
    all_responses = await teacher_client.query_batch(
        prompts=prompts,
        system_prompt=SYSTEM_PROMPT_UNIFIED,
        num_samples_per_prompt=num_samples,
    )

    # Parse responses and create training examples
    print("\nParsing responses and extracting code...")
    synthetic_examples = []
    parse_stats = {
        "total": 0,
        "with_thinking": 0,
        "with_code": 0,
        "valid_code": 0,
        "parse_failures": 0,
    }

    for task, responses in tqdm(zip(train_data, all_responses), total=len(train_data)):
        for resp in responses:
            parse_stats["total"] += 1

            # Parse response to extract thinking and code
            # Use unified extraction that properly handles Qwen's thinking format
            parsed = extract_thinking_and_code_unified(resp.full_text)

            thinking = parsed["thinking"]
            code = parsed["code"]

            # Track statistics
            if thinking:
                parse_stats["with_thinking"] += 1

            if code:
                parse_stats["with_code"] += 1

                # Validate code
                is_valid, error_msg = validate_python_code(code)
                if is_valid:
                    parse_stats["valid_code"] += 1
                else:
                    print(f"\nWarning: Invalid code for task {task['task_id']}")
                    print(f"  Error: {error_msg}")
                    print(f"  Code preview: {code[:100]}...")
            else:
                parse_stats["parse_failures"] += 1
                print(f"\nWarning: Failed to extract code for task {task['task_id']}")
                print(f"  Response preview: {resp.full_text[:200]}...")

            # Create example
            example = {
                "task_id": task["task_id"],
                "dataset": task.get("dataset", "unknown"),
                "prompt": task["prompt"],
                "thinking": thinking,
                "code": code if code else "",
                "full_response": resp.full_text,
                "teacher_metadata": resp.metadata,
                "test_list": task.get("test_list", []),
                "test_setup_code": task.get("test_setup_code", ""),
            }

            synthetic_examples.append(example)

    # Save synthetic data
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(synthetic_examples)} synthetic examples to {output_path}...")
    with open(output_file, 'w') as f:
        for example in synthetic_examples:
            f.write(json.dumps(example) + '\n')

    # Print statistics
    print("\n" + "=" * 70)
    print("✓ Phase 2 Complete!")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"\nGeneration statistics:")
    print(f"  Total responses:        {parse_stats['total']}")
    print(f"  With thinking blocks:   {parse_stats['with_thinking']} ({100*parse_stats['with_thinking']/parse_stats['total']:.1f}%)")
    print(f"  With extracted code:    {parse_stats['with_code']} ({100*parse_stats['with_code']/parse_stats['total']:.1f}%)")
    print(f"  Valid Python code:      {parse_stats['valid_code']} ({100*parse_stats['valid_code']/parse_stats['total']:.1f}%)")
    print(f"  Parse failures:         {parse_stats['parse_failures']} ({100*parse_stats['parse_failures']/parse_stats['total']:.1f}%)")
    print("=" * 70)

    print("\nNext step:")
    print("  python phase3_train_grpo.py")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Generate teacher traces")
    parser.add_argument(
        "--train_data",
        type=str,
        default="./data/train.jsonl",
        help="Path to training data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/teacher_traces.jsonl",
        help="Output path for teacher traces"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of samples per task"
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Maximum tasks to process (for testing)"
    )
    parser.add_argument(
        "--test_connection",
        action="store_true",
        help="Test teacher connection and exit"
    )

    args = parser.parse_args()
    config = get_default_config()

    # Update config with arguments
    config.teacher.num_samples_per_problem = args.num_samples

    if args.test_connection:
        print("Testing teacher connection...")
        from synthesis.teacher_query import test_teacher_connection
        asyncio.run(test_teacher_connection(config.teacher))
        return

    print("=" * 70)
    print("Phase 2: Generate Teacher Traces")
    print("=" * 70)
    print(f"Teacher API: {config.teacher.api_url}")
    print(f"Model: {config.teacher.model_name}")
    print(f"Samples per task: {args.num_samples}")
    print(f"Concurrency: {config.teacher.max_concurrent_requests}")
    print("=" * 70)

    asyncio.run(generate_traces(
        config=config,
        train_data_path=args.train_data,
        output_path=args.output,
        num_samples=args.num_samples,
        max_tasks=args.max_tasks,
    ))


if __name__ == "__main__":
    main()
