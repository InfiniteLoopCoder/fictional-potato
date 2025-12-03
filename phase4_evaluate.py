#!/usr/bin/env python
"""
Phase 4: Evaluate Model with Pass@k

Evaluates trained model on validation set using Pass@k metrics.
"""
import argparse
from pathlib import Path

from config import get_default_config
from evaluation.pass_at_k import evaluate_pass_at_k


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Evaluate model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./outputs/final_model",
        help="Path to trained model"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/val.jsonl",
        help="Path to test/validation data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/evaluation_results.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples per task (overrides config)"
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=None,
        help="k values for Pass@k (e.g., 1 5 10)"
    )

    args = parser.parse_args()
    config = get_default_config()

    # Apply overrides
    if args.num_samples:
        config.evaluation.num_samples_per_task = args.num_samples
    if args.k_values:
        config.evaluation.k_values = args.k_values

    print("=" * 70)
    print("Phase 4: Evaluation")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Test data: {args.test_data}")
    print(f"Samples per task: {config.evaluation.num_samples_per_task}")
    print(f"Pass@k values: {config.evaluation.k_values}")
    print("=" * 70)

    # Check if files exist
    if not Path(args.model_path).exists():
        print(f"\n✗ Error: Model not found at {args.model_path}")
        print("   Run phase3_train_grpo.py first")
        return

    if not Path(args.test_data).exists():
        print(f"\n✗ Error: Test data not found at {args.test_data}")
        print("   Run phase1_prepare_data.py first")
        return

    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluate_pass_at_k(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_path=args.output,
        config=config,
        num_samples_per_task=config.evaluation.num_samples_per_task,
        k_values=config.evaluation.k_values,
    )

    print("\n" + "=" * 70)
    print("✓ Phase 4 Complete!")
    print("=" * 70)
    print(f"Results saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
