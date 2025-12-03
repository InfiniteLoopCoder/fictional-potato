#!/usr/bin/env python
"""
Phase 1: Prepare Training Data

Downloads MBPP and APPS datasets, combines them, and splits into train/val.
- 80% MBPP + 80% APPS → training set
- 20% MBPP + 20% APPS → validation set
"""
import argparse
from pathlib import Path
from config import get_default_config
from utils.dataset_loader import (
    load_mbpp_dataset,
    load_apps_dataset,
    combine_and_split_datasets,
    save_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Prepare datasets")
    parser.add_argument(
        "--apps_difficulty",
        type=str,
        default="all",
        choices=["all", "introductory", "interview", "competition"],
        help="APPS difficulty level to include"
    )
    parser.add_argument(
        "--skip_apps",
        action="store_true",
        help="Skip APPS dataset, use only MBPP"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting"
    )

    args = parser.parse_args()
    config = get_default_config()

    print("=" * 70)
    print("Phase 1: Data Preparation")
    print("=" * 70)

    # Load MBPP
    mbpp_examples = load_mbpp_dataset(cache_dir=config.data.cache_dir)

    # Load APPS (if not skipped)
    if args.skip_apps:
        print("\nSkipping APPS dataset (--skip_apps)")
        apps_examples = []
    else:
        apps_examples = load_apps_dataset(
            cache_dir=config.data.cache_dir,
            difficulty=args.apps_difficulty
        )

    # Combine and split
    print("\n" + "-" * 70)
    print("Combining and splitting datasets...")
    print("-" * 70)

    train_data, val_data = combine_and_split_datasets(
        mbpp_examples=mbpp_examples,
        apps_examples=apps_examples,
        train_ratio=0.8,
        seed=args.seed,
    )

    # Save datasets
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 70)
    print("Saving datasets...")
    print("-" * 70)

    save_dataset(train_data, output_dir / "train.jsonl")
    save_dataset(val_data, output_dir / "val.jsonl")

    print("\n" + "=" * 70)
    print("✓ Phase 1 Complete!")
    print("=" * 70)
    print(f"Training data:   {output_dir / 'train.jsonl'}")
    print(f"Validation data: {output_dir / 'val.jsonl'}")
    print(f"\nTotal examples: {len(train_data) + len(val_data)}")
    print(f"  - Training:   {len(train_data)}")
    print(f"  - Validation: {len(val_data)}")

    # Print dataset composition
    mbpp_train = sum(1 for x in train_data if x['dataset'] == 'mbpp')
    apps_train = sum(1 for x in train_data if x['dataset'] == 'apps')
    mbpp_val = sum(1 for x in val_data if x['dataset'] == 'mbpp')
    apps_val = sum(1 for x in val_data if x['dataset'] == 'apps')

    print(f"\nDataset composition:")
    print(f"  Training:   {mbpp_train} MBPP + {apps_train} APPS")
    print(f"  Validation: {mbpp_val} MBPP + {apps_val} APPS")
    print("=" * 70)

    print("\nNext step:")
    print("  python phase2_generate_teacher_traces.py")


if __name__ == "__main__":
    main()
