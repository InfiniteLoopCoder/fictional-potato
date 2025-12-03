#!/usr/bin/env python
"""
Phase 3: Train Student Model with GRPO

Trains student model using LoRA and Dual-Source objective:
- GRPO loss (execution feedback)
- Teacher-SFT loss (reasoning traces)
- Self-SFT loss (best samples)
"""
import argparse
from pathlib import Path

from config import get_default_config
from training.grpo_trainer import GRPOTrainer
from utils.dataset_loader import load_dataset_from_file


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Train with GRPO")
    parser.add_argument(
        "--train_data",
        type=str,
        default="./data/train.jsonl",
        help="Path to training data"
    )
    parser.add_argument(
        "--teacher_traces",
        type=str,
        default="./data/teacher_traces.jsonl",
        help="Path to teacher traces"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="./data/val.jsonl",
        help="Path to validation data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="LoRA rank (overrides config)"
    )

    args = parser.parse_args()
    config = get_default_config()

    # Apply argument overrides
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.lora_rank:
        config.lora.r = args.lora_rank
        config.lora.lora_alpha = args.lora_rank * 2
    if args.output_dir:
        config.training.output_dir = args.output_dir

    print("=" * 70)
    print("Phase 3: GRPO Training")
    print("=" * 70)
    print(f"Student Model: {config.student.model_name}")
    print(f"LoRA Rank: {config.lora.r}")
    print(f"LoRA Alpha: {config.lora.lora_alpha}")
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config.training.num_train_epochs}")
    print(f"  Batch size: {config.training.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"\nDual-Source Loss Weights:")
    print(f"  GRPO:        {config.dual_source.grpo_weight}")
    print(f"  Teacher-SFT: {config.dual_source.teacher_sft_weight}")
    print(f"  Self-SFT:    {config.dual_source.self_sft_weight}")
    print(f"\nGRPO Configuration:")
    print(f"  KL coefficient: {config.grpo.kl_coef}")
    print(f"  Clip range: {config.grpo.clip_range}")
    print(f"  Samples per prompt: {config.grpo.num_samples_per_prompt}")
    print("=" * 70)

    # Load data
    print("\nLoading training data...")
    train_data = load_dataset_from_file(args.train_data)
    print(f"  Training examples: {len(train_data)}")

    print("Loading teacher traces...")
    teacher_data = load_dataset_from_file(args.teacher_traces)
    print(f"  Teacher traces: {len(teacher_data)}")

    val_data = None
    if Path(args.val_data).exists():
        print("Loading validation data...")
        val_data = load_dataset_from_file(args.val_data)
        print(f"  Validation examples: {len(val_data)}")

    # Initialize trainer
    print("\n" + "-" * 70)
    print("Initializing GRPO Trainer...")
    print("-" * 70)

    trainer = GRPOTrainer(config)

    # Train
    print("\n" + "-" * 70)
    print("Starting Training...")
    print("-" * 70)

    trainer.train(
        train_data=train_data,
        teacher_data=teacher_data,
        val_data=val_data,
    )

    print("\n" + "=" * 70)
    print("✓ Phase 3 Complete!")
    print("=" * 70)
    print(f"Model saved to: {config.training.output_dir}/final_model")
    print("=" * 70)

    print("\nNext step:")
    print("  python phase4_evaluate.py")


if __name__ == "__main__":
    main()
