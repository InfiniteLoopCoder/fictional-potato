"""
Main training script for Teacher-Guided GRPO
"""
import argparse
import json
import asyncio
from pathlib import Path

from config import get_default_config, MasterConfig
from data.download_mbpp import load_mbpp_dataset, create_train_val_split, save_split, get_test_split
from synthesis.generate_traces import generate_synthetic_data, load_synthetic_data
from training.grpo_trainer import GRPOTrainer
from evaluation.pass_at_k import evaluate_pass_at_k


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Teacher-Guided GRPO Training Pipeline")

    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "data", "synthesis", "train", "eval"],
        default="all",
        help="Which stage to run (default: all)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config JSON file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for models and results",
    )

    parser.add_argument(
        "--skip_data_download",
        action="store_true",
        help="Skip data download if already exists",
    )

    parser.add_argument(
        "--skip_synthesis",
        action="store_true",
        help="Skip teacher synthesis if synthetic data already exists",
    )

    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation on existing model",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model for evaluation",
    )

    return parser.parse_args()


def load_custom_config(config_path: str) -> MasterConfig:
    """
    Load custom configuration from JSON file

    Args:
        config_path: Path to JSON config file

    Returns:
        MasterConfig object
    """
    with open(config_path, 'r') as f:
        custom_config = json.load(f)

    # Start with default config
    config = get_default_config()

    # Update with custom values
    for key, value in custom_config.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


async def run_data_stage(config: MasterConfig, skip_download: bool = False):
    """
    Stage 1: Download and split MBPP dataset

    Args:
        config: Configuration object
        skip_download: Skip if data already exists
    """
    print("\n" + "=" * 70)
    print("STAGE 1: Data Preparation")
    print("=" * 70)

    data_dir = Path("./data")
    train_file = data_dir / "mbpp_train.jsonl"
    val_file = data_dir / "mbpp_validation.jsonl"
    test_file = data_dir / "mbpp_test.jsonl"

    if skip_download and train_file.exists() and val_file.exists():
        print("✓ Data files already exist, skipping download...")
        return

    # Load dataset
    dataset = load_mbpp_dataset(cache_dir=config.data.cache_dir)

    # Create splits
    train_data, val_data = create_train_val_split(
        dataset,
        train_ratio=config.data.train_split_ratio,
        seed=config.data.seed,
        max_train_samples=config.data.max_train_samples,
        max_val_samples=config.data.max_val_samples,
    )

    test_data = get_test_split(dataset)

    # Save splits
    save_split(train_data, train_file)
    save_split(val_data, val_file)

    if test_data:
        save_split(test_data, test_file)

    print("✓ Data preparation complete!")


async def run_synthesis_stage(config: MasterConfig, skip_synthesis: bool = False):
    """
    Stage 2: Generate synthetic reasoning traces from teacher

    Args:
        config: Configuration object
        skip_synthesis: Skip if synthetic data already exists
    """
    print("\n" + "=" * 70)
    print("STAGE 2: Teacher Synthesis")
    print("=" * 70)

    synthetic_path = Path(config.data.synthetic_data_path)

    if skip_synthesis and synthetic_path.exists():
        print("✓ Synthetic data already exists, skipping synthesis...")
        return

    # Generate synthetic data
    await generate_synthetic_data(
        config=config,
        train_data_path="./data/mbpp_train.jsonl",
        output_path=config.data.synthetic_data_path,
        max_samples=config.data.max_train_samples,
    )

    print("✓ Teacher synthesis complete!")


def run_training_stage(config: MasterConfig):
    """
    Stage 3: Train student model with GRPO + Dual-Source objective

    Args:
        config: Configuration object
    """
    print("\n" + "=" * 70)
    print("STAGE 3: GRPO Training")
    print("=" * 70)

    # Load training data
    print("Loading training data...")
    train_data = []
    with open("./data/mbpp_train.jsonl", 'r') as f:
        for line in f:
            train_data.append(json.loads(line))

    # Load teacher synthetic data
    print("Loading teacher synthetic data...")
    teacher_data = load_synthetic_data(config.data.synthetic_data_path)

    # Load validation data
    val_data = []
    val_file = Path("./data/mbpp_validation.jsonl")
    if val_file.exists():
        with open(val_file, 'r') as f:
            for line in f:
                val_data.append(json.loads(line))

    # Initialize trainer
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(config)

    # Train
    trainer.train(
        train_data=train_data,
        teacher_data=teacher_data,
        val_data=val_data if val_data else None,
    )

    print("✓ Training complete!")


def run_evaluation_stage(config: MasterConfig, model_path: str = None):
    """
    Stage 4: Evaluate trained model with Pass@k

    Args:
        config: Configuration object
        model_path: Path to model (default: use config.training.output_dir)
    """
    print("\n" + "=" * 70)
    print("STAGE 4: Evaluation")
    print("=" * 70)

    if model_path is None:
        model_path = str(Path(config.training.output_dir) / "final_model")

    test_data_path = "./data/mbpp_test.jsonl"

    # Check if test file exists, else use validation
    if not Path(test_data_path).exists():
        test_data_path = "./data/mbpp_validation.jsonl"
        print(f"Using validation set for evaluation: {test_data_path}")

    output_path = str(Path(config.training.output_dir) / "evaluation_results.json")

    # Run evaluation
    results = evaluate_pass_at_k(
        model_path=model_path,
        test_data_path=test_data_path,
        output_path=output_path,
        config=config,
        num_samples_per_task=config.evaluation.num_samples_per_task,
        k_values=config.evaluation.k_values,
    )

    print("✓ Evaluation complete!")

    return results


async def main():
    """Main entry point"""
    args = parse_args()

    # Load configuration
    if args.config:
        print(f"Loading custom config from {args.config}...")
        config = load_custom_config(args.config)
    else:
        print("Using default configuration...")
        config = get_default_config()

    # Override output dir if specified
    if args.output_dir:
        config.training.output_dir = args.output_dir

    # Print configuration summary
    print("\n" + "=" * 70)
    print("Configuration Summary")
    print("=" * 70)
    print(f"Teacher Model: {config.teacher.model_name}")
    print(f"Teacher API: {config.teacher.api_url}")
    print(f"Student Model: {config.student.model_name}")
    print(f"LoRA Rank: {config.lora.r}")
    print(f"GRPO KL Coef: {config.grpo.kl_coef}")
    print(f"Teacher-SFT Weight: {config.dual_source.teacher_sft_weight}")
    print(f"Self-SFT Weight: {config.dual_source.self_sft_weight}")
    print(f"Output Directory: {config.training.output_dir}")
    print("=" * 70)

    # Run appropriate stages
    if args.eval_only:
        run_evaluation_stage(config, model_path=args.model_path)
        return

    if args.stage == "all":
        # Run full pipeline
        await run_data_stage(config, skip_download=args.skip_data_download)
        await run_synthesis_stage(config, skip_synthesis=args.skip_synthesis)
        run_training_stage(config)
        run_evaluation_stage(config)

    elif args.stage == "data":
        await run_data_stage(config, skip_download=args.skip_data_download)

    elif args.stage == "synthesis":
        await run_synthesis_stage(config, skip_synthesis=args.skip_synthesis)

    elif args.stage == "train":
        run_training_stage(config)

    elif args.stage == "eval":
        run_evaluation_stage(config, model_path=args.model_path)

    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
