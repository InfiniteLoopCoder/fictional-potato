"""
Download and split MBPP dataset into train/validation sets
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset
from tqdm import tqdm


def load_mbpp_dataset(cache_dir: str = "./cache") -> Dict:
    """Load MBPP dataset from HuggingFace"""
    print("Loading MBPP dataset...")

    # Load the full dataset
    dataset = load_dataset("mbpp", "full", cache_dir=cache_dir, trust_remote_code=True)

    # MBPP has train, validation, test, and prompt splits
    # We'll use train+validation for our training, and test for final evaluation
    return dataset


def format_mbpp_example(example: Dict) -> Dict:
    """Format MBPP example into standard format"""
    return {
        "task_id": example.get("task_id", example.get("id", 0)),
        "prompt": example["text"],
        "canonical_solution": example["code"],
        "test_list": example["test_list"],
        "test_setup_code": example.get("test_setup_code", ""),
        "challenge_test_list": example.get("challenge_test_list", []),
    }


def create_train_val_split(
    dataset: Dict,
    train_ratio: float = 0.8,
    seed: int = 42,
    max_train_samples: int = None,
    max_val_samples: int = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split MBPP dataset into training and validation sets

    Args:
        dataset: MBPP dataset
        train_ratio: Ratio of data to use for training
        seed: Random seed
        max_train_samples: Maximum training samples (None for all)
        max_val_samples: Maximum validation samples (None for all)

    Returns:
        Tuple of (train_data, val_data)
    """
    random.seed(seed)

    # Combine train and validation splits for our training
    # Reserve test split for final evaluation
    train_examples = []

    if "train" in dataset:
        train_examples.extend([format_mbpp_example(ex) for ex in dataset["train"]])

    if "validation" in dataset:
        train_examples.extend([format_mbpp_example(ex) for ex in dataset["validation"]])

    if "prompt" in dataset:
        train_examples.extend([format_mbpp_example(ex) for ex in dataset["prompt"]])

    # Shuffle and split
    random.shuffle(train_examples)

    split_idx = int(len(train_examples) * train_ratio)
    train_data = train_examples[:split_idx]
    val_data = train_examples[split_idx:]

    # Apply max samples limits
    if max_train_samples is not None:
        train_data = train_data[:max_train_samples]

    if max_val_samples is not None:
        val_data = val_data[:max_val_samples]

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    return train_data, val_data


def get_test_split(dataset: Dict) -> List[Dict]:
    """Get held-out test split for final evaluation"""
    if "test" in dataset:
        test_data = [format_mbpp_example(ex) for ex in dataset["test"]]
        print(f"Test samples: {len(test_data)}")
        return test_data
    return []


def save_split(data: List[Dict], filepath: Path):
    """Save dataset split to JSONL file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        for example in tqdm(data, desc=f"Saving to {filepath.name}"):
            f.write(json.dumps(example) + '\n')

    print(f"Saved {len(data)} examples to {filepath}")


def main():
    """Main function to download and split MBPP dataset"""
    from config import get_default_config

    config = get_default_config()

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
    data_dir = Path("./data")
    save_split(train_data, data_dir / "mbpp_train.jsonl")
    save_split(val_data, data_dir / "mbpp_validation.jsonl")

    if test_data:
        save_split(test_data, data_dir / "mbpp_test.jsonl")

    print("\nDataset splitting completed!")
    print(f"Files saved in: {data_dir.absolute()}")


if __name__ == "__main__":
    main()
