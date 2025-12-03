"""
Unified dataset loader for MBPP and APPS
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset
from tqdm import tqdm


def load_mbpp_dataset(cache_dir: str = "./cache") -> List[Dict]:
    """
    Load MBPP dataset

    Args:
        cache_dir: Cache directory

    Returns:
        List of MBPP examples
    """
    print("Loading MBPP dataset...")
    dataset = load_dataset("mbpp", "full", cache_dir=cache_dir, trust_remote_code=True)

    examples = []

    # Combine all splits for MBPP
    for split_name in ["train", "validation", "test", "prompt"]:
        if split_name in dataset:
            for ex in dataset[split_name]:
                examples.append({
                    "dataset": "mbpp",
                    "task_id": f"mbpp_{ex.get('task_id', ex.get('id', len(examples)))}",
                    "prompt": ex["text"],
                    "canonical_solution": ex["code"],
                    "test_list": ex["test_list"],
                    "test_setup_code": ex.get("test_setup_code", ""),
                    "challenge_test_list": ex.get("challenge_test_list", []),
                })

    print(f"Loaded {len(examples)} MBPP examples")
    return examples


def load_apps_dataset(
    cache_dir: str = "./cache",
    difficulty: str = "all"
) -> List[Dict]:
    """
    Load APPS dataset

    Args:
        cache_dir: Cache directory
        difficulty: Difficulty level ("introductory", "interview", "competition", "all")

    Returns:
        List of APPS examples
    """
    print("Loading APPS dataset...")
    dataset = load_dataset("codeparrot/apps", cache_dir=cache_dir, trust_remote_code=True)

    examples = []

    # Process train and test splits
    for split_name in ["train", "test"]:
        if split_name not in dataset:
            continue

        for idx, ex in enumerate(tqdm(dataset[split_name], desc=f"Loading APPS {split_name}")):
            # Filter by difficulty if specified
            if difficulty != "all" and ex.get("difficulty") != difficulty:
                continue

            # Parse test cases
            try:
                input_output = json.loads(ex.get("input_output", "{}"))
                inputs = input_output.get("inputs", [])
                outputs = input_output.get("outputs", [])

                # Create test assertions
                test_list = []
                for inp, out in zip(inputs, outputs):
                    # APPS uses stdin/stdout, convert to assertions
                    test_list.append({
                        "input": inp,
                        "output": out
                    })

            except:
                test_list = []

            # Get solutions (APPS may have multiple)
            solutions = ex.get("solutions", "")
            if solutions:
                try:
                    solutions = json.loads(solutions)
                    canonical_solution = solutions[0] if solutions else ""
                except:
                    canonical_solution = solutions
            else:
                canonical_solution = ""

            examples.append({
                "dataset": "apps",
                "task_id": f"apps_{split_name}_{idx}",
                "prompt": ex["question"],
                "canonical_solution": canonical_solution,
                "test_list": test_list,
                "test_setup_code": "",
                "difficulty": ex.get("difficulty", "unknown"),
                "starter_code": ex.get("starter_code", ""),
            })

    print(f"Loaded {len(examples)} APPS examples")
    return examples


def combine_and_split_datasets(
    mbpp_examples: List[Dict],
    apps_examples: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Combine MBPP and APPS, then split into train/val

    Args:
        mbpp_examples: MBPP examples
        apps_examples: APPS examples
        train_ratio: Ratio for training split
        seed: Random seed

    Returns:
        Tuple of (train_data, val_data)
    """
    random.seed(seed)

    # Split MBPP (80% train, 20% val)
    random.shuffle(mbpp_examples)
    mbpp_split = int(len(mbpp_examples) * train_ratio)
    mbpp_train = mbpp_examples[:mbpp_split]
    mbpp_val = mbpp_examples[mbpp_split:]

    print(f"MBPP split: {len(mbpp_train)} train, {len(mbpp_val)} val")

    # Split APPS (80% train, 20% val)
    random.shuffle(apps_examples)
    apps_split = int(len(apps_examples) * train_ratio)
    apps_train = apps_examples[:apps_split]
    apps_val = apps_examples[apps_split:]

    print(f"APPS split: {len(apps_train)} train, {len(apps_val)} val")

    # Combine
    train_data = mbpp_train + apps_train
    val_data = mbpp_val + apps_val

    # Shuffle combined data
    random.shuffle(train_data)
    random.shuffle(val_data)

    print(f"\nCombined: {len(train_data)} train, {len(val_data)} val")
    print(f"  Train: {sum(1 for x in train_data if x['dataset']=='mbpp')} MBPP + {sum(1 for x in train_data if x['dataset']=='apps')} APPS")
    print(f"  Val:   {sum(1 for x in val_data if x['dataset']=='mbpp')} MBPP + {sum(1 for x in val_data if x['dataset']=='apps')} APPS")

    return train_data, val_data


def save_dataset(data: List[Dict], filepath: Path):
    """Save dataset to JSONL file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        for example in tqdm(data, desc=f"Saving {filepath.name}"):
            f.write(json.dumps(example) + '\n')

    print(f"Saved {len(data)} examples to {filepath}")


def load_dataset_from_file(filepath: str) -> List[Dict]:
    """Load dataset from JSONL file"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


if __name__ == "__main__":
    # Test loading
    mbpp = load_mbpp_dataset()
    apps = load_apps_dataset(difficulty="introductory")  # Start with easier APPS

    train, val = combine_and_split_datasets(mbpp, apps)

    print(f"\nTotal: {len(train)} train + {len(val)} val = {len(train) + len(val)}")
