"""
Pass@k metric calculation for code generation evaluation
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from code_executor import execute_code_with_tests_multiprocess


def estimate_pass_at_k(
    num_samples: int,
    num_correct: int,
    k: int
) -> float:
    """
    Estimate pass@k using the formula from the Codex paper

    Args:
        num_samples: Total number of samples (n)
        num_correct: Number of correct samples (c)
        k: k value for pass@k

    Returns:
        pass@k probability
    """
    if num_samples - num_correct < k:
        return 1.0

    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))


def calculate_pass_at_k(
    task_results: Dict[str, List[bool]],
    k_values: List[int] = [1, 5, 10, 25, 50, 100]
) -> Dict[int, float]:
    """
    Calculate pass@k metrics for all tasks

    Args:
        task_results: Dict mapping task_id to list of test results (bool)
        k_values: List of k values to compute

    Returns:
        Dict mapping k to pass@k score
    """
    pass_at_k = {k: [] for k in k_values}

    for task_id, results in task_results.items():
        num_samples = len(results)
        num_correct = sum(results)

        for k in k_values:
            if k > num_samples:
                continue

            pass_k = estimate_pass_at_k(num_samples, num_correct, k)
            pass_at_k[k].append(pass_k)

    # Average across all tasks
    return {k: np.mean(scores) if scores else 0.0 for k, scores in pass_at_k.items()}


def generate_samples(
    model,
    tokenizer,
    prompts: List[str],
    num_samples_per_prompt: int,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
    batch_size: int = 8,
) -> List[List[str]]:
    """
    Generate code samples from model

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of problem prompts
        num_samples_per_prompt: Number of samples per prompt
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for generation

    Returns:
        List of lists of generated code samples
    """
    device = next(model.parameters()).device
    all_samples = []

    for prompt in tqdm(prompts, desc="Generating samples"):
        samples = []

        # Format prompt for chat model
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate multiple samples
        for i in range(0, num_samples_per_prompt, batch_size):
            batch_size_actual = min(batch_size, num_samples_per_prompt - i)

            inputs = tokenizer(
                [formatted_prompt] * batch_size_actual,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode outputs
            for output in outputs:
                # Remove input prompt from output
                generated = output[inputs["input_ids"].shape[1]:]
                code = tokenizer.decode(generated, skip_special_tokens=True)
                samples.append(code)

        all_samples.append(samples)

    return all_samples


def evaluate_pass_at_k(
    model_path: str,
    test_data_path: str,
    output_path: str,
    config,
    num_samples_per_task: int = 100,
    k_values: List[int] = [1, 5, 10, 25, 50, 100],
    device: str = "cuda",
) -> Dict:
    """
    Evaluate model using pass@k metric

    Args:
        model_path: Path to trained model or checkpoint
        test_data_path: Path to test data JSONL
        output_path: Path to save results
        config: Configuration object
        num_samples_per_task: Number of samples to generate per task
        k_values: List of k values for pass@k
        device: Device to use

    Returns:
        Dictionary of results
    """
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    # Load test data
    print(f"Loading test data from {test_data_path}...")
    test_tasks = []
    with open(test_data_path, 'r') as f:
        for line in f:
            test_tasks.append(json.loads(line))

    print(f"Evaluating on {len(test_tasks)} tasks...")

    # Prepare prompts
    prompts = [f"Write a Python function to solve the following problem:\n\n{task['prompt']}"
               for task in test_tasks]

    # Generate samples
    print(f"Generating {num_samples_per_task} samples per task...")
    all_samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        num_samples_per_prompt=num_samples_per_task,
        temperature=config.evaluation.temperature,
        top_p=config.evaluation.top_p,
        max_new_tokens=1024,
        batch_size=4,
    )

    # Evaluate samples
    print("Evaluating generated samples...")
    task_results = {}

    for task, samples in tqdm(zip(test_tasks, all_samples), total=len(test_tasks)):
        task_id = task["task_id"]
        test_cases = task["test_list"]
        setup_code = task.get("test_setup_code", "")

        results = []
        for sample in samples:
            all_passed, _, _, _ = execute_code_with_tests_multiprocess(
                code=sample,
                test_cases=test_cases,
                setup_code=setup_code,
                timeout=config.evaluation.timeout,
            )
            results.append(all_passed)

        task_results[task_id] = results

    # Calculate pass@k
    print("Calculating pass@k metrics...")
    pass_at_k_scores = calculate_pass_at_k(task_results, k_values)

    # Print results
    print("\n" + "=" * 50)
    print("Pass@k Results:")
    print("=" * 50)
    for k, score in sorted(pass_at_k_scores.items()):
        print(f"pass@{k:3d}: {score:.4f} ({score * 100:.2f}%)")
    print("=" * 50)

    # Save detailed results
    results = {
        "model_path": model_path,
        "test_data_path": test_data_path,
        "num_tasks": len(test_tasks),
        "num_samples_per_task": num_samples_per_task,
        "pass_at_k": {k: float(v) for k, v in pass_at_k_scores.items()},
        "task_results": {
            task_id: {
                "num_correct": sum(results),
                "num_total": len(results),
                "pass_rate": sum(results) / len(results),
            }
            for task_id, results in task_results.items()
        }
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    from config import get_default_config

    config = get_default_config()

    # Evaluate trained model
    evaluate_pass_at_k(
        model_path=config.training.output_dir + "/final_model",
        test_data_path="./data/mbpp_test.jsonl",
        output_path="./outputs/evaluation_results.json",
        config=config,
        num_samples_per_task=config.evaluation.num_samples_per_task,
        k_values=config.evaluation.k_values,
    )
