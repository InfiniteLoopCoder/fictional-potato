"""
Test individual components of the GRPO pipeline
"""
import argparse
import asyncio
import json
from pathlib import Path

from config import get_default_config
from synthesis.teacher_query import TeacherModelClient, create_code_generation_prompt
from evaluation.code_executor import execute_code_with_tests_multiprocess, compute_reward


def test_teacher_query(config):
    """Test teacher model query functionality"""
    print("\n" + "=" * 70)
    print("Testing Teacher Model Query")
    print("=" * 70)

    async def run_test():
        client = TeacherModelClient(config.teacher)

        # Test single query
        test_prompt = create_code_generation_prompt(
            "Write a function to check if a number is prime."
        )

        print(f"\nTest prompt:\n{test_prompt}\n")
        print("Querying teacher model...")

        responses = await client.query_batch(
            prompts=[test_prompt],
            num_samples_per_prompt=2,
        )

        for i, response in enumerate(responses[0]):
            print(f"\n--- Response {i+1} ---")
            if response.thinking:
                print(f"Thinking: {response.thinking[:200]}...")
            print(f"Code:\n{response.response[:300]}...")
            print(f"Metadata: {response.metadata}")

        print("\n✓ Teacher query test passed!")

    asyncio.run(run_test())


def test_code_execution(config):
    """Test code execution sandbox"""
    print("\n" + "=" * 70)
    print("Testing Code Execution Sandbox")
    print("=" * 70)

    # Test case 1: Correct code
    test_code_correct = """
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
"""

    test_cases = [
        "assert is_prime(2) == True",
        "assert is_prime(3) == True",
        "assert is_prime(4) == False",
        "assert is_prime(17) == True",
        "assert is_prime(1) == False",
    ]

    print("\nTest 1: Correct implementation")
    all_passed, num_passed, num_total, messages = execute_code_with_tests_multiprocess(
        test_code_correct, test_cases, timeout=5
    )

    print(f"Results: {num_passed}/{num_total} tests passed")
    print(f"All passed: {all_passed}")

    reward = compute_reward(test_code_correct, test_cases, reward_type="binary")
    print(f"Binary reward: {reward}")

    # Test case 2: Incorrect code
    test_code_incorrect = """
def is_prime(n):
    return n % 2 != 0  # Wrong implementation
"""

    print("\nTest 2: Incorrect implementation")
    all_passed, num_passed, num_total, messages = execute_code_with_tests_multiprocess(
        test_code_incorrect, test_cases, timeout=5
    )

    print(f"Results: {num_passed}/{num_total} tests passed")
    print(f"All passed: {all_passed}")
    print(f"Failed tests: {[msg for msg in messages if 'failed' in msg.lower()]}")

    reward = compute_reward(test_code_incorrect, test_cases, reward_type="partial")
    print(f"Partial reward: {reward}")

    print("\n✓ Code execution test passed!")


def test_data_loading(config):
    """Test data loading and processing"""
    print("\n" + "=" * 70)
    print("Testing Data Loading")
    print("=" * 70)

    # Check if data files exist
    data_dir = Path("./data")
    train_file = data_dir / "mbpp_train.jsonl"
    val_file = data_dir / "mbpp_validation.jsonl"

    if not train_file.exists():
        print("⚠ Training data not found. Run: python main.py --stage data")
        return

    # Load and display sample
    print(f"\nLoading from {train_file}...")
    with open(train_file, 'r') as f:
        samples = [json.loads(line) for line in f]

    print(f"Loaded {len(samples)} training samples")

    if samples:
        sample = samples[0]
        print(f"\nSample task:")
        print(f"  Task ID: {sample['task_id']}")
        print(f"  Prompt: {sample['prompt'][:100]}...")
        print(f"  Test cases: {len(sample['test_list'])}")
        print(f"  First test: {sample['test_list'][0]}")

    # Check synthetic data
    synthetic_file = Path(config.data.synthetic_data_path)
    if synthetic_file.exists():
        with open(synthetic_file, 'r') as f:
            synthetic_samples = [json.loads(line) for line in f]

        print(f"\nLoaded {len(synthetic_samples)} synthetic samples")

        if synthetic_samples:
            sample = synthetic_samples[0]
            print(f"\nSynthetic sample:")
            print(f"  Task ID: {sample['task_id']}")
            print(f"  Has thinking: {bool(sample.get('thinking'))}")
            print(f"  Teacher response length: {len(sample['teacher_response'])}")

    else:
        print("\n⚠ Synthetic data not found. Run: python main.py --stage synthesis")

    print("\n✓ Data loading test passed!")


def test_model_loading(config):
    """Test model and tokenizer loading"""
    print("\n" + "=" * 70)
    print("Testing Model Loading")
    print("=" * 70)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print(f"\nLoading tokenizer: {config.student.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            config.student.model_name,
            trust_remote_code=config.student.trust_remote_code
        )

        print(f"✓ Tokenizer loaded")
        print(f"  Vocab size: {len(tokenizer)}")
        print(f"  Pad token: {tokenizer.pad_token}")

        print(f"\nLoading model: {config.student.model_name}")
        print(f"  Device: {config.student.device_map}")
        print(f"  Dtype: {config.student.torch_dtype}")

        model = AutoModelForCausalLM.from_pretrained(
            config.student.model_name,
            torch_dtype=config.student.torch_dtype,
            device_map=config.student.device_map,
            trust_remote_code=config.student.trust_remote_code,
        )

        print(f"✓ Model loaded")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

        # Test generation
        print("\nTesting generation...")
        test_input = "def hello_world():"
        inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Input: {test_input}")
        print(f"  Output: {generated[:100]}...")

        print("\n✓ Model loading test passed!")

    except Exception as e:
        print(f"\n✗ Model loading failed: {e}")
        raise


def test_lora_setup(config):
    """Test LoRA adapter setup"""
    print("\n" + "=" * 70)
    print("Testing LoRA Setup")
    print("=" * 70)

    try:
        from transformers import AutoModelForCausalLM
        from peft import get_peft_model, LoraConfig, TaskType

        print(f"\nLoading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            config.student.model_name,
            torch_dtype=config.student.torch_dtype,
            device_map=config.student.device_map,
            trust_remote_code=config.student.trust_remote_code,
        )

        print(f"✓ Base model loaded")

        # Apply LoRA
        print(f"\nApplying LoRA adapters...")
        print(f"  Rank: {config.lora.r}")
        print(f"  Alpha: {config.lora.lora_alpha}")
        print(f"  Target modules: {config.lora.target_modules}")

        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)

        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())

        print(f"\n✓ LoRA applied")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Total parameters: {all_params:,}")
        print(f"  Trainable %: {100 * trainable_params / all_params:.2f}%")

        print("\n✓ LoRA setup test passed!")

    except Exception as e:
        print(f"\n✗ LoRA setup failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Test GRPO pipeline components")
    parser.add_argument(
        "--component",
        type=str,
        choices=["teacher", "executor", "data", "model", "lora", "all"],
        default="all",
        help="Component to test"
    )

    args = parser.parse_args()
    config = get_default_config()

    print("\n" + "=" * 70)
    print("GRPO Pipeline Component Tests")
    print("=" * 70)
    print(f"Testing: {args.component}")

    if args.component in ["teacher", "all"]:
        test_teacher_query(config)

    if args.component in ["executor", "all"]:
        test_code_execution(config)

    if args.component in ["data", "all"]:
        test_data_loading(config)

    if args.component in ["model", "all"]:
        test_model_loading(config)

    if args.component in ["lora", "all"]:
        test_lora_setup(config)

    print("\n" + "=" * 70)
    print("All Tests Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
