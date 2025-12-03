#!/usr/bin/env python
"""
完整测试脚本 - 验证所有假设和提取过程

测试内容:
1. Student模型输出格式 (thinking + code)
2. Teacher模型(vLLM)输出提取
3. Thinking块提取的正确性
4. 代码提取的正确性
5. SFT标签构建的正确性
6. 代码执行流程
7. 所有提取方法的一致性

All-in-One Comprehensive Test Script

Tests:
1. Student model output format (thinking + code)
2. Teacher model (vLLM) output extraction
3. Thinking block extraction correctness
4. Code extraction correctness
5. SFT label construction correctness
6. Code execution pipeline
7. Consistency across all extraction methods
"""

import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json

# Import all our utilities
from utils.prompts import get_unified_prompt, SYSTEM_PROMPT_UNIFIED
from utils.thinking_extraction import extract_thinking_and_code_unified
from utils.code_parser import validate_python_code
from utils.sft_data import prepare_teacher_sft_batch, verify_labels
from evaluation.code_executor import (
    extract_function_from_code,
    execute_code_with_tests_multiprocess,
    compute_reward
)


class Colors:
    """终端颜色 / Terminal colors"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """打印标题 / Print header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_pass(text):
    """打印成功 / Print pass"""
    print(f"{Colors.GREEN}✓ PASS{Colors.END} {text}")


def print_fail(text):
    """打印失败 / Print fail"""
    print(f"{Colors.RED}✗ FAIL{Colors.END} {text}")


def print_warn(text):
    """打印警告 / Print warning"""
    print(f"{Colors.YELLOW}⚠ WARN{Colors.END} {text}")


class TestResults:
    """测试结果记录 / Test results tracker"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def add_pass(self, name):
        self.total += 1
        self.passed += 1
        print_pass(name)

    def add_fail(self, name):
        self.total += 1
        self.failed += 1
        print_fail(name)

    def add_warn(self, name):
        self.warnings += 1
        print_warn(name)

    def summary(self):
        print_header("测试总结 / Test Summary")
        print(f"总测试数 / Total:   {self.total}")
        print(f"通过 / Passed:       {Colors.GREEN}{self.passed}{Colors.END}")
        print(f"失败 / Failed:       {Colors.RED}{self.failed}{Colors.END}")
        print(f"警告 / Warnings:     {Colors.YELLOW}{self.warnings}{Colors.END}")

        if self.failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}所有测试通过! / All Tests Passed!{Colors.END}\n")
            return True
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}有测试失败! / Some Tests Failed!{Colors.END}\n")
            return False


results = TestResults()


def test_1_student_model_output():
    """测试1: Student模型输出格式 / Test 1: Student model output format"""
    print_header("测试1: Student模型输出 / Test 1: Student Model Output")

    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    try:
        print(f"加载模型 / Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        results.add_pass("模型加载成功 / Model loaded")
    except Exception as e:
        results.add_fail(f"模型加载失败 / Model loading failed: {e}")
        return

    # 测试统一提示词 / Test unified prompt
    test_problem = "Write a function to check if a number is prime."
    prompt = get_unified_prompt(test_problem)

    print(f"\n使用的提示词 / Using prompt:")
    print(f"{Colors.YELLOW}{prompt[:200]}...{Colors.END}\n")

    # 应用chat模板 / Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 生成 / Generate
    print("生成中... / Generating...")
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    print(f"\n生成的输出 / Generated output:")
    print(f"{Colors.YELLOW}{'-'*70}")
    print(generated[:500] + "..." if len(generated) > 500 else generated)
    print(f"{'-'*70}{Colors.END}\n")

    # 检查thinking块 / Check for thinking blocks
    has_think = "<think>" in generated.lower()
    if has_think:
        results.add_pass("输出包含 <think> 块 / Output contains <think> blocks")
    else:
        results.add_fail("输出不包含 <think> 块 / Output missing <think> blocks")
        results.add_warn("提示: 模型可能需要显式要求thinking / May need explicit thinking request")

    # 检查markdown / Check for markdown
    has_markdown = "```python" in generated or "```" in generated
    if has_markdown:
        results.add_pass("输出包含markdown代码块 / Output contains markdown blocks")
    else:
        results.add_fail("输出不包含markdown / Output missing markdown")

    # 提取thinking和代码 / Extract thinking and code
    parsed = extract_thinking_and_code_unified(generated)

    if parsed["thinking"]:
        results.add_pass(f"成功提取thinking / Thinking extracted ({len(parsed['thinking'])} chars)")
        print(f"  预览 / Preview: {parsed['thinking'][:100]}...\n")
    else:
        if has_think:
            results.add_fail("有<think>但提取失败 / <think> present but extraction failed")
        else:
            results.add_warn("无thinking内容 / No thinking content")

    if parsed["code"]:
        results.add_pass(f"成功提取代码 / Code extracted ({len(parsed['code'])} chars)")
        print(f"  预览 / Preview:\n{parsed['code'][:150]}...\n")

        # 验证Python语法 / Validate Python syntax
        is_valid, error = validate_python_code(parsed["code"])
        if is_valid:
            results.add_pass("代码语法有效 / Code syntax valid")
        else:
            results.add_fail(f"代码语法错误 / Code syntax error: {error}")
    else:
        results.add_fail("代码提取失败 / Code extraction failed")

    return generated, parsed


def test_2_teacher_vllm_extraction():
    """测试2: Teacher模型(vLLM)输出提取 / Test 2: Teacher (vLLM) extraction"""
    print_header("测试2: Teacher模型输出提取 / Test 2: Teacher Model Extraction")

    # 模拟vLLM响应 / Simulate vLLM response
    simulated_vllm_response = """<think>
To check if a number is prime, I need to:
1. Handle edge cases (n <= 1 not prime, n == 2 is prime)
2. Check if even (not prime except 2)
3. Check odd divisors from 3 to sqrt(n)
</think>

```python
def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False

    return True
```"""

    print("模拟的vLLM响应 / Simulated vLLM response:")
    print(f"{Colors.YELLOW}{simulated_vllm_response}{Colors.END}\n")

    # 测试提取 / Test extraction
    parsed = extract_thinking_and_code_unified(simulated_vllm_response)

    if parsed["thinking"] and "edge cases" in parsed["thinking"]:
        results.add_pass("vLLM thinking提取正确 / vLLM thinking extraction correct")
        print(f"  提取的thinking / Extracted thinking:\n  {parsed['thinking'][:100]}...\n")
    else:
        results.add_fail("vLLM thinking提取失败 / vLLM thinking extraction failed")

    if parsed["code"] and "def is_prime" in parsed["code"]:
        results.add_pass("vLLM代码提取正确 / vLLM code extraction correct")
        print(f"  提取的代码 / Extracted code:\n  {parsed['code'][:100]}...\n")
    else:
        results.add_fail("vLLM代码提取失败 / vLLM code extraction failed")

    # 测试不同格式 / Test different formats
    test_cases = [
        {
            "name": "大小写不敏感 / Case insensitive",
            "text": "<THINK>reasoning</THINK>\n\n```python\ncode```",
            "should_have_thinking": True
        },
        {
            "name": "无markdown / No markdown",
            "text": "<think>reasoning</think>\n\ndef func(): pass",
            "should_have_thinking": True
        },
        {
            "name": "只有代码 / Code only",
            "text": "```python\ndef func(): pass\n```",
            "should_have_thinking": False
        }
    ]

    for case in test_cases:
        parsed = extract_thinking_and_code_unified(case["text"])
        has_thinking = bool(parsed["thinking"])
        has_code = bool(parsed["code"])

        if has_thinking == case["should_have_thinking"] and has_code:
            results.add_pass(f"{case['name']}: 提取正确 / extraction correct")
        else:
            results.add_fail(f"{case['name']}: 提取失败 / extraction failed")


def test_3_code_extraction_consistency():
    """测试3: 代码提取一致性 / Test 3: Code extraction consistency"""
    print_header("测试3: 代码提取一致性 / Test 3: Code Extraction Consistency")

    # 测试用例: 完整响应 / Test case: full response
    full_response = """<think>
I need to implement binary search.
</think>

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```"""

    # 方法1: unified extraction / Method 1: unified extraction
    parsed = extract_thinking_and_code_unified(full_response)
    code1 = parsed["code"]

    # 方法2: code executor extraction / Method 2: code executor extraction
    code2 = extract_function_from_code(full_response)

    # 方法3: 直接从markdown提取 / Method 3: direct markdown extraction
    from utils.code_parser import extract_code_from_markdown
    code3 = extract_code_from_markdown(full_response)

    print(f"方法1 (unified): {len(code1) if code1 else 0} chars")
    print(f"方法2 (executor): {len(code2) if code2 else 0} chars")
    print(f"方法3 (markdown): {len(code3) if code3 else 0} chars\n")

    # 检查一致性 / Check consistency
    if code1 and code2 and code3:
        # 去除空白后比较 / Compare after removing whitespace
        c1 = code1.replace(" ", "").replace("\n", "")
        c2 = code2.replace(" ", "").replace("\n", "")
        c3 = code3.replace(" ", "").replace("\n", "")

        if c1 == c2 == c3:
            results.add_pass("所有提取方法一致 / All extraction methods consistent")
        else:
            results.add_fail("提取方法不一致 / Extraction methods inconsistent")
            print(f"  方法1长度: {len(c1)}")
            print(f"  方法2长度: {len(c2)}")
            print(f"  方法3长度: {len(c3)}")
    else:
        results.add_fail("某些提取方法失败 / Some extraction methods failed")

    # 验证提取的代码可执行 / Verify extracted code is executable
    if code1:
        is_valid, error = validate_python_code(code1)
        if is_valid:
            results.add_pass("提取的代码语法正确 / Extracted code syntax valid")
        else:
            results.add_fail(f"提取的代码语法错误 / Syntax error: {error}")


def test_4_sft_label_construction():
    """测试4: SFT标签构建 / Test 4: SFT label construction"""
    print_header("测试4: SFT标签构建 / Test 4: SFT Label Construction")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        results.add_pass("Tokenizer加载成功 / Tokenizer loaded")
    except Exception as e:
        results.add_fail(f"Tokenizer加载失败 / Failed: {e}")
        return

    # 测试样例 / Test example
    teacher_example = {
        "prompt": "Write a function to add two numbers.",
        "thinking": "This is simple addition. Take two parameters and return their sum.",
        "code": "def add(a, b):\n    return a + b"
    }

    # 准备SFT批次 / Prepare SFT batch
    try:
        batch = prepare_teacher_sft_batch(
            teacher_examples=[teacher_example],
            tokenizer=tokenizer,
            include_thinking=True
        )
        results.add_pass("SFT批次准备成功 / SFT batch prepared")
    except Exception as e:
        results.add_fail(f"SFT批次准备失败 / Failed: {e}")
        return

    # 检查标签 / Check labels
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    # 统计 / Statistics
    num_prompt_tokens = (labels == -100).sum().item()
    num_response_tokens = ((labels != -100) & (labels != tokenizer.pad_token_id)).sum().item()
    total_tokens = len(labels)

    print(f"\n标签统计 / Label statistics:")
    print(f"  总token数 / Total tokens: {total_tokens}")
    print(f"  提示token (=-100): {num_prompt_tokens}")
    print(f"  响应token (用于loss): {num_response_tokens}")
    print(f"  响应比例 / Response ratio: {num_response_tokens/total_tokens*100:.1f}%\n")

    # 验证: 应该有提示和响应部分 / Verify: should have both prompt and response
    if num_prompt_tokens > 0 and num_response_tokens > 0:
        results.add_pass("标签包含提示和响应部分 / Labels have prompt and response")
    else:
        results.add_fail("标签结构错误 / Label structure incorrect")

    # 验证: 响应部分应该包含代码 / Verify: response should contain code
    if num_response_tokens > 20:  # 代码应该有合理长度 / Code should be reasonable length
        results.add_pass("响应部分长度合理 / Response length reasonable")
    else:
        results.add_warn("响应部分可能太短 / Response may be too short")

    # 详细验证 / Detailed verification
    print("标签详细验证 / Detailed label verification:")
    verify_labels(batch["input_ids"], batch["labels"], tokenizer, example_idx=0)


def test_5_code_execution():
    """测试5: 代码执行流程 / Test 5: Code execution pipeline"""
    print_header("测试5: 代码执行 / Test 5: Code Execution")

    # 测试用例 / Test cases
    test_code = """def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True"""

    test_cases = [
        "assert is_prime(2) == True",
        "assert is_prime(3) == True",
        "assert is_prime(4) == False",
        "assert is_prime(17) == True",
        "assert is_prime(1) == False"
    ]

    print("测试代码 / Test code:")
    print(f"{Colors.YELLOW}{test_code}{Colors.END}\n")

    # 执行测试 / Execute tests
    try:
        all_passed, num_passed, num_total, messages = execute_code_with_tests_multiprocess(
            code=test_code,
            test_cases=test_cases,
            setup_code="",
            timeout=5
        )

        print(f"测试结果 / Test results: {num_passed}/{num_total} passed\n")

        if all_passed:
            results.add_pass("所有测试用例通过 / All test cases passed")
        else:
            results.add_fail(f"部分测试失败 / Some tests failed: {num_passed}/{num_total}")
            for i, msg in enumerate(messages):
                if "failed" in msg.lower() or "error" in msg.lower():
                    print(f"  测试 {i+1}: {msg}")
    except Exception as e:
        results.add_fail(f"代码执行失败 / Execution failed: {e}")
        return

    # 测试奖励计算 / Test reward computation
    try:
        reward = compute_reward(
            code=test_code,
            test_cases=test_cases,
            reward_type="binary"
        )

        if reward == 1.0:
            results.add_pass(f"奖励计算正确 / Reward correct: {reward}")
        else:
            results.add_fail(f"奖励计算错误 / Reward incorrect: {reward}")
    except Exception as e:
        results.add_fail(f"奖励计算失败 / Reward computation failed: {e}")

    # 测试带thinking和markdown的完整响应 / Test full response with thinking and markdown
    full_response_with_thinking = """<think>
Testing prime number checking.
</think>

```python
def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
```"""

    try:
        extracted_code = extract_function_from_code(full_response_with_thinking)
        if extracted_code and "def is_prime" in extracted_code:
            results.add_pass("从完整响应提取代码成功 / Code extracted from full response")

            # 执行提取的代码 / Execute extracted code
            all_passed, _, _, _ = execute_code_with_tests_multiprocess(
                code=extracted_code,
                test_cases=test_cases
            )

            if all_passed:
                results.add_pass("提取的代码执行成功 / Extracted code executes correctly")
            else:
                results.add_fail("提取的代码执行失败 / Extracted code execution failed")
        else:
            results.add_fail("从完整响应提取代码失败 / Failed to extract from full response")
    except Exception as e:
        results.add_fail(f"完整流程测试失败 / Full pipeline test failed: {e}")


async def test_6_teacher_api_if_available():
    """测试6: Teacher API (如果可用) / Test 6: Teacher API (if available)"""
    print_header("测试6: Teacher API测试 / Test 6: Teacher API Test")

    try:
        from synthesis.teacher_query import TeacherModelClient
        from config import get_default_config

        config = get_default_config()

        print(f"尝试连接: {config.teacher.api_url}")
        print("(如果vLLM未运行,此测试将跳过) / (Skip if vLLM not running)\n")

        client = TeacherModelClient(config.teacher)

        # 简单测试 / Simple test
        messages = [{
            "role": "user",
            "content": get_unified_prompt("Write a function to multiply two numbers.")
        }]

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                response = await asyncio.wait_for(
                    client.query_single(session, messages),
                    timeout=10.0
                )

            results.add_pass("Teacher API连接成功 / Teacher API connected")

            print(f"Teacher响应预览 / Teacher response preview:")
            print(f"{Colors.YELLOW}{response.full_text[:300]}...{Colors.END}\n")

            # 验证提取 / Verify extraction
            if response.thinking:
                results.add_pass("Teacher返回thinking / Teacher returned thinking")
            else:
                results.add_warn("Teacher未返回thinking / Teacher no thinking")

            if response.response:
                results.add_pass("Teacher返回代码 / Teacher returned code")
            else:
                results.add_fail("Teacher未返回代码 / Teacher no code")

        except asyncio.TimeoutError:
            results.add_warn("Teacher API超时(可能未运行) / API timeout (may not be running)")
        except Exception as e:
            results.add_warn(f"Teacher API不可用: {e} / API unavailable: {e}")

    except ImportError as e:
        results.add_warn(f"Teacher API模块未找到: {e} / Module not found: {e}")


def test_7_unified_prompt_format():
    """测试7: 统一提示词格式 / Test 7: Unified prompt format"""
    print_header("测试7: 统一提示词 / Test 7: Unified Prompt")

    from utils.prompts import get_unified_prompt, SYSTEM_PROMPT_UNIFIED

    # 测试提示词生成 / Test prompt generation
    problem = "Write a function to check if a string is a palindrome."
    prompt = get_unified_prompt(problem)

    print(f"生成的提示词 / Generated prompt:")
    print(f"{Colors.YELLOW}{prompt}{Colors.END}\n")

    # 验证关键元素 / Verify key elements
    checks = [
        ("<think>", "包含<think>标签示例 / Contains <think> example"),
        ("```python", "包含markdown示例 / Contains markdown example"),
        (problem, "包含问题描述 / Contains problem"),
    ]

    for keyword, description in checks:
        if keyword in prompt:
            results.add_pass(description)
        else:
            results.add_fail(description)

    # 验证系统提示词 / Verify system prompt
    print(f"系统提示词 / System prompt:")
    print(f"{Colors.YELLOW}{SYSTEM_PROMPT_UNIFIED}{Colors.END}\n")

    if "<think>" in SYSTEM_PROMPT_UNIFIED and "markdown" in SYSTEM_PROMPT_UNIFIED:
        results.add_pass("系统提示词包含关键指令 / System prompt has key instructions")
    else:
        results.add_fail("系统提示词缺少关键指令 / System prompt missing instructions")


def main():
    """主测试流程 / Main test pipeline"""
    print(f"\n{Colors.BOLD}{'='*70}")
    print("完整测试脚本 - 验证所有假设")
    print("Complete Test Script - Verify All Assumptions")
    print(f"{'='*70}{Colors.END}\n")

    # 运行所有测试 / Run all tests
    test_7_unified_prompt_format()

    student_output, parsed = test_1_student_model_output()

    test_2_teacher_vllm_extraction()

    test_3_code_extraction_consistency()

    test_4_sft_label_construction()

    test_5_code_execution()

    # Teacher API测试(可选) / Teacher API test (optional)
    try:
        asyncio.run(test_6_teacher_api_if_available())
    except Exception as e:
        results.add_warn(f"Teacher API测试跳过: {e} / Skipped: {e}")

    # 显示总结 / Show summary
    success = results.summary()

    # 建议 / Recommendations
    if not success:
        print_header("建议 / Recommendations")
        print("1. 检查失败的测试 / Check failed tests")
        print("2. 确保模型正确加载 / Ensure models load correctly")
        print("3. 验证提示词格式 / Verify prompt format")
        print("4. 检查代码提取逻辑 / Check extraction logic")
    else:
        print_header("下一步 / Next Steps")
        print("✓ 所有测试通过,可以开始训练! / All tests passed, ready to train!")
        print("✓ 运行: python phase1_prepare_data.py")
        print("✓ 运行: python phase2_generate_teacher_traces.py")
        print("✓ 运行: python phase3_train_grpo.py")


if __name__ == "__main__":
    main()
