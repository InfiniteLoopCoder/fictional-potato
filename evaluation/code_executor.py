"""
Secure code execution sandbox for reward computation
"""
import sys
import io
import multiprocessing
import signal
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Tuple, Any, Optional
import ast


class TimeoutException(Exception):
    """Exception raised when code execution times out"""
    pass


def timeout_handler(signum, frame):
    """Handler for timeout signal"""
    raise TimeoutException("Code execution timed out")


def extract_function_from_code(code: str) -> Optional[str]:
    """
    Extract the main function from generated code

    Args:
        code: Generated code string

    Returns:
        Cleaned code or None if parsing fails
    """
    try:
        # Remove markdown code blocks if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        # Try to parse the code
        tree = ast.parse(code)

        # Extract all function definitions
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        if not functions:
            return None

        # Return the original code if it's valid Python
        return code.strip()

    except SyntaxError:
        # Try to find function definition manually
        lines = code.split('\n')
        func_lines = []
        in_function = False

        for line in lines:
            if line.strip().startswith('def '):
                in_function = True

            if in_function:
                func_lines.append(line)

        if func_lines:
            return '\n'.join(func_lines)

        return None


def execute_code_with_test(
    code: str,
    test_case: str,
    setup_code: str = "",
    timeout: int = 5,
) -> Tuple[bool, str]:
    """
    Execute code with a single test case in a subprocess

    Args:
        code: Generated code to execute
        test_case: Test assertion (e.g., "assert func(1, 2) == 3")
        setup_code: Setup code to run before test
        timeout: Timeout in seconds

    Returns:
        Tuple of (success: bool, message: str)
    """
    def run_test():
        """Inner function to run test"""
        try:
            # Capture stdout/stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Create execution namespace
                namespace = {}

                # Execute setup code if provided
                if setup_code:
                    exec(setup_code, namespace)

                # Execute the generated code
                exec(code, namespace)

                # Execute the test case
                exec(test_case, namespace)

            return True, "Test passed"

        except AssertionError as e:
            return False, f"Assertion failed: {str(e)}"
        except TimeoutException:
            return False, "Execution timed out"
        except Exception as e:
            return False, f"Runtime error: {type(e).__name__}: {str(e)}"

    # Set timeout alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        result = run_test()
        signal.alarm(0)  # Cancel alarm
        return result
    except TimeoutException:
        signal.alarm(0)
        return False, "Execution timed out"
    except Exception as e:
        signal.alarm(0)
        return False, f"Execution error: {str(e)}"


def execute_code_with_tests_multiprocess(
    code: str,
    test_cases: List[str],
    setup_code: str = "",
    timeout: int = 5,
) -> Tuple[bool, int, int, List[str]]:
    """
    Execute code with multiple test cases using multiprocessing

    Args:
        code: Generated code
        test_cases: List of test assertions
        setup_code: Setup code
        timeout: Timeout per test case in seconds

    Returns:
        Tuple of (all_passed, num_passed, num_total, messages)
    """
    # Clean up code
    cleaned_code = extract_function_from_code(code)

    if cleaned_code is None:
        return False, 0, len(test_cases), ["Failed to parse code"]

    results = []
    messages = []

    for test_case in test_cases:
        try:
            # Use multiprocessing to isolate execution
            with multiprocessing.Pool(processes=1) as pool:
                async_result = pool.apply_async(
                    execute_code_with_test,
                    (cleaned_code, test_case, setup_code, timeout)
                )

                try:
                    success, message = async_result.get(timeout=timeout + 1)
                    results.append(success)
                    messages.append(message)
                except multiprocessing.TimeoutError:
                    results.append(False)
                    messages.append("Test execution timed out")
                    pool.terminate()

        except Exception as e:
            results.append(False)
            messages.append(f"Execution failed: {str(e)}")

    num_passed = sum(results)
    num_total = len(test_cases)
    all_passed = all(results)

    return all_passed, num_passed, num_total, messages


def compute_reward(
    code: str,
    test_cases: List[str],
    setup_code: str = "",
    timeout: int = 5,
    reward_type: str = "binary",
) -> float:
    """
    Compute reward based on test case execution

    Args:
        code: Generated code
        test_cases: List of test assertions
        setup_code: Setup code
        timeout: Timeout per test
        reward_type: Type of reward ("binary", "partial", "scaled")

    Returns:
        Reward value (0.0 to 1.0)
    """
    all_passed, num_passed, num_total, _ = execute_code_with_tests_multiprocess(
        code, test_cases, setup_code, timeout
    )

    if reward_type == "binary":
        # Binary reward: 1.0 if all tests pass, 0.0 otherwise
        return 1.0 if all_passed else 0.0

    elif reward_type == "partial":
        # Partial reward: proportion of tests passed
        if num_total == 0:
            return 0.0
        return num_passed / num_total

    elif reward_type == "scaled":
        # Scaled reward with bonus for passing all tests
        if num_total == 0:
            return 0.0

        base_reward = num_passed / num_total

        if all_passed:
            # Bonus for passing all tests
            return min(1.0, base_reward * 1.2)
        else:
            return base_reward

    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def batch_compute_rewards(
    codes: List[str],
    test_cases_list: List[List[str]],
    setup_codes: List[str] = None,
    timeout: int = 5,
    reward_type: str = "binary",
    max_workers: int = 4,
) -> List[float]:
    """
    Compute rewards for batch of code samples

    Args:
        codes: List of generated code strings
        test_cases_list: List of test case lists
        setup_codes: List of setup code strings
        timeout: Timeout per test
        reward_type: Type of reward
        max_workers: Maximum parallel workers

    Returns:
        List of reward values
    """
    if setup_codes is None:
        setup_codes = [""] * len(codes)

    rewards = []

    for code, test_cases, setup_code in zip(codes, test_cases_list, setup_codes):
        reward = compute_reward(
            code=code,
            test_cases=test_cases,
            setup_code=setup_code,
            timeout=timeout,
            reward_type=reward_type,
        )
        rewards.append(reward)

    return rewards


if __name__ == "__main__":
    # Test the executor
    test_code = """
def add_numbers(a, b):
    return a + b
"""

    test_cases = [
        "assert add_numbers(1, 2) == 3",
        "assert add_numbers(0, 0) == 0",
        "assert add_numbers(-1, 1) == 0",
    ]

    all_passed, num_passed, num_total, messages = execute_code_with_tests_multiprocess(
        test_code, test_cases
    )

    print(f"Tests passed: {num_passed}/{num_total}")
    print(f"All passed: {all_passed}")

    reward = compute_reward(test_code, test_cases, reward_type="binary")
    print(f"Binary reward: {reward}")

    reward = compute_reward(test_code, test_cases, reward_type="partial")
    print(f"Partial reward: {reward}")
