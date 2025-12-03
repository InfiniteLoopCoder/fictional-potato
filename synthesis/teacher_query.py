"""
High-concurrency teacher model query script using vLLM API
"""
import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from tqdm.asyncio import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class TeacherResponse:
    """Teacher model response"""
    prompt: str
    response: str
    thinking: Optional[str] = None
    full_text: str = ""
    metadata: Dict = None


class TeacherModelClient:
    """High-concurrency client for querying teacher model via vLLM API"""

    def __init__(self, config):
        """
        Initialize teacher model client

        Args:
            config: TeacherConfig object
        """
        self.api_url = config.api_url
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.presence_penalty = config.presence_penalty
        self.max_tokens = config.max_tokens
        self.enable_thinking = config.enable_thinking
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.max_concurrent = config.max_concurrent_requests

        # Semaphore for controlling concurrency
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    def create_payload(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Create API payload with strict structure for vLLM

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters to override defaults

        Returns:
            Complete API payload dictionary
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }

        # CRITICAL: Add chat_template_kwargs for thinking mode
        if self.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}

        return payload

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Make single API request with retry logic

        Args:
            session: aiohttp session
            payload: API payload

        Returns:
            API response dictionary
        """
        async with session.post(
            self.api_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def query_single(
        self,
        session: aiohttp.ClientSession,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> TeacherResponse:
        """
        Query teacher model for single prompt

        Args:
            session: aiohttp session
            messages: Conversation messages
            **kwargs: Additional parameters

        Returns:
            TeacherResponse object
        """
        async with self.semaphore:
            payload = self.create_payload(messages, **kwargs)

            try:
                response_data = await self._make_request(session, payload)

                # Extract response
                choice = response_data["choices"][0]
                message = choice["message"]
                full_text = message["content"]

                # Parse thinking blocks if enabled
                thinking = None
                response_text = full_text

                if self.enable_thinking and "<think>" in full_text:
                    # Extract thinking content
                    parts = full_text.split("<think>")
                    if len(parts) > 1:
                        think_parts = parts[1].split("</think>")
                        if len(think_parts) > 1:
                            thinking = think_parts[0].strip()
                            response_text = think_parts[1].strip()

                return TeacherResponse(
                    prompt=messages[-1]["content"] if messages else "",
                    response=response_text,
                    thinking=thinking,
                    full_text=full_text,
                    metadata={
                        "model": response_data.get("model"),
                        "usage": response_data.get("usage"),
                        "finish_reason": choice.get("finish_reason"),
                    }
                )

            except Exception as e:
                print(f"Error querying teacher model: {e}")
                raise

    async def query_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        num_samples_per_prompt: int = 1,
        **kwargs
    ) -> List[List[TeacherResponse]]:
        """
        Query teacher model for batch of prompts with multiple samples each

        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt
            num_samples_per_prompt: Number of samples to generate per prompt
            **kwargs: Additional parameters

        Returns:
            List of lists of TeacherResponse objects
        """
        async with aiohttp.ClientSession() as session:
            tasks = []

            for prompt in prompts:
                for _ in range(num_samples_per_prompt):
                    messages = []

                    if system_prompt:
                        messages.append({
                            "role": "system",
                            "content": system_prompt
                        })

                    messages.append({
                        "role": "user",
                        "content": prompt
                    })

                    tasks.append(self.query_single(session, messages, **kwargs))

            # Execute all requests concurrently with progress bar
            responses = await tqdm.gather(
                *tasks,
                desc="Querying teacher model",
                total=len(tasks),
            )

            # Group responses by prompt
            grouped_responses = []
            idx = 0
            for _ in prompts:
                group = responses[idx:idx + num_samples_per_prompt]
                grouped_responses.append(group)
                idx += num_samples_per_prompt

            return grouped_responses


def create_code_generation_prompt(task_prompt: str) -> str:
    """
    Create formatted prompt for code generation task

    Args:
        task_prompt: MBPP task description

    Returns:
        Formatted prompt string
    """
    return f"""Write a Python function to solve the following problem:

{task_prompt}

Please think through the problem step by step, then provide a complete Python function implementation."""


async def test_teacher_connection(config):
    """Test connection to teacher model"""
    print(f"Testing connection to teacher model at {config.api_url}...")

    client = TeacherModelClient(config)

    test_prompt = "Write a function to add two numbers."
    messages = [
        {"role": "user", "content": test_prompt}
    ]

    try:
        async with aiohttp.ClientSession() as session:
            response = await client.query_single(session, messages)

        print("\n✓ Connection successful!")
        print(f"\nResponse preview:")
        if response.thinking:
            print(f"Thinking: {response.thinking}...")
        print(f"Code: {response.response}...")

        return True

    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        return False


if __name__ == "__main__":
    from config import get_default_config

    config = get_default_config()

    # Test connection
    asyncio.run(test_teacher_connection(config.teacher))
