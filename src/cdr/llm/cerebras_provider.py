"""
Cerebras LLM Provider

Integration with Cerebras Inference API via OpenAI-compatible endpoint.
Cerebras offers FREE tier with generous limits:
- All models: 60K TPM, 1M TPD, 30 RPM, 900 RPH, 14.4K RPD
- llama-3.3-70b: Best for reasoning
- qwen-3-32b: Good balance
- llama3.1-8b: Fast

Base URL: https://api.cerebras.ai/v1
Uses standard OpenAI SDK with modified base_url.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Any, AsyncIterator, Iterator

from cdr.core.exceptions import LLMError, LLMProviderError, LLMRateLimitError
from cdr.llm.base import BaseLLMProvider, LLMResponse, Message, StreamChunk

try:
    import openai
    from openai import AsyncOpenAI, OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Retry configuration for rate limits
MAX_RETRIES = 10
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 60.0  # seconds


# Cerebras FREE tier models
CEREBRAS_FREE_MODELS = {
    "default": "llama-3.3-70b",  # Best quality
    "fast": "llama3.1-8b",  # Fastest
    "reasoning": "llama-3.3-70b",  # Best reasoning
    "qwen": "qwen-3-32b",  # Good balance
    "large": "gpt-oss-120b",  # Largest (preview)
}

# Cerebras Free tier limits (per model)
CEREBRAS_FREE_LIMITS = {
    "tokens_per_minute": 60_000,
    "tokens_per_day": 1_000_000,
    "requests_per_minute": 30,
    "requests_per_hour": 900,
    "requests_per_day": 14_400,
}


class CerebrasProvider(BaseLLMProvider):
    """Cerebras LLM provider - FREE tier via OpenAI-compatible API.

    Cerebras provides extremely fast inference on their custom LPU chips.
    OpenAI-compatible API endpoint for easy integration.

    API Key: Get from https://cloud.cerebras.ai/
    Free tier: 60K tokens/min, 1M tokens/day
    """

    CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

    def __init__(
        self,
        model: str = "llama-3.3-70b",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize Cerebras provider.

        Args:
            model: Model name (llama-3.3-70b, qwen-3-32b, llama3.1-8b).
            api_key: Cerebras API key. Falls back to CEREBRAS_API_KEY env var.
            timeout: Request timeout.
            max_retries: Maximum retries on transient errors.
        """
        if not OPENAI_AVAILABLE:
            raise LLMProviderError(
                provider="cerebras",
                message="OpenAI package not installed. Run: pip install openai",
            )

        # Get API key from env if not provided
        api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise LLMProviderError(
                provider="cerebras",
                message="Cerebras API key required. Set CEREBRAS_API_KEY env var.",
            )

        super().__init__(model, api_key, self.CEREBRAS_BASE_URL, timeout, max_retries)

        # Initialize OpenAI-compatible clients with Cerebras base URL
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.CEREBRAS_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.CEREBRAS_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        return "cerebras"

    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous completion with automatic retry for rate limits."""
        normalized = self._normalize_messages(messages)

        # Remove unsupported kwargs for Cerebras
        kwargs.pop("response_format", None)
        kwargs.pop("model", None)
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        # Cerebras doesn't support these
        kwargs.pop("frequency_penalty", None)
        kwargs.pop("presence_penalty", None)
        kwargs.pop("logit_bias", None)

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[m.to_dict() for m in normalized],
                    temperature=temperature,
                    max_completion_tokens=max_tokens or 4096,
                    **kwargs,
                )

                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    provider="cerebras",
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens
                        if response.usage
                        else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    },
                )
            except openai.RateLimitError as e:
                last_error = e
                error_msg = str(e)

                # Extract wait time if available
                import re

                wait_match = re.search(r"try again in (\d+)", error_msg)
                if wait_match:
                    wait_seconds = int(wait_match.group(1))
                    print(f"[Cerebras] Rate limit hit. Waiting {wait_seconds + 5} seconds...")
                    time.sleep(wait_seconds + 5)
                else:
                    delay = min(BASE_DELAY * (2**attempt) + random.uniform(0, 1), MAX_DELAY)
                    print(
                        f"[Cerebras] Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)
            except openai.APIError as e:
                raise LLMError(f"Cerebras API error: {e}") from e

        raise LLMRateLimitError(provider="cerebras") from last_error

    async def acomplete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async completion with automatic retry for rate limits."""
        normalized = self._normalize_messages(messages)

        # Remove unsupported kwargs
        kwargs.pop("response_format", None)
        kwargs.pop("model", None)
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        kwargs.pop("frequency_penalty", None)
        kwargs.pop("presence_penalty", None)
        kwargs.pop("logit_bias", None)

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await self._async_client.chat.completions.create(
                    model=self._model,
                    messages=[m.to_dict() for m in normalized],
                    temperature=temperature,
                    max_completion_tokens=max_tokens or 4096,
                    **kwargs,
                )

                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    provider="cerebras",
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens
                        if response.usage
                        else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    },
                )
            except openai.RateLimitError as e:
                last_error = e
                error_msg = str(e)

                import re

                wait_match = re.search(r"try again in (\d+)", error_msg)
                if wait_match:
                    wait_seconds = int(wait_match.group(1))
                    print(f"[Cerebras] Rate limit hit. Waiting {wait_seconds + 5} seconds...")
                    await asyncio.sleep(wait_seconds + 5)
                else:
                    delay = min(BASE_DELAY * (2**attempt) + random.uniform(0, 1), MAX_DELAY)
                    print(
                        f"[Cerebras] Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(delay)
            except openai.APIError as e:
                raise LLMError(f"Cerebras API error: {e}") from e

        raise LLMRateLimitError(provider="cerebras") from last_error

    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Streaming completion."""
        normalized = self._normalize_messages(messages)
        kwargs.pop("response_format", None)
        kwargs.pop("frequency_penalty", None)
        kwargs.pop("presence_penalty", None)

        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in normalized],
                temperature=temperature,
                max_completion_tokens=max_tokens or 4096,
                stream=True,
                **kwargs,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        content=chunk.choices[0].delta.content,
                        finish_reason=chunk.choices[0].finish_reason,
                    )
        except openai.RateLimitError as e:
            raise LLMRateLimitError(provider="cerebras") from e
        except openai.APIError as e:
            raise LLMError(f"Cerebras API error: {e}") from e

    async def astream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Async streaming completion."""
        normalized = self._normalize_messages(messages)
        kwargs.pop("response_format", None)
        kwargs.pop("frequency_penalty", None)
        kwargs.pop("presence_penalty", None)

        try:
            stream = await self._async_client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in normalized],
                temperature=temperature,
                max_completion_tokens=max_tokens or 4096,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        content=chunk.choices[0].delta.content,
                        finish_reason=chunk.choices[0].finish_reason,
                    )
        except openai.RateLimitError as e:
            raise LLMRateLimitError(provider="cerebras") from e
        except openai.APIError as e:
            raise LLMError(f"Cerebras API error: {e}") from e
