"""
Groq LLM Provider

Integration with Groq API (OpenAI-compatible).
Groq offers FREE tier with generous limits:
- Llama 3.1 8B: 14,400 requests/day, 6,000 tokens/minute
- Llama 3.3 70B: 1,000 requests/day, 12,000 tokens/minute

This is the PRIMARY FREE provider for CDR.
Includes automatic retry with exponential backoff for rate limits.
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
MAX_RETRIES = 10  # Increased for daily limit handling
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 30.0  # seconds


# Groq FREE tier models - NO PAYMENT REQUIRED
GROQ_FREE_MODELS = {
    "default": "llama-3.1-8b-instant",  # Fast, 14k req/day
    "fast": "llama-3.1-8b-instant",
    "large": "llama-3.3-70b-versatile",  # Better quality, 1k req/day
    "reasoning": "llama-3.3-70b-versatile",
    "medical": "llama-3.3-70b-versatile",  # Best for medical reasoning
}

# Model limits for reference
GROQ_LIMITS = {
    "llama-3.1-8b-instant": {"requests_per_day": 14400, "tokens_per_minute": 6000},
    "llama-3.3-70b-versatile": {"requests_per_day": 1000, "tokens_per_minute": 12000},
    "llama-3.1-70b-versatile": {"requests_per_day": 1000, "tokens_per_minute": 6000},
}


class GroqProvider(BaseLLMProvider):
    """Groq LLM provider - FREE tier with fast inference.

    Groq provides OpenAI-compatible API with free tier.
    Uses LPU (Language Processing Unit) for fast inference.

    This is the PRIMARY FREE provider for CDR.
    """

    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize Groq provider.

        Args:
            model: Model name (llama-3.1-8b-instant, llama-3.3-70b-versatile).
            api_key: Groq API key. Falls back to GROQ_API_KEY env var.
            timeout: Request timeout.
            max_retries: Maximum retries on transient errors.
        """
        if not OPENAI_AVAILABLE:
            raise LLMProviderError(
                provider="groq",
                message="OpenAI package not installed. Run: pip install openai",
            )

        # Get API key from env if not provided
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise LLMProviderError(
                provider="groq",
                message="Groq API key required. Set GROQ_API_KEY env var or pass api_key.",
            )

        super().__init__(model, api_key, self.GROQ_BASE_URL, timeout, max_retries)

        # Initialize OpenAI-compatible clients with Groq base URL
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.GROQ_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.GROQ_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        return "groq"

    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous completion with automatic retry for rate limits."""
        normalized = self._normalize_messages(messages)

        # Remove unsupported or conflicting kwargs
        kwargs.pop("response_format", None)
        kwargs.pop("model", None)
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[m.to_dict() for m in normalized],
                    temperature=temperature,
                    max_tokens=max_tokens or 4096,
                    **kwargs,
                )

                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    provider="groq",
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

                # Check if this is a daily limit (TPD) vs per-minute limit (TPM)
                if "tokens per day" in error_msg.lower() or "tpd" in error_msg.lower():
                    # Daily limit - extract wait time from message if available
                    import re

                    wait_match = re.search(r"try again in (\d+)m", error_msg)
                    if wait_match:
                        wait_minutes = int(wait_match.group(1))
                        print(
                            f"[Groq] Daily limit (TPD) hit. Waiting {wait_minutes + 1} minutes..."
                        )
                        time.sleep((wait_minutes + 1) * 60)
                    else:
                        # No wait time specified, wait longer (2 minutes)
                        print(f"[Groq] Daily limit (TPD) hit. Waiting 2 minutes...")
                        time.sleep(120)
                else:
                    # Per-minute limit - use exponential backoff
                    delay = min(BASE_DELAY * (2**attempt) + random.uniform(0, 1), MAX_DELAY)
                    print(
                        f"[Groq] Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)
            except openai.APIError as e:
                raise LLMError(f"Groq API error: {e}") from e

        # All retries exhausted
        raise LLMRateLimitError(provider="groq") from last_error

    async def acomplete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async completion with automatic retry for rate limits."""
        normalized = self._normalize_messages(messages)

        # Remove unsupported or conflicting kwargs
        kwargs.pop("response_format", None)
        kwargs.pop("model", None)
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await self._async_client.chat.completions.create(
                    model=self._model,
                    messages=[m.to_dict() for m in normalized],
                    temperature=temperature,
                    max_tokens=max_tokens or 4096,
                    **kwargs,
                )

                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    provider="groq",
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

                # Check if this is a daily limit (TPD) vs per-minute limit (TPM)
                if "tokens per day" in error_msg.lower() or "tpd" in error_msg.lower():
                    # Daily limit - extract wait time from message if available
                    import re

                    wait_match = re.search(r"try again in (\d+)m", error_msg)
                    if wait_match:
                        wait_minutes = int(wait_match.group(1))
                        print(
                            f"[Groq] Daily limit (TPD) hit. Waiting {wait_minutes + 1} minutes..."
                        )
                        await asyncio.sleep((wait_minutes + 1) * 60)
                    else:
                        # No wait time specified, wait longer (2 minutes)
                        print(f"[Groq] Daily limit (TPD) hit. Waiting 2 minutes...")
                        await asyncio.sleep(120)
                else:
                    delay = min(BASE_DELAY * (2**attempt) + random.uniform(0, 1), MAX_DELAY)
                    print(
                        f"[Groq] Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(delay)
            except openai.APIError as e:
                raise LLMError(f"Groq API error: {e}") from e

        raise LLMRateLimitError(provider="groq") from last_error

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

        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in normalized],
                temperature=temperature,
                max_tokens=max_tokens or 4096,
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
            raise LLMRateLimitError(provider="groq") from e
        except openai.APIError as e:
            raise LLMError(f"Groq API error: {e}") from e

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

        try:
            stream = await self._async_client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in normalized],
                temperature=temperature,
                max_tokens=max_tokens or 4096,
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
            raise LLMRateLimitError(provider="groq") from e
        except openai.APIError as e:
            raise LLMError(f"Groq API error: {e}") from e


def get_groq(model: str | None = None, **kwargs: Any) -> GroqProvider:
    """Factory function for Groq provider.

    Args:
        model: Model name. Defaults to llama-3.3-70b-versatile.
        **kwargs: Additional provider arguments.

    Returns:
        Configured GroqProvider instance.
    """
    return GroqProvider(
        model=model or GROQ_FREE_MODELS["default"],
        **kwargs,
    )


# Exported models for factory
RECOMMENDED_MODELS = GROQ_FREE_MODELS
