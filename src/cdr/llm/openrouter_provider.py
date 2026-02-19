"""
OpenRouter LLM Provider

Integration with OpenRouter API via OpenAI-compatible endpoint.
OpenRouter provides unified access to 400+ AI models through a single API.

Features:
- Access to many free models (suffix :free)
- Automatic fallbacks between providers
- Pay-per-use for premium models
- Support for tool calling, structured outputs, etc.

Base URL: https://openrouter.ai/api/v1
Uses standard OpenAI SDK with modified base_url.

Free tier limits (models ending in :free):
- 20 requests/minute
- 50 requests/day (if <10 credits purchased)
- 1000 requests/day (if ≥10 credits purchased)
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


# OpenRouter recommended models (all very affordable)
OPENROUTER_MODELS = {
    # Primary models (very cheap or free)
    "default": "meta-llama/llama-3.1-8b-instruct",  # Very cheap
    "fast": "meta-llama/llama-3.1-8b-instruct",
    "large": "meta-llama/llama-3.1-70b-instruct",  # Better quality
    # Reasoning models
    "reasoning": "deepseek/deepseek-r1",
    # Best quality (more expensive)
    "premium": "anthropic/claude-3.5-sonnet",
    "gpt": "openai/gpt-4o-mini",
}


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter LLM provider - unified access to 400+ models.

    OpenRouter provides a single API endpoint for accessing many different
    AI models from various providers (OpenAI, Anthropic, Google, Meta, etc.).

    API Key: Get from https://openrouter.ai/keys
    Documentation: https://openrouter.ai/docs

    Free models: Append :free to model name (e.g., meta-llama/llama-3.1-8b-instruct:free)
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str = "meta-llama/llama-3.1-8b-instruct",
        api_key: str | None = None,
        timeout: float = 120.0,  # Some models can be slow
        max_retries: int = 3,
        site_url: str | None = None,  # For rankings on openrouter.ai
        site_name: str | None = None,  # For rankings on openrouter.ai
    ) -> None:
        """Initialize OpenRouter provider.

        Args:
            model: Model name (e.g., meta-llama/llama-3.1-8b-instruct:free).
            api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
            timeout: Request timeout.
            max_retries: Maximum retries on transient errors.
            site_url: Optional site URL for rankings.
            site_name: Optional site name for rankings.
        """
        if not OPENAI_AVAILABLE:
            raise LLMProviderError(
                provider="openrouter",
                message="OpenAI package not installed. Run: pip install openai",
            )

        # Get API key from env if not provided
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMProviderError(
                provider="openrouter",
                message="OpenRouter API key required. Set OPENROUTER_API_KEY env var.",
            )

        super().__init__(model, api_key, self.OPENROUTER_BASE_URL, timeout, max_retries)

        # Build default headers for OpenRouter
        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if site_name:
            default_headers["X-Title"] = site_name

        # Initialize OpenAI-compatible clients with OpenRouter base URL
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.OPENROUTER_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers if default_headers else None,
        )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.OPENROUTER_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers if default_headers else None,
        )

    @property
    def name(self) -> str:
        return "openrouter"

    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous completion with automatic retry for rate limits."""
        normalized = self._normalize_messages(messages)

        # Remove conflicting kwargs (OpenRouter passes most through)
        kwargs.pop("model", None)

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
                    provider="openrouter",
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

                # Check for daily limit on free models
                if "daily" in error_msg.lower() or "limit" in error_msg.lower():
                    print(f"[OpenRouter] Daily/rate limit hit: {error_msg}")
                    # Wait longer for daily limits
                    delay = min(60 * (attempt + 1), 300)  # Up to 5 minutes
                else:
                    delay = min(BASE_DELAY * (2**attempt) + random.uniform(0, 1), MAX_DELAY)

                print(
                    f"[OpenRouter] Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(delay)
            except openai.APIError as e:
                raise LLMError(f"OpenRouter API error: {e}") from e

        raise LLMRateLimitError(provider="openrouter") from last_error

    async def acomplete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async completion with automatic retry for rate limits."""
        normalized = self._normalize_messages(messages)

        # Remove conflicting kwargs
        kwargs.pop("model", None)

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
                    provider="openrouter",
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

                if "daily" in error_msg.lower() or "limit" in error_msg.lower():
                    print(f"[OpenRouter] Daily/rate limit hit: {error_msg}")
                    delay = min(60 * (attempt + 1), 300)
                else:
                    delay = min(BASE_DELAY * (2**attempt) + random.uniform(0, 1), MAX_DELAY)

                print(
                    f"[OpenRouter] Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                )
                await asyncio.sleep(delay)
            except openai.APIError as e:
                raise LLMError(f"OpenRouter API error: {e}") from e

        raise LLMRateLimitError(provider="openrouter") from last_error

    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Streaming completion."""
        normalized = self._normalize_messages(messages)

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
            raise LLMRateLimitError(provider="openrouter") from e
        except openai.APIError as e:
            raise LLMError(f"OpenRouter API error: {e}") from e

    async def astream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Async streaming completion."""
        normalized = self._normalize_messages(messages)

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
            raise LLMRateLimitError(provider="openrouter") from e
        except openai.APIError as e:
            raise LLMError(f"OpenRouter API error: {e}") from e
