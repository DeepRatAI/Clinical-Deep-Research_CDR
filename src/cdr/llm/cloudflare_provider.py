"""
Cloudflare Workers AI LLM Provider

Integration with Cloudflare Workers AI API via OpenAI-compatible endpoint.
Cloudflare offers FREE tier with generous limits:
- 10,000 Neurons per day free (roughly equivalent to ~100K tokens)
- Text Generation: 300 requests/min
- Models: llama-3.1-8b-instruct, llama-3.3-70b-instruct-fp8-fast, etc.

Base URL: https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1
Uses standard OpenAI SDK with modified base_url.
Requires both API Token and Account ID.
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
MAX_DELAY = 30.0  # seconds


# Cloudflare Workers AI models (using @cf prefix format)
CLOUDFLARE_MODELS = {
    "default": "@cf/meta/llama-3.1-8b-instruct",  # Fast, good quality
    "fast": "@cf/meta/llama-3.1-8b-instruct-fast",  # Fastest
    "large": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",  # Best quality
    "reasoning": "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",  # Reasoning
    "qwen": "@cf/qwen/qwen3-30b-a3b-fp8",  # Qwen model
}

# Cloudflare Free tier limits
CLOUDFLARE_FREE_LIMITS = {
    "neurons_per_day": 10_000,
    "requests_per_minute_text_gen": 300,
}


class CloudflareProvider(BaseLLMProvider):
    """Cloudflare Workers AI LLM provider - FREE tier via OpenAI-compatible API.

    Cloudflare Workers AI runs models at edge locations globally.
    OpenAI-compatible API endpoint for easy integration.

    API Token: Create at https://dash.cloudflare.com/profile/api-tokens
    Account ID: Found at https://dash.cloudflare.com/?to=/:account/ai/workers-ai
    Free tier: 10,000 neurons/day (roughly 100K tokens)
    """

    @staticmethod
    def get_base_url(account_id: str) -> str:
        """Get the base URL for Cloudflare Workers AI."""
        return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"

    def __init__(
        self,
        model: str = "@cf/meta/llama-3.1-8b-instruct",
        api_key: str | None = None,
        account_id: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize Cloudflare Workers AI provider.

        Args:
            model: Model name (e.g., @cf/meta/llama-3.1-8b-instruct).
            api_key: Cloudflare API Token. Falls back to CLOUDFLARE_API_KEY env var.
            account_id: Cloudflare Account ID. Falls back to CLOUDFLARE_ACCOUNT_ID env var.
            timeout: Request timeout.
            max_retries: Maximum retries on transient errors.
        """
        if not OPENAI_AVAILABLE:
            raise LLMProviderError(
                provider="cloudflare",
                message="OpenAI package not installed. Run: pip install openai",
            )

        # Get API key and account ID from env if not provided
        api_key = api_key or os.getenv("CLOUDFLARE_API_KEY")
        if not api_key:
            raise LLMProviderError(
                provider="cloudflare",
                message="Cloudflare API key required. Set CLOUDFLARE_API_KEY env var.",
            )

        account_id = account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID")
        if not account_id:
            raise LLMProviderError(
                provider="cloudflare",
                message="Cloudflare Account ID required. Set CLOUDFLARE_ACCOUNT_ID env var.",
            )

        base_url = self.get_base_url(account_id)
        super().__init__(model, api_key, base_url, timeout, max_retries)

        # Initialize OpenAI-compatible clients with Cloudflare base URL
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        return "cloudflare"

    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous completion with automatic retry for rate limits."""
        normalized = self._normalize_messages(messages)

        # Remove unsupported kwargs for Cloudflare
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
                    provider="cloudflare",
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
                delay = min(BASE_DELAY * (2**attempt) + random.uniform(0, 1), MAX_DELAY)
                print(
                    f"[Cloudflare] Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(delay)
            except openai.APIError as e:
                raise LLMError(f"Cloudflare API error: {e}") from e

        raise LLMRateLimitError(provider="cloudflare") from last_error

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
                    provider="cloudflare",
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
                delay = min(BASE_DELAY * (2**attempt) + random.uniform(0, 1), MAX_DELAY)
                print(
                    f"[Cloudflare] Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                )
                await asyncio.sleep(delay)
            except openai.APIError as e:
                raise LLMError(f"Cloudflare API error: {e}") from e

        raise LLMRateLimitError(provider="cloudflare") from last_error

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
            raise LLMRateLimitError(provider="cloudflare") from e
        except openai.APIError as e:
            raise LLMError(f"Cloudflare API error: {e}") from e

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
            raise LLMRateLimitError(provider="cloudflare") from e
        except openai.APIError as e:
            raise LLMError(f"Cloudflare API error: {e}") from e
