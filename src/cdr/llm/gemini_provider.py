"""
Google Gemini LLM Provider

Integration with Google AI Studio (Gemini) API via OpenAI-compatible endpoint.
Gemini offers FREE tier with generous limits:
- gemini-2.5-flash: Free tier available
- gemini-2.0-flash: Free tier available
- gemini-1.5-flash: Free tier available

Base URL: https://generativelanguage.googleapis.com/v1beta/openai/
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
MAX_DELAY = 30.0  # seconds


# Gemini FREE tier models
GEMINI_FREE_MODELS = {
    "default": "gemini-2.5-flash",  # Best balance of quality and speed
    "fast": "gemini-2.0-flash",  # Fastest
    "reasoning": "gemini-2.5-flash",  # Good reasoning capabilities
    "large": "gemini-2.5-flash",  # Larger context
    "pro": "gemini-2.5-pro",  # Most capable (may have limits)
}


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider - FREE tier via OpenAI-compatible API.

    Gemini provides OpenAI-compatible API endpoint for easy integration.
    This allows using the standard OpenAI SDK with just a base_url change.

    API Key: Get from https://aistudio.google.com/apikey
    """

    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        timeout: float = 120.0,  # Gemini can be slower
        max_retries: int = 3,
    ) -> None:
        """Initialize Gemini provider.

        Args:
            model: Model name (gemini-2.5-flash, gemini-2.0-flash, gemini-2.5-pro).
            api_key: Google AI API key. Falls back to GOOGLE_API_KEY or GEMINI_API_KEY env var.
            timeout: Request timeout.
            max_retries: Maximum retries on transient errors.
        """
        if not OPENAI_AVAILABLE:
            raise LLMProviderError(
                provider="gemini",
                message="OpenAI package not installed. Run: pip install openai",
            )

        # Get API key from env if not provided
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMProviderError(
                provider="gemini",
                message="Google AI API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY env var.",
            )

        super().__init__(model, api_key, self.GEMINI_BASE_URL, timeout, max_retries)

        # Initialize OpenAI-compatible clients with Gemini base URL
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.GEMINI_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.GEMINI_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        return "gemini"

    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous completion with automatic retry for rate limits."""
        normalized = self._normalize_messages(messages)

        # Remove unsupported kwargs for Gemini
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
                    max_tokens=max_tokens or 8192,  # Gemini supports larger outputs
                    **kwargs,
                )

                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    provider="gemini",
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
                    print(f"[Gemini] Rate limit hit. Waiting {wait_seconds + 5} seconds...")
                    time.sleep(wait_seconds + 5)
                else:
                    delay = min(BASE_DELAY * (2**attempt) + random.uniform(0, 1), MAX_DELAY)
                    print(
                        f"[Gemini] Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)
            except openai.APIError as e:
                raise LLMError(f"Gemini API error: {e}") from e

        raise LLMRateLimitError(provider="gemini") from last_error

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
                    max_tokens=max_tokens or 8192,
                    **kwargs,
                )

                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    provider="gemini",
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
                    print(f"[Gemini] Rate limit hit. Waiting {wait_seconds + 5} seconds...")
                    await asyncio.sleep(wait_seconds + 5)
                else:
                    delay = min(BASE_DELAY * (2**attempt) + random.uniform(0, 1), MAX_DELAY)
                    print(
                        f"[Gemini] Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(delay)
            except openai.APIError as e:
                raise LLMError(f"Gemini API error: {e}") from e

        raise LLMRateLimitError(provider="gemini") from last_error

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
                max_tokens=max_tokens or 8192,
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
            raise LLMRateLimitError(provider="gemini") from e
        except openai.APIError as e:
            raise LLMError(f"Gemini API error: {e}") from e

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
                max_tokens=max_tokens or 8192,
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
            raise LLMRateLimitError(provider="gemini") from e
        except openai.APIError as e:
            raise LLMError(f"Gemini API error: {e}") from e
