"""
OpenAI Provider

LLM provider implementation for OpenAI API.
"""

import asyncio
from typing import Any, AsyncIterator, Iterator

from cdr.core.exceptions import LLMError, LLMProviderError, LLMRateLimitError
from cdr.llm.base import BaseLLMProvider, LLMResponse, Message, StreamChunk

try:
    import openai
    from openai import AsyncOpenAI, OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API provider.

    Supports:
        - GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo
        - Streaming responses
        - JSON mode
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize OpenAI provider.

        Args:
            model: Model name (gpt-4o, gpt-4-turbo, etc.).
            api_key: OpenAI API key.
            base_url: Custom base URL (for Azure or proxies).
            organization: OpenAI organization ID.
            timeout: Request timeout.
            max_retries: Maximum retries on transient errors.
        """
        if not OPENAI_AVAILABLE:
            raise LLMProviderError(
                provider="openai", message="OpenAI package not installed. Run: pip install openai"
            )

        super().__init__(model, api_key, base_url, timeout, max_retries)
        self._organization = organization

        # Initialize clients
        client_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization

        self._client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs)

    @property
    def name(self) -> str:
        return "openai"

    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous completion."""
        # Normalize messages to handle both Message objects and dicts
        normalized = self._normalize_messages(messages)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in normalized],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                provider=self.name,
                usage={
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                finish_reason=response.choices[0].finish_reason,
                raw_response=response,
            )

        except openai.RateLimitError as e:
            raise LLMRateLimitError(
                provider=self.name,
                retry_after=float(e.response.headers.get("Retry-After", 60))
                if hasattr(e, "response") and e.response
                else 60.0,
            ) from e

        except openai.APIError as e:
            raise LLMProviderError(provider=self.name, message=str(e)) from e

        except Exception as e:
            raise LLMError(f"OpenAI error: {e}") from e

    async def acomplete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async completion."""
        # Normalize messages to handle both Message objects and dicts
        normalized = self._normalize_messages(messages)
        try:
            response = await self._async_client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in normalized],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                provider=self.name,
                usage={
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                finish_reason=response.choices[0].finish_reason,
                raw_response=response,
            )

        except openai.RateLimitError as e:
            raise LLMRateLimitError(
                provider=self.name,
                retry_after=float(e.response.headers.get("Retry-After", 60))
                if hasattr(e, "response") and e.response
                else 60.0,
            ) from e

        except openai.APIError as e:
            raise LLMProviderError(provider=self.name, message=str(e)) from e

        except Exception as e:
            raise LLMError(f"OpenAI async error: {e}") from e

    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Streaming completion."""
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        content=chunk.choices[0].delta.content,
                        is_final=chunk.choices[0].finish_reason is not None,
                        finish_reason=chunk.choices[0].finish_reason,
                    )

        except Exception as e:
            raise LLMError(f"OpenAI streaming error: {e}") from e

    async def astream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Async streaming completion."""
        try:
            stream = await self._async_client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        content=chunk.choices[0].delta.content,
                        is_final=chunk.choices[0].finish_reason is not None,
                        finish_reason=chunk.choices[0].finish_reason,
                    )

        except Exception as e:
            raise LLMError(f"OpenAI async streaming error: {e}") from e

    def complete_json(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Completion with JSON mode enabled.

        Note: The prompt should instruct the model to output JSON.
        """
        return self.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            **kwargs,
        )

    async def acomplete_json(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async completion with JSON mode."""
        return await self.acomplete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            **kwargs,
        )
