"""
Anthropic Provider

LLM provider implementation for Anthropic Claude API.
"""

from typing import Any, AsyncIterator, Iterator

from cdr.core.exceptions import LLMError, LLMProviderError, LLMRateLimitError
from cdr.llm.base import BaseLLMProvider, LLMResponse, Message, StreamChunk

try:
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude API provider.

    Supports:
        - Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
        - Streaming responses
        - Extended context (up to 200K tokens)
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize Anthropic provider.

        Args:
            model: Model name (claude-3-5-sonnet-20241022, etc.).
            api_key: Anthropic API key.
            base_url: Custom base URL.
            timeout: Request timeout.
            max_retries: Maximum retries.
        """
        if not ANTHROPIC_AVAILABLE:
            raise LLMProviderError(
                provider="anthropic",
                message="Anthropic package not installed. Run: pip install anthropic",
            )

        super().__init__(model, api_key, base_url, timeout, max_retries)

        # Initialize clients
        client_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = Anthropic(**client_kwargs)
        self._async_client = AsyncAnthropic(**client_kwargs)

    @property
    def name(self) -> str:
        return "anthropic"

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, str]]]:
        """
        Convert messages to Anthropic format.

        Anthropic uses separate system parameter.

        Returns:
            Tuple of (system_prompt, messages).
        """
        system_prompt: str | None = None
        converted: list[dict[str, str]] = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                converted.append({"role": msg.role, "content": msg.content})

        return system_prompt, converted

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
        system_prompt, converted_messages = self._convert_messages(normalized)

        try:
            call_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": converted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096,
            }
            if system_prompt:
                call_kwargs["system"] = system_prompt
            call_kwargs.update(kwargs)

            response = self._client.messages.create(**call_kwargs)

            # Extract text from content blocks
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.name,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                finish_reason=response.stop_reason,
                raw_response=response,
            )

        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(
                provider=self.name,
                retry_after=60.0,  # Anthropic doesn't always provide retry-after
            ) from e

        except anthropic.APIError as e:
            raise LLMProviderError(provider=self.name, message=str(e)) from e

        except Exception as e:
            raise LLMError(f"Anthropic error: {e}") from e

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
        system_prompt, converted_messages = self._convert_messages(normalized)

        try:
            call_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": converted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096,
            }
            if system_prompt:
                call_kwargs["system"] = system_prompt
            call_kwargs.update(kwargs)

            response = await self._async_client.messages.create(**call_kwargs)

            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.name,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                finish_reason=response.stop_reason,
                raw_response=response,
            )

        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(provider=self.name, retry_after=60.0) from e

        except anthropic.APIError as e:
            raise LLMProviderError(provider=self.name, message=str(e)) from e

        except Exception as e:
            raise LLMError(f"Anthropic async error: {e}") from e

    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Streaming completion."""
        system_prompt, converted_messages = self._convert_messages(messages)

        try:
            call_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": converted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096,
            }
            if system_prompt:
                call_kwargs["system"] = system_prompt
            call_kwargs.update(kwargs)

            with self._client.messages.stream(**call_kwargs) as stream:
                for text in stream.text_stream:
                    yield StreamChunk(content=text, is_final=False)

                # Final chunk
                yield StreamChunk(content="", is_final=True)

        except Exception as e:
            raise LLMError(f"Anthropic streaming error: {e}") from e

    async def astream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Async streaming completion."""
        system_prompt, converted_messages = self._convert_messages(messages)

        try:
            call_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": converted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096,
            }
            if system_prompt:
                call_kwargs["system"] = system_prompt
            call_kwargs.update(kwargs)

            async with self._async_client.messages.stream(**call_kwargs) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(content=text, is_final=False)

                yield StreamChunk(content="", is_final=True)

        except Exception as e:
            raise LLMError(f"Anthropic async streaming error: {e}") from e
