"""
LLM Provider Base

Abstract base class and protocols for LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class Message:
    """Chat message."""

    role: str  # system, user, assistant
    content: str
    name: str | None = None

    def to_dict(self) -> dict[str, str]:
        """Convert to dict for API calls."""
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    raw_response: Any | None = None

    @property
    def input_tokens(self) -> int:
        """Get input token count."""
        return self.usage.get("input_tokens", 0) or self.usage.get("prompt_tokens", 0)

    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        return self.usage.get("output_tokens", 0) or self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.input_tokens + self.output_tokens


@dataclass
class StreamChunk:
    """Streaming response chunk."""

    content: str
    is_final: bool = False
    finish_reason: str | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    @property
    def name(self) -> str:
        """Provider name."""
        ...

    @property
    def model(self) -> str:
        """Current model."""
        ...

    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous completion."""
        ...

    async def acomplete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async completion."""
        ...


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Provides common functionality and enforces interface.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize provider.

        Args:
            model: Model identifier.
            api_key: API key (if required).
            base_url: Custom API endpoint.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
        """
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries

    @staticmethod
    def _normalize_messages(messages: list[Message] | list[dict]) -> list[Message]:
        """
        Normalize messages to Message objects.

        Accepts both Message objects and dicts with role/content keys.
        This provides backward compatibility for code that uses dict-based messages.

        Args:
            messages: List of Message objects or dicts with 'role' and 'content' keys.

        Returns:
            List of Message objects.

        Raises:
            ValueError: If message format is invalid.
        """
        normalized = []
        for msg in messages:
            if isinstance(msg, Message):
                normalized.append(msg)
            elif isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
                if not role or content is None:
                    raise ValueError(
                        f"Dict message must have 'role' and 'content' keys, got: {msg.keys()}"
                    )
                name = msg.get("name")
                normalized.append(Message(role=role, content=content, name=name))
            else:
                raise ValueError(
                    f"Message must be Message object or dict, got: {type(msg).__name__}"
                )
        return normalized

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic')."""
        ...

    @property
    def model(self) -> str:
        """Current model."""
        return self._model

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Synchronous completion.

        Args:
            messages: List of chat messages.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate.
            **kwargs: Provider-specific arguments.

        Returns:
            LLMResponse with generated content.
        """
        ...

    @abstractmethod
    async def acomplete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async completion."""
        ...

    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """
        Streaming completion (sync).

        Default implementation falls back to non-streaming.
        Override in subclasses for true streaming.
        """
        response = self.complete(messages, temperature, max_tokens, **kwargs)
        yield StreamChunk(content=response.content, is_final=True)

    async def astream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """
        Streaming completion (async).

        Default implementation falls back to non-streaming.
        """
        response = await self.acomplete(messages, temperature, max_tokens, **kwargs)
        yield StreamChunk(content=response.content, is_final=True)

    def _build_system_message(self, system_prompt: str) -> Message:
        """Build a system message."""
        return Message(role="system", content=system_prompt)

    def _build_user_message(self, content: str) -> Message:
        """Build a user message."""
        return Message(role="user", content=content)

    def _build_assistant_message(self, content: str) -> Message:
        """Build an assistant message."""
        return Message(role="assistant", content=content)


def build_messages(
    system: str | None = None, user: str | None = None, history: list[tuple[str, str]] | None = None
) -> list[Message]:
    """
    Convenience function to build message list.

    Args:
        system: System prompt.
        user: User message.
        history: List of (user, assistant) message pairs.

    Returns:
        List of Message objects.
    """
    messages: list[Message] = []

    if system:
        messages.append(Message(role="system", content=system))

    if history:
        for user_msg, assistant_msg in history:
            messages.append(Message(role="user", content=user_msg))
            messages.append(Message(role="assistant", content=assistant_msg))

    if user:
        messages.append(Message(role="user", content=user))

    return messages
