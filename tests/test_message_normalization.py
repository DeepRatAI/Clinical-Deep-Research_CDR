"""
Tests for message normalization in LLM providers.

Verifies that providers correctly handle both Message objects and dict-based messages.
This ensures backward compatibility for code using dicts while enforcing type safety.

Per MEDIUM-4 audit issue: dict messages in question_parser, search_planner, synthesizer
should work with any provider.
"""

import pytest
from unittest.mock import Mock, patch

from cdr.llm.base import BaseLLMProvider, Message, LLMResponse


class ConcreteProvider(BaseLLMProvider):
    """Concrete implementation for testing."""

    @property
    def name(self) -> str:
        return "test"

    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        normalized = self._normalize_messages(messages)
        return LLMResponse(
            content="test",
            model="test-model",
            provider="test",
            usage={"input_tokens": 10, "output_tokens": 5},
        )

    async def acomplete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        normalized = self._normalize_messages(messages)
        return LLMResponse(
            content="test",
            model="test-model",
            provider="test",
            usage={"input_tokens": 10, "output_tokens": 5},
        )


class TestNormalizeMessages:
    """Tests for _normalize_messages method."""

    def test_normalize_message_objects(self):
        """Message objects pass through unchanged."""
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
        ]

        provider = ConcreteProvider(model="test")
        normalized = provider._normalize_messages(messages)

        assert len(normalized) == 2
        assert all(isinstance(m, Message) for m in normalized)
        assert normalized[0].role == "system"
        assert normalized[0].content == "You are helpful."
        assert normalized[1].role == "user"
        assert normalized[1].content == "Hello"

    def test_normalize_dict_messages(self):
        """Dict messages are converted to Message objects."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        provider = ConcreteProvider(model="test")
        normalized = provider._normalize_messages(messages)

        assert len(normalized) == 2
        assert all(isinstance(m, Message) for m in normalized)
        assert normalized[0].role == "system"
        assert normalized[0].content == "You are helpful."
        assert normalized[1].role == "user"
        assert normalized[1].content == "Hello"

    def test_normalize_mixed_messages(self):
        """Mixed Message and dict messages are normalized."""
        messages = [
            Message(role="system", content="System prompt"),
            {"role": "user", "content": "User input"},
            Message(role="assistant", content="Response"),
        ]

        provider = ConcreteProvider(model="test")
        normalized = provider._normalize_messages(messages)

        assert len(normalized) == 3
        assert all(isinstance(m, Message) for m in normalized)
        assert normalized[0].role == "system"
        assert normalized[1].role == "user"
        assert normalized[2].role == "assistant"

    def test_normalize_with_name_field(self):
        """Dict with name field is preserved."""
        messages = [
            {"role": "user", "content": "Hello", "name": "alice"},
        ]

        provider = ConcreteProvider(model="test")
        normalized = provider._normalize_messages(messages)

        assert len(normalized) == 1
        assert normalized[0].name == "alice"

    def test_normalize_empty_list(self):
        """Empty list returns empty list."""
        provider = ConcreteProvider(model="test")
        normalized = provider._normalize_messages([])
        assert normalized == []

    def test_normalize_missing_role_raises(self):
        """Dict without 'role' raises ValueError."""
        messages = [
            {"content": "Missing role"},
        ]

        provider = ConcreteProvider(model="test")
        with pytest.raises(ValueError, match="must have 'role' and 'content'"):
            provider._normalize_messages(messages)

    def test_normalize_missing_content_raises(self):
        """Dict without 'content' raises ValueError."""
        messages = [
            {"role": "user"},
        ]

        provider = ConcreteProvider(model="test")
        with pytest.raises(ValueError, match="must have 'role' and 'content'"):
            provider._normalize_messages(messages)

    def test_normalize_invalid_type_raises(self):
        """Non-Message, non-dict raises ValueError."""
        messages = [
            "just a string",
        ]

        provider = ConcreteProvider(model="test")
        with pytest.raises(ValueError, match="must be Message object or dict"):
            provider._normalize_messages(messages)

    def test_normalize_empty_content_allowed(self):
        """Empty string content is allowed (not None)."""
        messages = [
            {"role": "user", "content": ""},
        ]

        provider = ConcreteProvider(model="test")
        normalized = provider._normalize_messages(messages)

        assert len(normalized) == 1
        assert normalized[0].content == ""


class TestOpenAIProviderNormalization:
    """Tests for OpenAI provider message normalization."""

    @pytest.fixture
    def mock_openai_env(self, monkeypatch):
        """Set up mock OpenAI environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    @patch("cdr.llm.openai_provider.OpenAI")
    @patch("cdr.llm.openai_provider.AsyncOpenAI")
    def test_complete_accepts_dict_messages(self, mock_async, mock_sync, mock_openai_env):
        """OpenAI complete() accepts dict messages."""
        from cdr.llm.openai_provider import OpenAIProvider

        # Set up mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"), finish_reason="stop")]
        mock_response.model = "gpt-4"
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock_sync.return_value.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test-key")

        # Use dict messages (as in question_parser.py)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        response = provider.complete(messages)

        assert response.content == "Response"
        # Verify that to_dict was called on normalized messages
        mock_sync.return_value.chat.completions.create.assert_called_once()

    @patch("cdr.llm.openai_provider.OpenAI")
    @patch("cdr.llm.openai_provider.AsyncOpenAI")
    def test_acomplete_accepts_dict_messages(self, mock_async, mock_sync, mock_openai_env):
        """OpenAI acomplete() accepts dict messages."""
        import asyncio
        from cdr.llm.openai_provider import OpenAIProvider

        # Set up mock async response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Async response"), finish_reason="stop")]
        mock_response.model = "gpt-4"
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)

        async def mock_create(*args, **kwargs):
            return mock_response

        mock_async.return_value.chat.completions.create = mock_create

        provider = OpenAIProvider(api_key="test-key")

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User message"},
        ]

        response = asyncio.run(provider.acomplete(messages))  # type: ignore[arg-type]

        assert response.content == "Async response"


class TestAnthropicProviderNormalization:
    """Tests for Anthropic provider message normalization."""

    @pytest.fixture
    def mock_anthropic_env(self, monkeypatch):
        """Set up mock Anthropic environment."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    @patch("cdr.llm.anthropic_provider.Anthropic")
    @patch("cdr.llm.anthropic_provider.AsyncAnthropic")
    def test_complete_accepts_dict_messages(self, mock_async, mock_sync, mock_anthropic_env):
        """Anthropic complete() accepts dict messages."""
        from cdr.llm.anthropic_provider import AnthropicProvider

        # Set up mock response
        mock_block = Mock()
        mock_block.text = "Response text"
        mock_response = Mock()
        mock_response.content = [mock_block]
        mock_response.model = "claude-3"
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_sync.return_value.messages.create.return_value = mock_response

        provider = AnthropicProvider(api_key="test-key")

        # Use dict messages
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        response = provider.complete(messages)

        assert response.content == "Response text"

    @patch("cdr.llm.anthropic_provider.Anthropic")
    @patch("cdr.llm.anthropic_provider.AsyncAnthropic")
    def test_acomplete_accepts_dict_messages(self, mock_async, mock_sync, mock_anthropic_env):
        """Anthropic acomplete() accepts dict messages."""
        import asyncio
        from cdr.llm.anthropic_provider import AnthropicProvider

        # Set up mock async response
        mock_block = Mock()
        mock_block.text = "Async response"
        mock_response = Mock()
        mock_response.content = [mock_block]
        mock_response.model = "claude-3"
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"

        async def mock_create(*args, **kwargs):
            return mock_response

        mock_async.return_value.messages.create = mock_create

        provider = AnthropicProvider(api_key="test-key")

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User message"},
        ]

        response = asyncio.run(provider.acomplete(messages))  # type: ignore[arg-type]

        assert response.content == "Async response"


class TestHuggingFaceProviderNormalization:
    """Tests for HuggingFace provider message handling."""

    @pytest.fixture
    def mock_hf_env(self, monkeypatch):
        """Set up mock HuggingFace environment."""
        monkeypatch.setenv("HF_TOKEN", "test-token")
        monkeypatch.setenv("HF_ENDPOINT_URL", "https://test.endpoint.huggingface.co")

    def test_extract_message_parts_dict(self, mock_hf_env):
        """HuggingFace _extract_message_parts handles dicts."""
        from cdr.llm.huggingface_provider import HuggingFaceProvider

        provider = HuggingFaceProvider()

        role, content = provider._extract_message_parts({"role": "user", "content": "Hello"})

        assert role == "user"
        assert content == "Hello"

    def test_extract_message_parts_message(self, mock_hf_env):
        """HuggingFace _extract_message_parts handles Message objects."""
        from cdr.llm.huggingface_provider import HuggingFaceProvider

        provider = HuggingFaceProvider()

        role, content = provider._extract_message_parts(
            Message(role="assistant", content="Response")
        )

        assert role == "assistant"
        assert content == "Response"

    def test_inherits_normalize_messages(self, mock_hf_env):
        """HuggingFace inherits _normalize_messages from BaseLLMProvider."""
        from cdr.llm.huggingface_provider import HuggingFaceProvider

        provider = HuggingFaceProvider()

        # Verify method exists and works
        messages = [{"role": "user", "content": "Test"}]
        normalized = provider._normalize_messages(messages)

        assert len(normalized) == 1
        assert isinstance(normalized[0], Message)


class TestIntegrationScenarios:
    """Integration tests matching real usage patterns."""

    def test_question_parser_pattern(self):
        """Dict pattern from question_parser.py works."""
        # This is the exact pattern used in question_parser.py lines 73-76
        messages = [
            {"role": "system", "content": "Parse research questions into PICO..."},
            {"role": "user", "content": "Parse this research question:\n\nDoes X improve Y?"},
        ]

        provider = ConcreteProvider(model="test")
        normalized = provider._normalize_messages(messages)

        assert len(normalized) == 2
        assert normalized[0].role == "system"
        assert "PICO" in normalized[0].content
        assert normalized[1].role == "user"

    def test_search_planner_pattern(self):
        """Dict pattern from search_planner.py works."""
        # Pattern from search_planner.py lines 75-78
        messages = [
            {"role": "system", "content": "Generate search strategies..."},
            {"role": "user", "content": "Generate search strategy for:\n\nPopulation: Adults..."},
        ]

        provider = ConcreteProvider(model="test")
        normalized = provider._normalize_messages(messages)

        assert len(normalized) == 2
        assert "search" in normalized[0].content.lower()

    def test_synthesizer_pattern(self):
        """Dict pattern from synthesizer.py works."""
        messages = [
            {"role": "system", "content": "Synthesize evidence..."},
            {"role": "user", "content": "Here are the studies:\n\n..."},
        ]

        provider = ConcreteProvider(model="test")
        normalized = provider._normalize_messages(messages)

        assert len(normalized) == 2
        assert normalized[0].role == "system"


class TestMessageToDict:
    """Tests for Message.to_dict() method."""

    def test_to_dict_basic(self):
        """Basic Message converts to dict."""
        msg = Message(role="user", content="Hello")
        d = msg.to_dict()

        assert d == {"role": "user", "content": "Hello"}

    def test_to_dict_with_name(self):
        """Message with name includes it in dict."""
        msg = Message(role="user", content="Hello", name="alice")
        d = msg.to_dict()

        assert d == {"role": "user", "content": "Hello", "name": "alice"}

    def test_round_trip_conversion(self):
        """Message -> dict -> Message preserves data."""
        original = Message(role="assistant", content="Response", name="bot")
        d = original.to_dict()

        restored = Message(role=d["role"], content=d["content"], name=d.get("name"))

        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.name == original.name
