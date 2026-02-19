"""
CDR LLM Abstraction Layer

Unified interface for multiple LLM providers.

FREE PROVIDERS (in order of preference):
1. Gemini (Google AI Studio) - 1M+ tokens/day free
2. Cerebras - 1M tokens/day, 60K TPM free
3. Groq - 14.4K req/day, fast inference
4. OpenRouter - 400+ models, free tier available
5. HuggingFace - Qwen/Llama via Router API

PAID FALLBACKS:
- OpenAI
- Anthropic
"""

from cdr.llm.base import (
    BaseLLMProvider,
    LLMProvider,
    LLMResponse,
    Message,
    StreamChunk,
    build_messages,
)
from cdr.llm.factory import (
    create_provider,
    create_provider_with_fallback,
    get_anthropic,
    get_cerebras,
    get_cloudflare,
    get_default_provider,
    get_gemini,
    get_groq,
    get_huggingface,
    get_openai,
    get_openrouter,
)

__all__ = [
    # Base
    "BaseLLMProvider",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "StreamChunk",
    "build_messages",
    # Factory
    "create_provider",
    "create_provider_with_fallback",
    "get_default_provider",
    # FREE providers (in order of limits)
    "get_gemini",  # PRIMARY (FREE - highest limits)
    "get_cerebras",  # FREE - very fast
    "get_cloudflare",  # FREE - edge inference
    "get_openrouter",  # FREE tier + 400+ models
    "get_groq",  # FREE - fast
    "get_huggingface",  # FREE via Router
    # Paid providers
    "get_openai",
    "get_anthropic",
]
