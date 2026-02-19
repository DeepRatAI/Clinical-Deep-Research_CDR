"""
LLM Factory

Factory for creating LLM provider instances based on configuration.

PROVIDER ORDER (all with FREE tiers):
1. Gemini (Google AI Studio) - 1M+ tokens/day free
2. Cerebras - 1M tokens/day, 60K TPM free
3. Groq - 14.4K req/day, fast inference
4. OpenRouter - 400+ models, free tier available
5. HuggingFace - Qwen/Llama via Router API
6. OpenAI/Anthropic - Paid fallbacks

Automatic fallback: If one provider hits rate limits or fails,
the system automatically tries the next available provider.
"""

import logging
import os
from typing import Literal

from cdr.config import get_settings
from cdr.core.exceptions import ConfigurationError, LLMProviderError
from cdr.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

ProviderType = Literal[
    "gemini", "cerebras", "cloudflare", "openrouter", "huggingface", "groq", "openai", "anthropic"
]


def create_provider(
    provider: ProviderType | None = None, model: str | None = None, **kwargs
) -> BaseLLMProvider:
    """
    Create an LLM provider instance.

    Args:
        provider: Provider name ('gemini', 'cerebras', 'cloudflare', 'openrouter',
                  'huggingface', 'groq', 'openai', 'anthropic').
                  Defaults to 'gemini' (FREE tier via Google AI Studio).
        model: Model name. Defaults to provider-specific default.
        **kwargs: Additional provider-specific arguments.

    Returns:
        Configured LLM provider instance.

    Raises:
        ConfigurationError: If provider is not configured.
        LLMProviderError: If provider package is not available.
    """
    settings = get_settings()

    # DEFAULT: Gemini (FREE PROVIDER - generous limits)
    if provider is None:
        provider = getattr(settings.llm, "default_provider", "gemini")

    if provider == "gemini":
        from cdr.llm.gemini_provider import GeminiProvider, GEMINI_FREE_MODELS

        # Use settings first, then fallback to os.getenv
        api_key = (
            kwargs.pop("api_key", None)
            or settings.llm.google_api_key
            or settings.llm.gemini_api_key
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        if not api_key:
            raise ConfigurationError(
                "Google AI API key not configured. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )

        default_model = GEMINI_FREE_MODELS["default"]  # gemini-2.5-flash
        return GeminiProvider(model=model or default_model, api_key=api_key, **kwargs)

    elif provider == "cerebras":
        from cdr.llm.cerebras_provider import CerebrasProvider, CEREBRAS_FREE_MODELS

        api_key = (
            kwargs.pop("api_key", None)
            or settings.llm.cerebras_api_key
            or os.getenv("CEREBRAS_API_KEY")
        )
        if not api_key:
            raise ConfigurationError(
                "Cerebras API key not configured. Set CEREBRAS_API_KEY environment variable."
            )

        default_model = CEREBRAS_FREE_MODELS["default"]  # llama-3.3-70b
        return CerebrasProvider(model=model or default_model, api_key=api_key, **kwargs)

    elif provider == "cloudflare":
        from cdr.llm.cloudflare_provider import CloudflareProvider, CLOUDFLARE_MODELS

        api_key = (
            kwargs.pop("api_key", None)
            or settings.llm.cloudflare_api_key
            or os.getenv("CLOUDFLARE_API_KEY")
        )
        account_id = (
            kwargs.pop("account_id", None)
            or settings.llm.cloudflare_account_id
            or os.getenv("CLOUDFLARE_ACCOUNT_ID")
        )
        if not api_key:
            raise ConfigurationError(
                "Cloudflare API key not configured. Set CLOUDFLARE_API_KEY environment variable."
            )
        if not account_id:
            raise ConfigurationError(
                "Cloudflare Account ID not configured. Set CLOUDFLARE_ACCOUNT_ID environment variable."
            )

        default_model = CLOUDFLARE_MODELS["default"]  # @cf/meta/llama-3.1-8b-instruct
        return CloudflareProvider(
            model=model or default_model, api_key=api_key, account_id=account_id, **kwargs
        )

    elif provider == "openrouter":
        from cdr.llm.openrouter_provider import OpenRouterProvider, OPENROUTER_MODELS

        api_key = (
            kwargs.pop("api_key", None)
            or settings.llm.openrouter_api_key
            or os.getenv("OPENROUTER_API_KEY")
        )
        if not api_key:
            raise ConfigurationError(
                "OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable."
            )

        default_model = OPENROUTER_MODELS["default"]  # meta-llama/llama-3.1-8b-instruct:free
        return OpenRouterProvider(model=model or default_model, api_key=api_key, **kwargs)

    elif provider == "huggingface":
        from cdr.llm.huggingface_provider import HuggingFaceProvider, RECOMMENDED_MODELS

        api_key = kwargs.pop("api_key", None) or settings.llm.hf_token
        if not api_key:
            api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise ConfigurationError(
                "HuggingFace API key not configured. Set HF_TOKEN environment variable."
            )

        # Use recommended model if not specified (FREE high-capacity models)
        default_model = RECOMMENDED_MODELS["default"]  # Qwen/Qwen2.5-72B-Instruct

        return HuggingFaceProvider(model=model or default_model, api_key=api_key, **kwargs)

    elif provider == "groq":
        from cdr.llm.groq_provider import GroqProvider, GROQ_FREE_MODELS

        api_key = kwargs.pop("api_key", None) or getattr(settings.llm, "groq_api_key", None)
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "Groq API key not configured. Set GROQ_API_KEY environment variable."
            )

        # Use settings model first, then fallback to GROQ_FREE_MODELS
        default_model = getattr(settings.llm, "groq_model", None) or GROQ_FREE_MODELS["large"]

        return GroqProvider(model=model or default_model, api_key=api_key, **kwargs)

    elif provider == "openai":
        from cdr.llm.openai_provider import OpenAIProvider

        api_key = kwargs.pop("api_key", None) or settings.llm.openai_api_key
        if not api_key:
            raise ConfigurationError(
                "OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
            )

        return OpenAIProvider(model=model or settings.llm.openai_model, api_key=api_key, **kwargs)

    elif provider == "anthropic":
        from cdr.llm.anthropic_provider import AnthropicProvider

        api_key = kwargs.pop("api_key", None) or settings.llm.anthropic_api_key
        if not api_key:
            raise ConfigurationError(
                "Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable."
            )

        return AnthropicProvider(
            model=model or settings.llm.anthropic_model, api_key=api_key, **kwargs
        )

    else:
        raise ConfigurationError(f"Unknown LLM provider: {provider}")


def create_provider_with_fallback(model: str | None = None, **kwargs) -> BaseLLMProvider:
    """
    Create an LLM provider with automatic fallback.

    Respects LLM_DEFAULT_PROVIDER env var to set preferred provider.
    Falls back to other providers if preferred one fails.

    Args:
        model: Model name. Defaults to provider-specific default.
        **kwargs: Additional provider-specific arguments.

    Returns:
        First successfully configured provider.

    Raises:
        ConfigurationError: If no provider is configured.
    """
    settings = get_settings()

    # Read preferred provider from config/env
    preferred = getattr(settings.llm, "default_provider", "groq")

    # Base order - will be reordered to put preferred first
    base_order = [
        "groq",  # Fast, good limits
        "cerebras",  # Fast, 1M/day
        "openrouter",  # 400+ models
        "gemini",  # 1M+/day but hitting quota
        "huggingface",
        "openai",
        "anthropic",
    ]

    # Put preferred provider first, keep others in order
    providers_to_try = [preferred] + [p for p in base_order if p != preferred]

    logger.info(f"[LLM Factory] Provider order: {providers_to_try} (preferred: {preferred})")
    errors = []

    for provider_name in providers_to_try:
        try:
            provider = create_provider(provider_name, model, **kwargs)
            logger.info(f"[LLM Factory] Using provider: {provider_name}")
            return provider
        except ConfigurationError as e:
            errors.append(f"{provider_name}: {e}")
            continue

    raise ConfigurationError(
        f"No LLM provider configured. Tried: {', '.join(providers_to_try)}. "
        f"Set GOOGLE_API_KEY, CEREBRAS_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, "
        f"HF_TOKEN, OPENAI_API_KEY, or ANTHROPIC_API_KEY. "
        f"Errors: {'; '.join(errors)}"
    )


def get_default_provider() -> BaseLLMProvider:
    """Get provider with default configuration (Gemini - FREE, highest limits)."""
    return create_provider()


# Convenience aliases for each provider
def get_gemini(model: str | None = None, **kwargs) -> BaseLLMProvider:
    """Get Gemini provider (FREE tier - 1M+ tokens/day)."""
    return create_provider("gemini", model, **kwargs)


def get_cerebras(model: str | None = None, **kwargs) -> BaseLLMProvider:
    """Get Cerebras provider (FREE tier - 1M tokens/day, ultra-fast)."""
    return create_provider("cerebras", model, **kwargs)


def get_cloudflare(model: str | None = None, **kwargs) -> BaseLLMProvider:
    """Get Cloudflare Workers AI provider (FREE tier - 10K neurons/day)."""
    return create_provider("cloudflare", model, **kwargs)


def get_openrouter(model: str | None = None, **kwargs) -> BaseLLMProvider:
    """Get OpenRouter provider (400+ models, free tier available)."""
    return create_provider("openrouter", model, **kwargs)


def get_huggingface(model: str | None = None, **kwargs) -> BaseLLMProvider:
    """Get HuggingFace provider (FREE tier via Router API)."""
    return create_provider("huggingface", model, **kwargs)


def get_groq(model: str | None = None, **kwargs) -> BaseLLMProvider:
    """Get Groq provider (FREE tier - 14.4K req/day)."""
    return create_provider("groq", model, **kwargs)


def get_openai(model: str | None = None, **kwargs) -> BaseLLMProvider:
    """Get OpenAI provider (paid)."""
    return create_provider(provider="openai", model=model, **kwargs)


def get_anthropic(model: str | None = None, **kwargs) -> BaseLLMProvider:
    """Get Anthropic provider (paid)."""
    return create_provider(provider="anthropic", model=model, **kwargs)
