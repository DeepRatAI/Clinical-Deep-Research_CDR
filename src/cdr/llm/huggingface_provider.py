"""
Hugging Face LLM Provider

Integration with Hugging Face Inference API / Endpoints / TGI.
This is the PRIMARY provider for CDR (not OpenAI/Anthropic).

Implements robust retry with exponential backoff for handling:
- 500 Internal Server Error (transient server issues)
- 503 Service Unavailable (model loading)
- 429 Too Many Requests (rate limiting)

Also implements concurrency throttling to avoid overwhelming the API
with parallel requests (which causes 500 errors on HF Router).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)

from cdr.llm.base import BaseLLMProvider, LLMResponse, Message

# Global semaphore to limit concurrent HuggingFace API requests
# This prevents 500 errors caused by too many parallel requests
_HF_CONCURRENCY_LIMIT = 3  # Max concurrent requests to HF Router
_hf_semaphore: asyncio.Semaphore | None = None


def _get_hf_semaphore() -> asyncio.Semaphore:
    """Get or create the global HuggingFace semaphore.

    Lazy initialization to ensure it's created in the correct event loop.
    """
    global _hf_semaphore
    if _hf_semaphore is None:
        _hf_semaphore = asyncio.Semaphore(_HF_CONCURRENCY_LIMIT)
    return _hf_semaphore


class RetryableHTTPError(Exception):
    """Exception raised for retryable HTTP errors (429, 500, 502, 503, 504).

    This is used to distinguish retryable errors from non-retryable ones
    (like 402 Payment Required) in the tenacity retry logic.
    """

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


logger = logging.getLogger(__name__)


class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face LLM provider.

    Supports:
    - Inference API (serverless)
    - Inference Endpoints (dedicated)
    - Text Generation Inference (TGI)

    This is the PRIMARY provider for CDR. OpenAI/Anthropic are fallbacks.
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint_url: str | None = None,
        model: str = "meta-llama/Llama-3.2-3B-Instruct",  # FREE tier model
        timeout: float = 120.0,
    ) -> None:
        """Initialize Hugging Face provider.

        Args:
            api_key: HF API token (or HF_TOKEN env var)
            endpoint_url: Custom endpoint URL (or HF_ENDPOINT_URL env var)
            model: Model identifier or endpoint model
            timeout: Request timeout in seconds
        """
        self._api_key = api_key or os.getenv("HF_TOKEN")
        if not self._api_key:
            raise ValueError("HuggingFace API key required. Set HF_TOKEN env var or pass api_key.")

        # Support custom endpoints (TGI, Inference Endpoints)
        _endpoint = endpoint_url or os.getenv("HF_ENDPOINT_URL")
        if not _endpoint:
            # Use new Router API with OpenAI-compatible endpoint
            # The old api-inference.huggingface.co is deprecated (410 Gone)
            _endpoint = "https://router.huggingface.co/v1/chat/completions"
        self._endpoint_url: str = _endpoint  # Always set to a valid URL

        # Flag to detect if using OpenAI-compatible Router API
        self._use_openai_format = "router.huggingface.co/v1" in self._endpoint_url

        self._model = model
        self._timeout = timeout

        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

        # Retry configuration for transient errors (500, 503, 429)
        # Using exponential backoff with jitter to avoid thundering herd
        self._max_retries = 5
        self._retry_min_wait = 2  # seconds
        self._retry_max_wait = 60  # seconds

    def _is_retryable_status(self, status_code: int) -> bool:
        """Check if HTTP status code is retryable.

        Retryable codes (transient server errors):
        - 429: Too Many Requests (rate limiting) - RETRYABLE
        - 500: Internal Server Error (transient)
        - 502: Bad Gateway (load balancer issues)
        - 503: Service Unavailable (model loading)
        - 504: Gateway Timeout

        NON-retryable codes (client/payment errors):
        - 400: Bad Request (client error)
        - 401: Unauthorized (auth error)
        - 402: Payment Required (NOT transient - billing issue)
        - 403: Forbidden (permission error)
        - 404: Not Found (resource doesn't exist)
        """
        return status_code in {429, 500, 502, 503, 504}

    @property
    def name(self) -> str:
        """Provider name."""
        return "huggingface"

    @property
    def model(self) -> str:
        """Current model."""
        return self._model

    def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous completion.

        Handles async context properly by running in a dedicated thread
        with its own event loop to avoid uvloop/nest_asyncio issues.
        """
        import asyncio
        import concurrent.futures
        import threading

        try:
            # Check if there's already a running event loop
            asyncio.get_running_loop()
            loop_running = True
        except RuntimeError:
            loop_running = False

        if loop_running:
            # Already in async context - run in separate thread with dedicated loop
            # Use a thread-local loop to avoid "Event loop is closed" issues

            # Create a persistent executor if not exists
            if not hasattr(self, "_sync_executor"):
                self._sync_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="hf_llm_sync_"
                )

            # Thread-local storage for event loops
            if not hasattr(self, "_thread_local"):
                self._thread_local = threading.local()

            def run_in_thread_loop():
                # Get or create thread-local event loop
                if not hasattr(self._thread_local, "loop") or self._thread_local.loop.is_closed():
                    self._thread_local.loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._thread_local.loop)

                return self._thread_local.loop.run_until_complete(
                    self.acomplete(messages, model, temperature, max_tokens, **kwargs)
                )

            future = self._sync_executor.submit(run_in_thread_loop)
            return future.result(timeout=self._timeout + 30)
        else:
            # No async context - use persistent event loop
            if not hasattr(self, "_sync_loop") or self._sync_loop.is_closed():
                self._sync_loop = asyncio.new_event_loop()
            return self._sync_loop.run_until_complete(
                self.acomplete(messages, model, temperature, max_tokens, **kwargs)
            )

    async def acomplete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async completion via Hugging Face API.

        Args:
            messages: List of message dicts with role/content
            model: Override model (if using Inference API)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model parameters

        Returns:
            LLMResponse with generated text and metadata
        """
        use_model = model or self._model

        # Check if using OpenAI-compatible Router API
        if self._use_openai_format:
            return await self._acomplete_openai_format(
                messages, use_model, temperature, max_tokens, **kwargs
            )

        # Legacy Inference API format
        return await self._acomplete_legacy_format(
            messages, use_model, temperature, max_tokens, **kwargs
        )

    async def _acomplete_openai_format(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> LLMResponse:
        """OpenAI-compatible API format (router.huggingface.co/v1).

        Implements robust retry with exponential backoff for transient errors:
        - 429 Too Many Requests (rate limiting)
        - 500/502/503/504 Server errors (transient issues)

        Supports structured outputs via response_format parameter:
        - response_format={"type": "json_object"} - Force JSON output
        - response_format={"type": "json_schema", "json_schema": {...}} - Force schema-compliant JSON

        Documentation: https://huggingface.co/docs/api-inference/tasks/chat-completion
        """
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            role, content = self._extract_message_parts(msg)
            openai_messages.append({"role": role, "content": content})

        payload = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Handle response_format for structured outputs
        response_format = kwargs.pop("response_format", None)
        if response_format:
            payload["response_format"] = response_format

        # Handle tools for function calling
        tools = kwargs.pop("tools", None)
        if tools:
            payload["tools"] = tools
            # Default to auto tool choice if tools provided
            tool_choice = kwargs.pop("tool_choice", "auto")
            payload["tool_choice"] = tool_choice

        # Add remaining kwargs
        payload.update(kwargs)

        # Define retry-aware request function with concurrency limiting
        async def _make_request_with_retry() -> httpx.Response:
            """Make HTTP request with retry logic and concurrency limiting.

            Uses a global semaphore to limit concurrent requests to HuggingFace API,
            preventing 500 errors caused by overwhelming the Router API with
            parallel requests. This is especially important for bulk operations
            like screening, extraction, and risk-of-bias assessment.

            Retry Logic:
            - Only retries on specific transient errors (429, 500, 502, 503, 504)
            - Does NOT retry on payment errors (402) or auth errors (401, 403)
            - Uses exponential backoff with jitter to avoid thundering herd
            """
            semaphore = _get_hf_semaphore()

            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential_jitter(
                    initial=self._retry_min_wait,
                    max=self._retry_max_wait,
                    jitter=2,  # Add random jitter to avoid thundering herd
                ),
                retry=retry_if_exception_type(RetryableHTTPError),  # Only retry RetryableHTTPError
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    # Acquire semaphore to limit concurrent requests
                    async with semaphore:
                        logger.debug(
                            f"[HF] Making request (attempt {attempt.retry_state.attempt_number}/{self._max_retries})"
                        )
                        response = await self.client.post(
                            self._endpoint_url,
                            json=payload,
                        )

                    # Check response status
                    if response.status_code >= 400:
                        if self._is_retryable_status(response.status_code):
                            # Retryable error - raise RetryableHTTPError to trigger retry
                            logger.warning(
                                f"[HF] Retryable error {response.status_code} on attempt "
                                f"{attempt.retry_state.attempt_number}/{self._max_retries}"
                            )
                            raise RetryableHTTPError(
                                response.status_code, f"Retryable error from {self._endpoint_url}"
                            )
                        else:
                            # Non-retryable error (402, 401, 403, 400, 404, etc.)
                            # Raise immediately without retry
                            logger.error(
                                f"[HF] Non-retryable error {response.status_code} - "
                                f"will not retry. Response: {response.text[:200]}"
                            )
                            response.raise_for_status()

                    return response

            # This should not be reached due to reraise=True
            raise RuntimeError("Retry exhausted without result")

        try:
            response = await _make_request_with_retry()
        except RetryableHTTPError as e:
            # All retries exhausted for a retryable error
            logger.error(f"[HF] Request failed after {self._max_retries} attempts: {e}")
            raise httpx.HTTPStatusError(
                str(e),
                request=httpx.Request("POST", self._endpoint_url),
                response=httpx.Response(e.status_code),
            )
        except httpx.HTTPStatusError as e:
            # If 400 error and we used response_format or tools, retry without them
            if e.response.status_code == 400 and (response_format or tools):
                logger.info("[HF] Model may not support response_format/tools, retrying without...")
                # Remove response_format and tools from payload
                payload.pop("response_format", None)
                payload.pop("tools", None)
                payload.pop("tool_choice", None)

                try:
                    response = await _make_request_with_retry()
                except RetryableHTTPError as retry_e:
                    logger.error(f"[HF] Retry without format also failed: {retry_e}")
                    raise httpx.HTTPStatusError(
                        str(retry_e),
                        request=httpx.Request("POST", self._endpoint_url),
                        response=httpx.Response(retry_e.status_code),
                    )
            else:
                # Log and re-raise for non-recoverable errors
                logger.error(f"[HF] Non-recoverable error: {e}")
                raise

        result = response.json()

        # Handle tool calls in response
        message = result["choices"][0]["message"]

        # Check for tool_calls (function calling response)
        if message.get("tool_calls"):
            # Extract function call arguments as content
            tool_call = message["tool_calls"][0]
            if tool_call.get("function"):
                generated_text = tool_call["function"].get("arguments", "{}")
            else:
                generated_text = message.get("content", "")
        else:
            generated_text = message.get("content", "")

        usage = result.get("usage", {})

        return LLMResponse(
            content=generated_text.strip() if generated_text else "",
            model=model,
            provider=self.name,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", -1),
                "completion_tokens": usage.get("completion_tokens", -1),
                "total_tokens": usage.get("total_tokens", -1),
            },
            raw_response={"endpoint": self._endpoint_url, "tool_calls": message.get("tool_calls")},
        )

    async def _acomplete_legacy_format(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> LLMResponse:
        """Legacy Inference API format (TGI, custom endpoints)."""
        # Build prompt from messages (chat template)
        prompt = self._build_prompt(messages)

        # Prepare request payload
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False,
                "do_sample": temperature > 0,
                **kwargs,
            },
        }

        # Make request
        response = await self.client.post(
            self._endpoint_url,
            json=payload,
        )

        response.raise_for_status()

        # Parse response
        result = response.json()

        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            # Inference API format
            generated_text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            # TGI / Endpoint format
            generated_text = result.get("generated_text", "")
        else:
            raise ValueError(f"Unexpected response format: {result}")

        return LLMResponse(
            content=generated_text.strip(),
            model=model,
            provider=self.name,
            usage={
                "prompt_tokens": -1,  # Not provided by HF API
                "completion_tokens": -1,
                "total_tokens": -1,
            },
            raw_response={"endpoint": self._endpoint_url},
        )

    def _build_prompt(self, messages: list[Message] | list[dict]) -> str:
        """Build prompt from messages using chat template.

        Detects model type and uses appropriate format.
        Supports Llama-3, Mistral-Instruct, and generic formats.
        Accepts both Message objects and dicts with role/content keys.
        """
        # Detect template based on model name
        model_lower = self._model.lower()

        if "mistral" in model_lower or "mixtral" in model_lower:
            return self._build_mistral_prompt(messages)
        elif "llama" in model_lower:
            return self._build_llama_prompt(messages)
        else:
            # Generic ChatML format
            return self._build_chatml_prompt(messages)

    def _extract_message_parts(self, message: Message | dict) -> tuple[str, str]:
        """Extract role and content from message."""
        if isinstance(message, dict):
            return message.get("role", ""), message.get("content", "")
        return message.role, message.content

    def _build_mistral_prompt(self, messages: list[Message] | list[dict]) -> str:
        """Build Mistral-Instruct format prompt."""
        prompt_parts = []

        for message in messages:
            role, content = self._extract_message_parts(message)

            if role == "system":
                # Mistral doesn't have explicit system, prepend to first user message
                prompt_parts.append(f"[INST] {content} [/INST]")
            elif role == "user":
                prompt_parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                prompt_parts.append(content)

        return " ".join(prompt_parts)

    def _build_llama_prompt(self, messages: list[Message] | list[dict]) -> str:
        """Build Llama-3-style prompt."""
        prompt_parts = []

        for message in messages:
            role, content = self._extract_message_parts(message)

            if role == "system":
                prompt_parts.append(
                    f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
                )
            elif role == "user":
                prompt_parts.append(
                    f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
                )
            elif role == "assistant":
                prompt_parts.append(
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
                )

        # Add assistant header to trigger response
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

        return "".join(prompt_parts)

    def _build_chatml_prompt(self, messages: list[Message] | list[dict]) -> str:
        """Build generic ChatML format prompt."""
        prompt_parts = []

        for message in messages:
            role, content = self._extract_message_parts(message)
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)

    async def aclose(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            import asyncio

            asyncio.run(self.aclose())
        except Exception:
            pass


# =============================================================================
# FACTORY HELPER
# =============================================================================


def create_huggingface_provider(
    model: str = "meta-llama/Llama-3.1-70B-Instruct",
    **kwargs: Any,
) -> HuggingFaceProvider:
    """Create HuggingFace provider with common defaults.

    Args:
        model: HF model identifier
        **kwargs: Additional provider arguments

    Returns:
        Configured HuggingFaceProvider
    """
    return HuggingFaceProvider(model=model, **kwargs)


# =============================================================================
# RECOMMENDED MODELS
# =============================================================================

# =============================================================================
# RECOMMENDED MODELS - All verified FREE on HuggingFace Router API
# =============================================================================
# Based on MedeX configurations - all tested and working with HF_TOKEN
# Source: MedeX/run_api.py MODEL_MAPPING (2026-01-14)

RECOMMENDED_MODELS = {
    # === FREE HIGH-CAPACITY MODELS (verified working) ===
    "default": "Qwen/Qwen2.5-72B-Instruct",  # FREE, best quality ~50s
    "fast": "google/gemma-3-27b-it",  # FREE, fastest ~20s
    "large": "meta-llama/Llama-3.3-70B-Instruct",  # FREE, balanced ~32s
    "medical": "Qwen/Qwen2.5-72B-Instruct",  # FREE, excellent for medical
    "coding": "meta-llama/Llama-3.3-70B-Instruct",  # FREE, good for code
    # === REASONING MODELS (with <think> tags) ===
    "reasoning": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # FREE, chain-of-thought
    "qwq": "Qwen/QwQ-32B",  # FREE, reasoning with <think> tags
    # === ALTERNATIVE ALIASES ===
    "llama70b": "meta-llama/Llama-3.3-70B-Instruct",  # FREE
    "qwen72b": "Qwen/Qwen2.5-72B-Instruct",  # FREE
    "gemma": "google/gemma-3-27b-it",  # FREE
}
