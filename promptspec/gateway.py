"""Unified LLM gateway using LiteLLM for multi-provider support."""

import asyncio
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

import litellm
from litellm import completion, ModelResponse


@dataclass
class LLMResponse:
    """Structured response from LLM gateway."""

    content: str
    model: str
    latency_ms: float
    raw_response: ModelResponse
    error: Optional[str] = None


class LLMGateway:
    """Unified interface for LLM providers via LiteLLM."""

    def __init__(self, default_judge_model: str = "gpt-3.5-turbo"):
        """Initialize the gateway.

        Args:
            default_judge_model: Default model to use for LLM-as-a-Judge assertions
        """
        self.default_judge_model = default_judge_model
        # Configure LiteLLM settings
        litellm.drop_params = True  # Drop unsupported parameters
        litellm.suppress_debug_info = True  # Reduce noise in logs

    def normalize_model_name(self, model: str) -> str:
        """Normalize model name for LiteLLM compatibility.

        Args:
            model: Model identifier (e.g., "ollama/llama3", "gpt-3.5-turbo")

        Returns:
            Normalized model name for LiteLLM
        """
        # LiteLLM handles ollama/ prefix automatically
        # Just ensure it's in the right format
        if model.startswith("ollama/"):
            return model
        # For other providers, LiteLLM handles them natively
        return model

    async def call(
        self,
        model: str,
        messages: list[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Call an LLM model asynchronously.

        Args:
            model: Model identifier
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters passed to LiteLLM

        Returns:
            LLMResponse with content, latency, and metadata

        Raises:
            Exception: If the API call fails after retries
        """
        normalized_model = self.normalize_model_name(model)
        start_time = time.time()

        try:
            # LiteLLM's completion is synchronous, so we run it in a thread
            # to avoid blocking the event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: completion(
                    model=normalized_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract content from response
            content = ""
            if hasattr(response, "choices") and len(response.choices) > 0:
                if hasattr(response.choices[0], "message"):
                    content = response.choices[0].message.content or ""
                elif hasattr(response.choices[0], "text"):
                    content = response.choices[0].text or ""

            return LLMResponse(
                content=content,
                model=normalized_model,
                latency_ms=latency_ms,
                raw_response=response,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)

            # Check for rate limit errors
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                raise RateLimitError(f"Rate limit exceeded for {normalized_model}: {error_msg}") from e

            return LLMResponse(
                content="",
                model=normalized_model,
                latency_ms=latency_ms,
                raw_response=None,  # type: ignore
                error=error_msg,
            )

    async def call_judge(
        self,
        prompt: str,
        judge_model: Optional[str] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Call the judge model for LLM-as-a-Judge assertions.

        Args:
            prompt: The prompt to send to the judge
            judge_model: Optional model override (defaults to self.default_judge_model)
            **kwargs: Additional parameters

        Returns:
            LLMResponse from the judge model
        """
        model = judge_model or self.default_judge_model
        messages = [{"role": "user", "content": prompt}]
        return await self.call(model=model, messages=messages, temperature=0.0, **kwargs)


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""

    pass

