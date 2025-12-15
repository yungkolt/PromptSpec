"""Utility functions for rate limiting and concurrency management."""

import asyncio
from typing import Callable, Any
from collections import defaultdict

from .gateway import RateLimitError


class RateLimiter:
    """Rate limiter with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        backoff_multiplier: float = 2.0,
    ):
        """Initialize rate limiter.

        Args:
            max_retries: Maximum number of retries
            initial_backoff: Initial backoff time in seconds
            max_backoff: Maximum backoff time in seconds
            backoff_multiplier: Multiplier for exponential backoff
        """
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_multiplier = backoff_multiplier
        self._model_retries: dict[str, int] = defaultdict(int)

    async def execute_with_retry(
        self,
        coro: Callable[[], Any],
        model: str,
    ) -> Any:
        """Execute a coroutine with exponential backoff on rate limit errors.

        Args:
            coro: Coroutine function to execute (takes no arguments)
            model: Model identifier (for per-model retry tracking)

        Returns:
            Result of coroutine execution

        Raises:
            RateLimitError: If max retries exceeded
            Exception: Other exceptions from coro
        """
        backoff = self.initial_backoff
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                return await coro()
            except RateLimitError as e:
                retry_count += 1
                self._model_retries[model] = retry_count

                if retry_count > self.max_retries:
                    raise RateLimitError(
                        f"Max retries ({self.max_retries}) exceeded for {model}"
                    ) from e

                # Calculate backoff with jitter
                sleep_time = min(backoff, self.max_backoff)
                # Add small random jitter to avoid thundering herd
                import random
                sleep_time += random.uniform(0, sleep_time * 0.1)

                await asyncio.sleep(sleep_time)
                backoff *= self.backoff_multiplier
            except Exception:
                # For non-rate-limit errors, don't retry
                raise

        raise RateLimitError(f"Unexpected retry loop exit for {model}")


class ConcurrencyManager:
    """Manages concurrent execution with semaphore-based limiting."""

    def __init__(self, max_concurrent: int = 10):
        """Initialize concurrency manager.

        Args:
            max_concurrent: Maximum number of concurrent operations
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent

    async def execute(
        self,
        coro: Callable[[], Any],
    ) -> Any:
        """Execute a coroutine with concurrency limiting.

        Args:
            coro: Coroutine function to execute (takes no arguments)

        Returns:
            Result of coroutine execution
        """
        async with self.semaphore:
            return await coro()

