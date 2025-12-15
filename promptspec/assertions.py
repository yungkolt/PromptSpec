"""Assertion engine for validating LLM outputs."""

import re
import json
from typing import Any, Dict, Optional

from .gateway import LLMGateway, LLMResponse


class AssertionError(Exception):
    """Raised when an assertion fails."""

    def __init__(self, assertion_type: str, message: str):
        self.assertion_type = assertion_type
        self.message = message
        super().__init__(f"[{assertion_type}] {message}")


class AssertionEngine:
    """Engine for running assertions on LLM outputs."""

    def __init__(self, gateway: LLMGateway):
        """Initialize the assertion engine.

        Args:
            gateway: LLM gateway instance for judge model calls
        """
        self.gateway = gateway
        self._judge_cache: Dict[str, bool] = {}

    async def run_assertion(
        self,
        assertion: Dict[str, Any],
        output: str,
        latency_ms: float,
        **context: Any
    ) -> tuple[bool, Optional[str]]:
        """Run a single assertion on the output.

        Args:
            assertion: Assertion configuration dict with 'type' and other fields
            output: The LLM output to test
            latency_ms: Response latency in milliseconds
            **context: Additional context (e.g., original prompt)

        Returns:
            Tuple of (passed: bool, error_message: Optional[str])
        """
        assertion_type = assertion.get("type", "").lower()

        try:
            if assertion_type == "contains":
                return self._assert_contains(output, assertion.get("value", ""))
            elif assertion_type == "regex":
                return self._assert_regex(output, assertion.get("pattern", ""))
            elif assertion_type == "json_valid":
                return self._assert_json_valid(output)
            elif assertion_type == "latency":
                threshold = assertion.get("threshold_ms", 0)
                return self._assert_latency(latency_ms, threshold)
            elif assertion_type == "sentiment":
                condition = assertion.get("condition", "positive")
                return await self._assert_sentiment(output, condition)
            elif assertion_type == "no_pii":
                return self._assert_no_pii(output)
            else:
                return (
                    False,
                    f"Unknown assertion type: {assertion_type}",
                )

        except AssertionError as e:
            return (False, str(e))
        except Exception as e:
            return (False, f"Assertion error: {str(e)}")

    def _assert_contains(self, output: str, value: str) -> tuple[bool, Optional[str]]:
        """Assert that output contains a substring (case-insensitive).

        Args:
            output: The LLM output
            value: The substring to search for

        Returns:
            Tuple of (passed, error_message)
        """
        if not value:
            return (False, "contains assertion requires 'value' field")

        if value.lower() in output.lower():
            return (True, None)
        else:
            return (False, f"Output does not contain '{value}'")

    def _assert_regex(self, output: str, pattern: str) -> tuple[bool, Optional[str]]:
        """Assert that output matches a regex pattern.

        Args:
            output: The LLM output
            pattern: The regex pattern to match

        Returns:
            Tuple of (passed, error_message)
        """
        if not pattern:
            return (False, "regex assertion requires 'pattern' field")

        try:
            if re.search(pattern, output, re.IGNORECASE | re.MULTILINE):
                return (True, None)
            else:
                return (False, f"Output does not match pattern '{pattern}'")
        except re.error as e:
            return (False, f"Invalid regex pattern: {str(e)}")

    def _assert_json_valid(self, output: str) -> tuple[bool, Optional[str]]:
        """Assert that output is valid JSON.

        Args:
            output: The LLM output

        Returns:
            Tuple of (passed, error_message)
        """
        try:
            json.loads(output)
            return (True, None)
        except json.JSONDecodeError as e:
            return (False, f"Output is not valid JSON: {str(e)}")

    def _assert_latency(
        self, latency_ms: float, threshold_ms: float
    ) -> tuple[bool, Optional[str]]:
        """Assert that latency is below threshold.

        Args:
            latency_ms: Actual latency in milliseconds
            threshold_ms: Maximum allowed latency

        Returns:
            Tuple of (passed, error_message)
        """
        if latency_ms <= threshold_ms:
            return (True, None)
        else:
            return (
                False,
                f"Latency {latency_ms:.1f}ms exceeds threshold {threshold_ms}ms",
            )

    async def _assert_sentiment(
        self, output: str, condition: str
    ) -> tuple[bool, Optional[str]]:
        """Assert sentiment using LLM-as-a-Judge.

        Args:
            output: The LLM output to evaluate
            condition: Expected sentiment (e.g., "positive", "negative", "neutral", "polite")

        Returns:
            Tuple of (passed, error_message)
        """
        # Create cache key
        cache_key = f"{output[:100]}:{condition}"

        # Check cache
        if cache_key in self._judge_cache:
            passed = self._judge_cache[cache_key]
            if passed:
                return (True, None)
            else:
                return (False, f"Sentiment is not {condition}")

        # Build judge prompt
        judge_prompt = f"""Analyze the following text and determine if it has a {condition} tone.

Text: "{output}"

Does this text have a {condition} tone? Reply ONLY with 'YES' or 'NO'."""

        try:
            # Call judge model
            judge_response = await self.gateway.call_judge(judge_prompt)

            if judge_response.error:
                return (
                    False,
                    f"Judge model error: {judge_response.error}",
                )

            # Parse response
            judge_output = judge_response.content.strip().upper()
            passed = "YES" in judge_output

            # Cache result
            self._judge_cache[cache_key] = passed

            if passed:
                return (True, None)
            else:
                return (False, f"Sentiment is not {condition} (judge: {judge_output})")

        except Exception as e:
            return (False, f"Failed to evaluate sentiment: {str(e)}")

    def _assert_no_pii(self, output: str) -> tuple[bool, Optional[str]]:
        """Assert that output does not contain PII (Personally Identifiable Information).

        Checks for common PII patterns:
        - Email addresses
        - Phone numbers (US format)
        - Social Security Numbers
        - Credit card numbers

        Args:
            output: The LLM output to check

        Returns:
            Tuple of (passed, error_message)
        """
        pii_patterns = {
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        }

        detected_pii = []
        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, output):
                detected_pii.append(pii_type)

        if detected_pii:
            return (False, f"PII detected: {', '.join(detected_pii)}")
        return (True, None)

