"""Test runner for executing prompt tests concurrently."""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .gateway import LLMGateway
from .assertions import AssertionEngine
from .spec import SpecParser
from .utils import RateLimiter, ConcurrencyManager


@dataclass
class TestResult:
    """Result of a single test execution."""

    description: str
    passed: bool
    output: str
    latency_ms: float
    assertion_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class RunResults:
    """Aggregated results from a test run."""

    results: List[TestResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100

    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return self.failed == 0


class TestRunner:
    """Runner for executing prompt tests."""

    def __init__(
        self,
        gateway: Optional[LLMGateway] = None,
        max_concurrent: int = 10,
        max_retries: int = 3,
    ):
        """Initialize the test runner.

        Args:
            gateway: LLM gateway instance (creates new one if not provided)
            max_concurrent: Maximum concurrent test executions
            max_retries: Maximum retries for rate-limited requests
        """
        self.gateway = gateway or LLMGateway()
        self.assertion_engine = AssertionEngine(self.gateway)
        self.rate_limiter = RateLimiter(max_retries=max_retries)
        self.concurrency_manager = ConcurrencyManager(max_concurrent=max_concurrent)

    async def run_test(self, test: Dict[str, Any]) -> TestResult:
        """Run a single test case.

        Args:
            test: Test case dictionary from spec

        Returns:
            TestResult with execution results
        """
        description = test["description"]
        prompt = test["prompt"]
        model = test["model"]
        temperature = test.get("temperature", 0.7)
        assertions = test["assertions"]

        # Prepare messages
        messages = [{"role": "user", "content": prompt}]

        # Execute LLM call with rate limiting and concurrency control
        try:
            async def call_llm():
                return await self.gateway.call(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )

            # Wrap in concurrency manager and rate limiter
            async def execute_with_limits():
                return await self.concurrency_manager.execute(call_llm)

            # Execute with rate limiting
            response = await self.rate_limiter.execute_with_retry(
                execute_with_limits, model
            )

            if response.error:
                return TestResult(
                    description=description,
                    passed=False,
                    output="",
                    latency_ms=response.latency_ms,
                    error=response.error,
                )

            output = response.content
            latency_ms = response.latency_ms

            # Run assertions
            assertion_results = []
            all_passed = True

            for assertion in assertions:
                passed, error_msg = await self.assertion_engine.run_assertion(
                    assertion=assertion,
                    output=output,
                    latency_ms=latency_ms,
                    prompt=prompt,
                )

                assertion_results.append(
                    {
                        "type": assertion["type"],
                        "passed": passed,
                        "error": error_msg,
                    }
                )

                if not passed:
                    all_passed = False

            return TestResult(
                description=description,
                passed=all_passed,
                output=output,
                latency_ms=latency_ms,
                assertion_results=assertion_results,
            )

        except Exception as e:
            return TestResult(
                description=description,
                passed=False,
                output="",
                latency_ms=0.0,
                error=f"Test execution failed: {str(e)}",
            )

    async def run_spec(self, spec_path: str) -> RunResults:
        """Run all tests from a spec file.

        Args:
            spec_path: Path to the YAML spec file

        Returns:
            RunResults with aggregated test results
        """
        # Parse spec
        parser = SpecParser(spec_path)
        spec = parser.parse()
        tests = spec["tests"]

        # Create tasks for all tests
        tasks = [self.run_test(test) for test in tests]

        # Execute all tests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        test_results: List[TestResult] = []
        for result in results:
            if isinstance(result, Exception):
                # Create error result
                test_results.append(
                    TestResult(
                        description="Unknown",
                        passed=False,
                        output="",
                        latency_ms=0.0,
                        error=f"Unexpected error: {str(result)}",
                    )
                )
            else:
                test_results.append(result)

        # Aggregate statistics
        run_results = RunResults(results=test_results)
        run_results.total = len(test_results)
        run_results.passed = sum(1 for r in test_results if r.passed)
        run_results.failed = run_results.total - run_results.passed
        run_results.total_latency_ms = sum(r.latency_ms for r in test_results)

        return run_results

