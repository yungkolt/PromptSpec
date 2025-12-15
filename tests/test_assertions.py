"""Tests for assertion engine."""

import pytest
from promptspec.assertions import AssertionEngine
from promptspec.gateway import LLMGateway


@pytest.fixture
def gateway():
    """Create a gateway instance for testing."""
    return LLMGateway()


@pytest.fixture
def engine(gateway):
    """Create an assertion engine instance."""
    return AssertionEngine(gateway)


@pytest.mark.asyncio
async def test_assert_contains_pass(engine):
    """Test contains assertion that should pass."""
    passed, error = await engine.run_assertion(
        {"type": "contains", "value": "hello"},
        "Hello, world!",
        100.0,
    )
    assert passed
    assert error is None


@pytest.mark.asyncio
async def test_assert_contains_fail(engine):
    """Test contains assertion that should fail."""
    passed, error = await engine.run_assertion(
        {"type": "contains", "value": "goodbye"},
        "Hello, world!",
        100.0,
    )
    assert not passed
    assert error is not None


@pytest.mark.asyncio
async def test_assert_regex_pass(engine):
    """Test regex assertion that should pass."""
    passed, error = await engine.run_assertion(
        {"type": "regex", "pattern": r"\d+"},
        "The number is 123",
        100.0,
    )
    assert passed
    assert error is None


@pytest.mark.asyncio
async def test_assert_regex_fail(engine):
    """Test regex assertion that should fail."""
    passed, error = await engine.run_assertion(
        {"type": "regex", "pattern": r"\d{5}"},
        "The number is 123",
        100.0,
    )
    assert not passed
    assert error is not None


@pytest.mark.asyncio
async def test_assert_json_valid_pass(engine):
    """Test JSON validation that should pass."""
    passed, error = await engine.run_assertion(
        {"type": "json_valid"},
        '{"name": "test", "value": 123}',
        100.0,
    )
    assert passed
    assert error is None


@pytest.mark.asyncio
async def test_assert_json_valid_fail(engine):
    """Test JSON validation that should fail."""
    passed, error = await engine.run_assertion(
        {"type": "json_valid"},
        '{"name": "test", "value": 123',
        100.0,
    )
    assert not passed
    assert error is not None


@pytest.mark.asyncio
async def test_assert_latency_pass(engine):
    """Test latency assertion that should pass."""
    passed, error = await engine.run_assertion(
        {"type": "latency", "threshold_ms": 500},
        "Some output",
        300.0,
    )
    assert passed
    assert error is None


@pytest.mark.asyncio
async def test_assert_latency_fail(engine):
    """Test latency assertion that should fail."""
    passed, error = await engine.run_assertion(
        {"type": "latency", "threshold_ms": 500},
        "Some output",
        600.0,
    )
    assert not passed
    assert error is not None


@pytest.mark.asyncio
async def test_assert_unknown_type(engine):
    """Test unknown assertion type."""
    passed, error = await engine.run_assertion(
        {"type": "unknown_type"},
        "Some output",
        100.0,
    )
    assert not passed
    assert "Unknown assertion type" in error


@pytest.mark.asyncio
async def test_assert_no_pii_pass(engine):
    """Test no_pii assertion that should pass (no PII present)."""
    passed, error = await engine.run_assertion(
        {"type": "no_pii"},
        "Hello, this is a generic response with no personal information.",
        100.0,
    )
    assert passed
    assert error is None


@pytest.mark.asyncio
async def test_assert_no_pii_fail_email(engine):
    """Test no_pii assertion that should fail (email detected)."""
    passed, error = await engine.run_assertion(
        {"type": "no_pii"},
        "Contact me at john.doe@example.com for more info.",
        100.0,
    )
    assert not passed
    assert "PII detected" in error
    assert "email" in error


@pytest.mark.asyncio
async def test_assert_no_pii_fail_phone(engine):
    """Test no_pii assertion that should fail (phone detected)."""
    passed, error = await engine.run_assertion(
        {"type": "no_pii"},
        "Call me at 555-123-4567 for support.",
        100.0,
    )
    assert not passed
    assert "PII detected" in error
    assert "phone" in error


@pytest.mark.asyncio
async def test_assert_no_pii_fail_ssn(engine):
    """Test no_pii assertion that should fail (SSN detected)."""
    passed, error = await engine.run_assertion(
        {"type": "no_pii"},
        "My SSN is 123-45-6789.",
        100.0,
    )
    assert not passed
    assert "PII detected" in error
    assert "ssn" in error

