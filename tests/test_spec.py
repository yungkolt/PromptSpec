"""Tests for spec parser."""

import pytest
import tempfile
import os
from pathlib import Path
from promptspec.spec import SpecParser, SpecError


def test_parse_valid_spec():
    """Test parsing a valid spec file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
tests:
  - description: "Test 1"
    prompt: "Hello"
    model: "gpt-3.5-turbo"
    assertions:
      - type: "contains"
        value: "hello"
""")
        spec_path = f.name

    try:
        parser = SpecParser(spec_path)
        spec = parser.parse()
        assert "tests" in spec
        assert len(spec["tests"]) == 1
        assert spec["tests"][0]["description"] == "Test 1"
    finally:
        os.unlink(spec_path)


def test_parse_missing_tests():
    """Test parsing a spec without tests key."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("defaults:\n  temperature: 0.7\n")
        spec_path = f.name

    try:
        parser = SpecParser(spec_path)
        with pytest.raises(SpecError, match="must contain a 'tests' key"):
            parser.parse()
    finally:
        os.unlink(spec_path)


def test_parse_empty_tests():
    """Test parsing a spec with empty tests list."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("tests: []\n")
        spec_path = f.name

    try:
        parser = SpecParser(spec_path)
        with pytest.raises(SpecError, match="cannot be empty"):
            parser.parse()
    finally:
        os.unlink(spec_path)


def test_parse_env_vars():
    """Test parsing spec with environment variables."""
    os.environ["TEST_MODEL"] = "gpt-4"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
tests:
  - description: "Test with env var"
    prompt: "Hello"
    model: "${TEST_MODEL}"
    assertions:
      - type: "contains"
        value: "hello"
""")
        spec_path = f.name

    try:
        parser = SpecParser(spec_path)
        spec = parser.parse()
        assert spec["tests"][0]["model"] == "gpt-4"
    finally:
        os.unlink(spec_path)
        if "TEST_MODEL" in os.environ:
            del os.environ["TEST_MODEL"]


def test_parse_missing_required_fields():
    """Test parsing a spec with missing required fields."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
tests:
  - description: "Test without prompt"
    model: "gpt-3.5-turbo"
    assertions:
      - type: "contains"
        value: "hello"
""")
        spec_path = f.name

    try:
        parser = SpecParser(spec_path)
        with pytest.raises(SpecError, match="missing required field"):
            parser.parse()
    finally:
        os.unlink(spec_path)

