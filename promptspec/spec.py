"""YAML spec parser and validator."""

import os
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

import yaml


class SpecError(Exception):
    """Raised when spec validation fails."""

    pass


class SpecParser:
    """Parser for PromptSpec YAML configuration files."""

    def __init__(self, spec_path: str):
        """Initialize the spec parser.

        Args:
            spec_path: Path to the YAML spec file
        """
        self.spec_path = Path(spec_path)
        if not self.spec_path.exists():
            raise SpecError(f"Spec file not found: {spec_path}")

    def parse(self) -> Dict[str, Any]:
        """Parse and validate the spec file.

        Returns:
            Parsed and validated spec dictionary

        Raises:
            SpecError: If parsing or validation fails
        """
        try:
            with open(self.spec_path, "r") as f:
                spec = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise SpecError(f"Invalid YAML: {str(e)}") from e

        if not spec:
            raise SpecError("Spec file is empty")

        # Validate structure
        if "tests" not in spec:
            raise SpecError("Spec must contain a 'tests' key")

        if not isinstance(spec["tests"], list):
            raise SpecError("'tests' must be a list")

        if len(spec["tests"]) == 0:
            raise SpecError("'tests' list cannot be empty")

        # Process each test
        processed_tests = []
        for i, test in enumerate(spec["tests"]):
            try:
                processed_test = self._process_test(test, i)
                processed_tests.append(processed_test)
            except Exception as e:
                raise SpecError(f"Error processing test {i + 1}: {str(e)}") from e

        spec["tests"] = processed_tests

        # Apply global defaults if specified
        if "defaults" in spec:
            defaults = spec["defaults"]
            for test in spec["tests"]:
                if "model" not in test and "model" in defaults:
                    test["model"] = defaults["model"]
                if "temperature" not in test and "temperature" in defaults:
                    test["temperature"] = defaults["temperature"]

        return spec

    def _process_test(self, test: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Process and validate a single test case.

        Args:
            test: Test case dictionary
            index: Test index (for error messages)

        Returns:
            Processed test case with defaults applied

        Raises:
            SpecError: If validation fails
        """
        # Required fields
        if "description" not in test:
            raise SpecError(f"Test {index + 1} missing required field: 'description'")
        if "prompt" not in test:
            raise SpecError(f"Test {index + 1} missing required field: 'prompt'")
        if "model" not in test:
            raise SpecError(f"Test {index + 1} missing required field: 'model'")
        if "assertions" not in test:
            raise SpecError(f"Test {index + 1} missing required field: 'assertions'")

        # Validate assertions
        if not isinstance(test["assertions"], list):
            raise SpecError(f"Test {index + 1}: 'assertions' must be a list")

        if len(test["assertions"]) == 0:
            raise SpecError(f"Test {index + 1}: 'assertions' list cannot be empty")

        # Process environment variables in model name
        model = self._expand_env_vars(str(test["model"]))
        test["model"] = model

        # Process environment variables in prompt
        prompt = self._expand_env_vars(str(test["prompt"]))
        test["prompt"] = prompt

        # Set defaults
        if "temperature" not in test:
            test["temperature"] = 0.7

        # Validate each assertion
        for j, assertion in enumerate(test["assertions"]):
            if not isinstance(assertion, dict):
                raise SpecError(
                    f"Test {index + 1}, assertion {j + 1}: must be a dictionary"
                )
            if "type" not in assertion:
                raise SpecError(
                    f"Test {index + 1}, assertion {j + 1}: missing required field 'type'"
                )

        return test

    def _expand_env_vars(self, text: str) -> str:
        """Expand environment variables in text.

        Supports ${VAR} and $VAR syntax.

        Args:
            text: Text potentially containing environment variables

        Returns:
            Text with environment variables expanded
        """
        # Handle ${VAR} syntax
        def replace_env(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))  # Return original if not found

        text = re.sub(r"\$\{([^}]+)\}", replace_env, text)

        # Handle $VAR syntax (but not $$)
        def replace_simple_env(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))

        text = re.sub(r"\$([A-Z_][A-Z0-9_]*)", replace_simple_env, text)

        return text

