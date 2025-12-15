# PromptSpec

**Unit test your LLM prompts before deploying.**

PromptSpec is a CLI tool for testing LLM prompts. Write test specs in YAML, run them against local models (Ollama) or cloud APIs (OpenAI, Anthropic, etc.), and get instant feedback on whether your prompts work as expected.

## Why PromptSpec?

- **🔒 Privacy First**: Test prompts locally with Ollama — your data never leaves your machine
- **💰 Cost Efficient**: Develop with local models for free, validate with cloud APIs
- **🚀 CI/CD Ready**: Returns exit code 1 on failure — integrate into your pipeline
- **🌐 Universal**: Supports 100+ models via LiteLLM (Ollama, OpenAI, Anthropic, and more)
- **⚡ Fast**: Run multiple tests concurrently with intelligent rate limiting

## Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3
```

### Installation

```bash
pip install promptspec
```

Or install from source:

```bash
git clone https://github.com/yungkolt/promptspec.git
cd promptspec
pip install -e .
```

### Create a Test Spec

Create `promptspec.yaml`:

```yaml
tests:
  - description: "Test greeting response"
    prompt: "Say hello in one sentence."
    model: "ollama/llama3"
    assertions:
      - type: "contains"
        value: "hello"
      - type: "latency"
        threshold_ms: 30000
```

### Run Tests

```bash
promptspec run promptspec.yaml
```

Output:

```
Running tests from: promptspec.yaml

                              Prompt Eval Results                               
╭────────────────────────┬───────────┬─────────────┬──────────────────────────╮
│ Test Case              │ Status    │     Latency │ Output Preview           │
├────────────────────────┼───────────┼─────────────┼──────────────────────────┤
│ Test greeting response │ ✓ PASS    │      5571ms │ Hello! It's nice to...   │
╰────────────────────────┴───────────┴─────────────┴──────────────────────────╯

╭───────────────────────────── ✓ All Tests Passed ─────────────────────────────╮
│ Total: 1 | Passed: 1 | Failed: 0 | Success Rate: 100.0% | Avg Latency: 5571ms│
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Assertion Types

| Type | Description | Example |
|------|-------------|---------|
| `contains` | Check if output contains substring (case-insensitive) | `value: "hello"` |
| `regex` | Pattern matching with regex | `pattern: "(yes\|no)"` |
| `json_valid` | Validate JSON output | — |
| `latency` | Ensure response time below threshold | `threshold_ms: 5000` |
| `sentiment` | LLM-as-a-Judge evaluation | `condition: "polite"` |
| `no_pii` | Detect PII (emails, phones, SSN, credit cards) | — |

## Examples

### JSON Validation

```yaml
tests:
  - description: "Generate valid JSON"
    prompt: "Return a JSON object with name and age fields."
    model: "ollama/llama3"
    assertions:
      - type: "json_valid"
      - type: "contains"
        value: "name"
```

### Regex Pattern Matching

```yaml
tests:
  - description: "Extract email from text"
    prompt: "Extract email from: 'Contact support@example.com'"
    model: "ollama/llama3"
    assertions:
      - type: "regex"
        pattern: "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
```

### PII Detection

```yaml
tests:
  - description: "Ensure no PII in response"
    prompt: "Summarize this customer feedback without including personal info."
    model: "ollama/llama3"
    assertions:
      - type: "no_pii"  # Fails if emails, phones, SSN, or credit cards detected
```

### Multiple Assertions

```yaml
tests:
  - description: "Explain Python"
    prompt: "Explain what Python is in one sentence."
    model: "ollama/llama3"
    assertions:
      - type: "contains"
        value: "Python"
      - type: "regex"
        pattern: "(programming|language|code)"
      - type: "latency"
        threshold_ms: 30000
```

## CLI Options

```bash
# Run with custom concurrency
promptspec run --max-concurrent 5 promptspec.yaml

# Verbose output (show failed test details)
promptspec run --verbose promptspec.yaml

# Show help
promptspec --help

# Show version
promptspec version
```

## Configuration

### Global Defaults

Apply defaults to all tests:

```yaml
defaults:
  temperature: 0.7

tests:
  - description: "Test 1"
    prompt: "Hello"
    model: "ollama/llama3"
    assertions:
      - type: "contains"
        value: "hello"
```

### Environment Variables

Use environment variables in specs:

```yaml
tests:
  - description: "Test with env var"
    prompt: "Hello"
    model: "${MODEL_NAME}"  # or $MODEL_NAME
    assertions:
      - type: "contains"
        value: "hello"
```

## CI/CD Integration

PromptSpec returns exit code 1 on failure, making it easy to integrate into CI/CD:

```yaml
# .github/workflows/test-prompts.yml
name: Test Prompts

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install promptspec
      - run: promptspec run promptspec.yaml
```

## Architecture

```
┌─────────────┐
│  YAML Spec  │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│   Runner    │────▶│   Gateway    │
│ (Concurrent)│     │  (LiteLLM)   │
└──────┬──────┘     └──────────────┘
       │
       ▼
┌─────────────┐
│ Assertions  │
│   Engine    │
└─────────────┘
```

## Supported Models

PromptSpec uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood, which supports:

| Provider | Model Examples | Environment Variable |
|----------|----------------|---------------------|
| Ollama | `ollama/llama3`, `ollama/mistral` | None (local) |
| OpenAI | `gpt-4`, `gpt-3.5-turbo`, `gpt-4o` | `OPENAI_API_KEY` |
| Anthropic | `claude-3-opus-20240229`, `claude-3-sonnet-20240229` | `ANTHROPIC_API_KEY` |
| Google | `gemini/gemini-pro` | `GEMINI_API_KEY` |
| Mistral | `mistral/mistral-medium` | `MISTRAL_API_KEY` |

And [100+ more providers](https://docs.litellm.ai/docs/providers).

### Testing with Cloud APIs

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Run tests with OpenAI
promptspec run promptspec.yaml
```

```yaml
tests:
  - description: "Test with GPT-4"
    prompt: "Say hello in one sentence."
    model: "gpt-4"
    assertions:
      - type: "contains"
        value: "hello"
      - type: "latency"
        threshold_ms: 5000
```

### Recommended Workflow

1. **Develop locally** with Ollama (free, fast iteration)
2. **Validate** with production APIs before merging
3. **CI/CD** runs both local and API tests

```yaml
# Same test, different models
tests:
  - description: "Test locally"
    prompt: "Explain AI briefly."
    model: "ollama/llama3"
    assertions:
      - type: "contains"
        value: "artificial"

  - description: "Validate with production API"
    prompt: "Explain AI briefly."
    model: "gpt-3.5-turbo"
    assertions:
      - type: "contains"
        value: "artificial"
```

## Development

```bash
# Clone the repository
git clone https://github.com/yungkolt/promptspec.git
cd promptspec

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Roadmap

- [ ] Plugin system for custom assertions
- [ ] Test result export (JSON, HTML reports)
- [ ] Prompt versioning and comparison
- [ ] Cost tracking per test run

---

**Built for AI engineers who value privacy, efficiency, and quality.**
