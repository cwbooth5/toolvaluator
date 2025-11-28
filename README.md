# Toolvaluator

An MCP tool schema evaluation framework for testing LLM tool-calling capabilities.

Toolvaluator helps you evaluate how well language models use your FastMCP tools by measuring:
- **Correctness**: Does the model choose the right tool with the right arguments?
- **Latency**: How fast does the model make tool-calling decisions?

You can use this to:
- Compare different models to find which works best with your tools
- Optimize your tool schemas and descriptions for better model performance
- Establish quality benchmarks for your MCP tools

# Design

I made this to fill out a little gap in my toolset where I would create all these
MCP servers but wouldn't figure out what to do when models stumbled around
my tool definintions. The tool name and organization along with the signature
and docstring is crucial to making sure the models just figure out my tools.
In practice, I've seen wildly different behavior from model to model. The accuracy
of the model's decisions is quite important. The latency is a secondary concern
of mine because there are some situations when I need to string together a bunch
of tool calls and want to see how many I can cram into a unit of time.

This doesn't get the model to call the tool. We are only measuring the model's decision.
It's reading the MCP tool definition in your tool and using that with the model,
just like when your AI client registers your tools with the model.

## Installation

### Using uv (recommended)

```bash
# Install the package
uv pip install toolvaluator

# Or install from source with dev dependencies
git clone https://github.com/cwbooth5/toolvaluator.git
cd toolvaluator
uv pip install -e ".[dev]"

or for an editable install...

uv tool install -e .
```

### Tool install

You can install the tool straight out of the github repo.

```bash
uv tool install --from https://github.com/cwbooth5/toolvaluator.git toolvaluator
```

### Using pip

```bash
pip install toolvaluator

# Or install from source with dev dependencies
git clone https://github.com/cwbooth5/toolvaluator.git
cd toolvaluator
pip install -e ".[dev]"
```

## Quick Start

### 1. Create Your FastMCP Server

First, create a FastMCP server with your tools (e.g., `my_server.py`):

```python
from fastmcp import FastMCP

mcp = FastMCP("My Server")

@mcp.tool()
def search_docs(query: str) -> str:
    """Search through company documentation."""
    return f"Found documents matching '{query}'"

if __name__ == "__main__":
    mcp.run()
```

### 2. Run the Evaluation

```bash
# Evaluate with OpenAI GPT-4o-mini (default)
toolvaluator --server my_server

# Use a different model
toolvaluator --server my_server --model gpt-4o

# Use a local/OSS model
toolvaluator --server my_server --model llama-3.1 --base-url http://localhost:1234/v1 --api-key none

# Set a minimum score threshold (useful for CI/CD)
toolvaluator --server my_server --min-score 0.85
```

### 3. Customize Your Evaluation Dataset

To create custom evaluation examples, you can use toolvaluator as a library:

```python
from toolvaluator import build_dataset, eval_model, get_tool_schemas_sync
from my_server import mcp

# Fetch tool schemas
tool_schemas = get_tool_schemas_sync(mcp)

# Customize the dataset (you can override build_dataset)
# See src/toolvaluator/evaluator.py for the default implementation

# Run evaluation
result = eval_model(
    model_name="gpt-4o-mini",
    api_key="your-api-key",
    base_url=None,
    dataset=dataset,
)

print(f"Score: {result['score']:.3f}")
```

## Project Structure

```
toolvaluator/
├── src/
│   └── toolvaluator/
│       ├── __init__.py          # Package initialization with version
│       ├── cli.py               # Command-line interface
│       ├── evaluator.py         # Core evaluation logic
│       └── test_server.py       # Example MCP server for testing
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures
│   ├── test_evaluator.py        # Tests for evaluator module
│   └── test_server.py           # Tests for test server
├── pyproject.toml               # Project configuration
└── README.md
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/cwbooth5/toolvaluator.git
cd toolvaluator

# Install with dev dependencies using uv
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=toolvaluator --cov-report=html

# Run specific test file
pytest tests/test_evaluator.py
```

### Code Quality

```bash
# Format code with black
black src tests

# Lint with ruff
ruff check src tests

# Fix auto-fixable issues
ruff check --fix src tests
```

### Building

```bash
# Build the package
uv build

# Or using hatch
hatch build

# This creates dist/toolvaluator-*.whl and dist/toolvaluator-*.tar.gz
```

### Version Management

The version is managed by Hatch and stored in `src/toolvaluator/__init__.py`.

To update the version:

```python
# Edit src/toolvaluator/__init__.py
__version__ = "0.2.0"
```

Then rebuild:

```bash
uv build
```

## Example: Using the Test Server

The package includes a test MCP server with example tools:

```bash
# Run evaluation against the test server
toolvaluator --server toolvaluator.test_server

# Or use it programmatically
python -c "from toolvaluator.test_server import mcp; print(mcp)"
```

The test server includes these tools:
- `search_docs(query)` - Search documentation
- `get_weather(location, units)` - Get weather information
- `calculate(operation, a, b)` - Perform calculations
- `send_email(to, subject, body, cc)` - Send emails
- `create_task(title, description, priority, due_date)` - Create tasks

## How It Works

1. **Schema Extraction**: Fetches tool schemas from your FastMCP server using the MCP protocol
2. **Dataset Creation**: Creates test examples with user queries and expected tool-calling behavior
3. **LLM Evaluation**: Uses DSPy to prompt the model to decide:
   - Should the tool be called?
   - Which tool should be called?
   - What arguments should be passed?
4. **Scoring**: Compares the model's decisions against expected behavior across three dimensions:
   - Should-call correctness (binary: did it correctly decide to use/not use the tool?)
   - Tool name correctness (binary: did it pick the right tool?)
   - Argument correctness (fraction: how many arguments matched?)
5. **Metrics**: Reports overall correctness score (0-1) and latency statistics (mean, p50, p95, max)

## Configuration Options

### CLI Options

- `--model`: Model name (default: `gpt-4o-mini`)
- `--api-key`: API key (defaults to `OPENAI_API_KEY` env var)
- `--base-url`: Base URL for OpenAI-compatible endpoints
- `--min-score`: Minimum acceptable score (exits with code 1 if below threshold)
- `--server`: Python module containing your FastMCP server (default: `server`)

### Environment Variables

- `OPENAI_API_KEY`: API key for OpenAI models

## Use Cases

### 1. Model Comparison

Test which model works best with your tools:

```bash
toolvaluator --server my_server --model gpt-4o
toolvaluator --server my_server --model gpt-4o-mini
toolvaluator --server my_server --model claude-3-5-sonnet
```

NOTE: If you're using a `base_url` to point to a custom hosted model,
we assume an openai provider or openai-style API is being provided.
This happens to be what LM studio and Ollama provide.

### 2. Tool Schema Optimization

Iterate on tool descriptions and schemas to improve model accuracy:

```python
# Before: vague description
@mcp.tool()
def search(q: str) -> str:
    """Search stuff."""
    ...

# After: clear, specific description
@mcp.tool()
def search_docs(query: str) -> str:
    """
    Search through company documentation including policies,
    procedures, and internal wikis.

    Args:
        query: Search keywords or natural language question
    """
    ...
```

### 3. CI/CD Quality Gates

Add to your CI pipeline:

```bash
# Fail the build if tool-calling accuracy drops below 85%
toolvaluator --server my_server --min-score 0.85
```

### 4. Latency Benchmarking

Track tool-calling latency across model versions or configurations.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Credits

Built with:
- [FastMCP](https://github.com/jlowin/fastmcp) - Model Context Protocol framework
- [DSPy](https://github.com/stanfordnlp/dspy) - Framework for LLM evaluation
- [Hatch](https://hatch.pypa.io/) - Build backend
- [Ruff](https://github.com/astral-sh/ruff) - Fast Python linter
- [Black](https://github.com/psf/black) - Code formatter
- [pytest](https://pytest.org/) - Testing framework
