# Toolvaluator

An MCP tool schema evaluation framework for testing LLM tool-calling capabilities.

Toolvaluator helps you evaluate how well language models use your FastMCP tools by measuring:
- **Correctness**: Does the model choose the right tool with the right arguments?
- **Latency**: How fast does the model make tool-calling decisions?

You can use this to:
- Compare different models to find which works best with your tools
- Optimize your tool schemas and descriptions for better model performance
- Establish quality benchmarks for your MCP tools
- Use simple synthetic tool definitions to see how good a model is at tool-calling

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

### 3. Create Custom Evaluation Scripts (Recommended)

For testing your own MCP tools, use the **kickstart tool** to generate a standalone evaluation script:

```bash
# Generate a custom evaluation script for your MCP server
toolvaluator-init \
  --server my_server \
  --server-var mcp \
  --output eval_my_tools.py

# This auto-detects your tools and creates a template script
# Edit eval_my_tools.py to customize the evaluation examples

# Run your custom evaluation
python eval_my_tools.py --model gpt-4o-mini --verbose

# Use in CI/CD with quality gates
python eval_my_tools.py --model gpt-4o --min-score 0.85
```

**What is a "dataset"?** The dataset is a collection of **evaluation examples** - test cases that verify if the model can:
1. Decide when to call a tool (`should_call`)
2. Choose the right tool (`tool_name`)
3. Extract correct arguments (`arguments`)

**Example evaluation example:**
```python
dspy.Example(
    user_query="Find the company vacation policy",
    tool_name="search_docs",
    tool_description="Search through company documentation",
    tool_schema=schema,
    expected_should_call=True,
    expected_tool_name="search_docs",
    expected_arguments={"query": "vacation policy"},
).with_inputs("user_query", "tool_name", "tool_description", "tool_schema")
```

Each example is scored 0-1 across these three dimensions, and the overall score is the average.

## Using Toolvaluator as a Library

You can also use toolvaluator programmatically in your own scripts:

```python
from toolvaluator import (
    build_dataset,
    eval_model,
    get_tool_schemas_sync,
    extract_input_schema,
    extract_tool_description
)
import dspy
from my_server import mcp

# Fetch tool schemas
tool_schemas = get_tool_schemas_sync(mcp)

# Create custom evaluation examples
examples = []
if "my_tool" in tool_schemas:
    tool_name = "my_tool"
    tool_description = extract_tool_description(tool_schemas[tool_name])
    tool_schema = extract_input_schema(tool_schemas[tool_name])

    examples.append(
        dspy.Example(
            user_query="Use my tool to process ABC",
            tool_name=tool_name,
            tool_description=tool_description,
            tool_schema=tool_schema,
            expected_should_call=True,
            expected_tool_name=tool_name,
            expected_arguments={"input": "ABC"},
        ).with_inputs("user_query", "tool_name", "tool_description", "tool_schema")
    )

# Run evaluation
result = eval_model(
    model_name="gpt-4o-mini",
    api_key="your-api-key",
    base_url=None,
    dataset=examples,
    verbose=True,
)

print(f"Score: {result['score']:.3f}")
```

### Reducing Boilerplate with ExampleBuilder

For even cleaner code, use the `ExampleBuilder` helper class to reduce boilerplate:

```python
from toolvaluator import ExampleBuilder, get_tool_schemas_sync, eval_model
from my_server import mcp

# Fetch tool schemas
tool_schemas = get_tool_schemas_sync(mcp)

# Create builder
builder = ExampleBuilder(tool_schemas)

# Add examples with minimal boilerplate
builder.add_positive(
    tool="search_docs",
    query="Find our vacation policy",
    arguments={"query": "vacation policy"}
)

builder.add_negative(
    tool="search_docs",
    query="What is 2+2?"  # Should answer directly
)

# Method chaining works too!
builder.add_positive(
    tool="get_weather",
    query="Weather in Tokyo?",
    arguments={"location": "Tokyo", "units": None}  # None = wildcard
).add_positive(
    tool="calculate",
    query="What is 5 * 10?",
    arguments={"operation": "multiply", "a": 5, "b": 10}
)

# Run evaluation
result = eval_model(
    model_name="gpt-4o-mini",
    api_key="your-api-key",
    dataset=builder.examples,
    verbose=True,
)
```

**Key benefits:**
- **No manual schema extraction**: Automatically extracts tool_name, tool_description, and tool_schema
- **Cleaner syntax**: Focus on query and expected arguments, not boilerplate
- **Method chaining**: Fluently build test suites
- **Built-in validation**: Raises errors if tool doesn't exist
- **Convenience methods**: `add_positive()` and `add_negative()` for common cases

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

## Advanced Features

### Wildcard Argument Values

You can use `None` as a wildcard value in `expected_arguments` to indicate that a specific argument **must exist** but you **don't care about its value**. This is useful when:
- You want to verify the model extracts a parameter but the exact value varies
- Testing argument presence without caring about content
- The value is dynamic (timestamps, IDs, generated text, etc.)

**Examples:**

```python
# Exact value checking (strict)
expected_arguments = {
    "location": "Tokyo",
    "units": "celsius"
}
# Model must return exactly: location="Tokyo", units="celsius"

# Wildcard value checking (flexible)
expected_arguments = {
    "location": None,      # Any location is acceptable
    "units": "celsius"     # Must be exactly "celsius"
}
# Model can return: location="Tokyo" ✓, location="Paris" ✓, etc.

# Mixed approach
expected_arguments = {
    "to": None,            # Any email address
    "subject": "Meeting",  # Must be exactly "Meeting"
    "body": None          # Any body text
}
```

**Important distinctions:**
- `expected_arguments=None` → Don't care about any arguments at all (score 1.0)
- `expected_arguments={}` → Expect NO arguments (score 0.0 if model provides any)
- `expected_arguments={"key": None}` → Expect "key" to exist with any value

### Enhanced Argument Scoring

The argument comparison uses sophisticated scoring that:

1. **Exact matches**: Each correctly matched argument contributes `1/n` to the score (where n = number of expected arguments)
2. **Wildcards**: Arguments with `None` values count as matched if the key exists
3. **Missing keys**: Arguments that should exist but don't contribute `0/n` to the score
4. **Extra keys penalty**: Unexpected arguments incur a `-0.5/n` penalty per extra key
5. **Final score**: `max(0.0, base_score - extra_penalty)`

**Examples:**

```python
# Perfect match
expected = {"arg1": "val1", "arg2": 42}
predicted = {"arg1": "val1", "arg2": 42}
# Score: 1.0

# Partial match
expected = {"arg1": "val1", "arg2": 42}
predicted = {"arg1": "val1", "arg2": 99}
# Score: 0.5 (only arg1 matched)

# Extra arguments penalty
expected = {"arg1": "val1", "arg2": 42}
predicted = {"arg1": "val1", "arg2": 42, "extra1": "foo", "extra2": "bar"}
# Base: 2/2 = 1.0, Penalty: (2 * 0.5) / 2 = 0.5, Final: 0.5

# Empty dict means "no arguments expected"
expected = {}
predicted = {"arg1": "val1"}
# Score: 0.0 (model should not have provided arguments)
```

### Tool Context Awareness

The evaluator provides the model with full tool context to prevent hallucination:
- **Tool name**: The exact name of the tool being evaluated
- **Tool description**: What the tool does
- **Tool schema**: Parameter definitions and types

This prevents the model from hallucinating tool names or misunderstanding tool purposes. Each evaluation asks: "Given this specific tool and this query, should the tool be called and with what arguments?"

### Chained Tool Calls (Advanced)

**WARNING: This feature actually executes tools on your MCP server!**

For testing multi-step workflows where one tool's output feeds into the next, use `ChainedEvaluator`:

```python
from toolvaluator import ChainedEvaluator, get_tool_schemas_sync
from my_server import mcp

tool_schemas = get_tool_schemas_sync(mcp)

# Create chained evaluator
chain = ChainedEvaluator(
    tool_schemas=tool_schemas,
    mcp_server=mcp,
    model_name="gpt-4o-mini",
    api_key="your-api-key"
)

# Define the chain - tools will be executed in sequence!
chain.add_step(
    initial_query="Calculate 15 * 23, then divide the result by 5",
    expected_tool="calculate",
    expected_arguments={"operation": "multiply", "a": 15, "b": 23}
)

chain.add_step(
    expected_tool="calculate",
    expected_arguments={"operation": "divide", "a": None, "b": 5}  # 'a' comes from step 1
)

# Execute and evaluate
result = chain.evaluate()

print(f"Overall score: {result['score']:.2f}")
print(f"Steps completed: {result['num_steps']}")
for step in result['step_results']:
    print(f"  Step {step['step']}: {step['predicted_tool']} → {step['tool_result']}")
```

**How it works:**
1. Model receives initial query
2. Model decides which tool to call (step 1)
3. **Tool is actually executed** on your MCP server
4. Result is fed back to model as context
5. Model decides next tool call (step 2)
6. Process continues for all steps
7. Each step is evaluated independently

**Side Effects & Warnings:**

**Tools are executed for real** - This is not a simulation!

- **Data modification**: Tools may create, update, or delete data
- **Network calls**: APIs may be called, emails sent, etc.
- **Resource consumption**: Database queries, file operations, etc.
- **Costs**: API calls to external services may incur charges
- **Idempotency**: Running the chain multiple times may produce different results

**Best practices:**
- Use test/staging MCP servers, not production
- Implement mock/test versions of tools with side effects
- Use tools that are safe to execute multiple times
- Log all tool executions for audit trails
- Consider implementing dry-run mode in your tools

**When to use:**
- Testing multi-step agent workflows
- Validating tool orchestration logic
- Integration testing of tool chains
- Debugging complex tool interactions

**When NOT to use:**
- Production data or services
- Tools with irreversible side effects
- Financial transactions or critical operations
- Unless you fully understand the implications!

## Configuration Options

### `toolvaluator` CLI Options

Run evaluations using the built-in test dataset:

```bash
toolvaluator [OPTIONS]
```

**Options:**
- `--model`: Model name (default: `gpt-4o-mini`)
- `--api-key`: API key (defaults to `OPENAI_API_KEY` env var)
- `--base-url`: Base URL for OpenAI-compatible endpoints (e.g., `http://localhost:1234/v1`)
- `--min-score`: Minimum acceptable score 0-1 (exits with code 1 if below threshold)
- `--server`: Python module containing your FastMCP server (default: `server`)
- `--server-var`: Name of the FastMCP instance variable (default: `mcp`)
  - Use this if your server uses a different variable name like `app` or `server`
- `--verbose`, `-v`: Show detailed debug information for each example

**Examples:**
```bash
# Basic usage
toolvaluator --server my_server

# Custom variable name
toolvaluator --server my_server --server-var app

# Local model with verbose output
toolvaluator --server my_server \
  --model devstral-small \
  --base-url http://localhost:1234/v1 \
  --api-key lm-studio \
  --verbose
```

### `toolvaluator-init` CLI Options

Generate custom evaluation scripts:

```bash
toolvaluator-init [OPTIONS]
```

**Options:**
- `--server`: Python module containing your MCP server (required)
- `--server-var`: Name of the FastMCP instance variable (default: `mcp`)
- `--output`, `-o`: Output filename (default: `eval_tools.py`)
- `--tools`: Tool names to generate examples for (auto-detected if not specified)
- `--force`, `-f`: Overwrite output file if it exists

**Examples:**
```bash
# Auto-detect tools and generate script
toolvaluator-init --server my_server --output eval_my_tools.py

# Specify custom variable name
toolvaluator-init --server my_server --server-var app --output eval.py

# Specify specific tools
toolvaluator-init --server my_server --tools tool1 tool2 tool3

# Overwrite existing file
toolvaluator-init --server my_server --output eval.py --force
```

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

Use generated evaluation scripts in your CI/CD pipeline to ensure tool quality:

**Step 1: Generate evaluation script (one-time)**
```bash
toolvaluator-init --server my_server --output eval_tools.py
# Edit eval_tools.py to add your evaluation examples
# Commit eval_tools.py to your repository
```

**Step 2: Add to your CI pipeline**

GitHub Actions example (`.github/workflows/test.yml`):
```yaml
name: Test MCP Tools

on: [push, pull_request]

jobs:
  test-tools:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install toolvaluator
          pip install -r requirements.txt  # Your project dependencies

      - name: Evaluate MCP Tools
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python eval_tools.py --model gpt-4o-mini --min-score 0.85 --verbose
```

**Using the CLI directly:**
```bash
# Fail the build if tool-calling accuracy drops below 85%
toolvaluator --server my_server --min-score 0.85
```

**Benefits:**
- Catch regressions in tool descriptions or schemas
- Ensure model compatibility before deployment
- Track quality metrics over time
- Prevent degradation of tool-calling accuracy

### 4. Latency Benchmarking

Track tool-calling latency across model versions or configurations:

```bash
# Compare latency between models
python eval_tools.py --model gpt-4o-mini --verbose > results_mini.txt
python eval_tools.py --model gpt-4o --verbose > results_4o.txt

# Analyze latency stats from output
grep "Latency stats" results_*.txt
```

### 5. Custom Evaluation Workflows

Create specialized evaluation scripts for different scenarios:

```bash
# Generate evaluation for production tools
toolvaluator-init --server prod_server --output eval_prod.py

# Generate evaluation for experimental tools
toolvaluator-init --server experimental_server --output eval_experimental.py

# Run both in your test suite
python eval_prod.py --model gpt-4o --min-score 0.90      # High bar for prod
python eval_experimental.py --model gpt-4o --min-score 0.70  # Lower bar for experiments
```

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
