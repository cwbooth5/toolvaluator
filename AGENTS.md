# AGENTS.md - Development Guide for AI Assistants

This document provides coding standards, architectural overview, and development guidelines for working on the toolvaluator project.

## Project Overview

Toolvaluator is an MCP (Model Context Protocol) tool schema evaluation framework for testing LLM tool-calling capabilities. It helps developers:
- Evaluate how well language models use FastMCP tools
- Measure correctness and latency of tool-calling decisions
- Compare different models and optimize tool schemas
- Test both single tool calls and chained multi-step workflows

## Architecture Overview

The codebase is organized into focused modules, each with a specific responsibility:

### Core Modules

#### 1. `src/toolvaluator/core.py` (350 lines)
Core evaluation components used by all other modules:

**Tool Schema Utilities:**
- `get_tool_schemas_sync()`: Fetch tool schemas from FastMCP servers
- `extract_input_schema()`: Extract input schema from tool definitions
- `extract_tool_description()`: Extract tool descriptions

**DSPy Components:**
- `GenericToolCaller`: DSPy signature for tool calling decisions
- `GenericToolCallerModule`: Wrapper that measures latency and handles system prompts

**Scoring Functions:**
- `compare_arguments()`: Compares expected vs predicted arguments with wildcard support
- `tool_call_metric()`: Overall scoring combining tool selection and argument correctness

#### 2. `src/toolvaluator/builders.py` (270 lines)
Dataset builder classes for creating evaluation examples:

- `ExampleBuilder`: Builds datasets for single-tool evaluation
  - Fluent API with method chaining
  - `add()`, `add_positive()`, `add_negative()` methods
  - Supports per-example system prompts

- `ChainedExampleBuilder`: Builds datasets for multi-step chained evaluation
  - `add_chain()` to start a new chain
  - `add_step()` to add steps to the current chain
  - Supports mocks and per-step system prompts

#### 3. `src/toolvaluator/chained.py` (430 lines)
Chained tool evaluation for multi-step workflows:

- `ChainedEvaluator`: Executes and evaluates multi-step tool workflows
  - Actually calls tools and feeds results to next step
  - Supports mocking to avoid side effects (global and per-step)
  - Tracks execution history

- `eval_chained_model()`: High-level function for chained evaluation
  - Takes dataset and model config
  - Returns overall score and per-chain results

#### 4. `src/toolvaluator/evaluation.py` (250 lines)
Regular (single tool) evaluation:

- `eval_model()`: Evaluates single tool call decisions
  - Configures DSPy with model credentials
  - Runs predictions and calculates scores
  - Prints latency statistics
  - Supports per-example system prompts

- `build_dataset()`: Legacy helper for building test datasets

#### 5. `src/toolvaluator/evaluator.py` (55 lines)
Re-exports all components from other modules for backwards compatibility.
Import from here or directly from specific modules.

#### 6. CLI Tools
- `src/toolvaluator/cli.py`: Main evaluation CLI
- `src/toolvaluator/init.py`: Kickstart tool for generating evaluation scripts

#### 7. Test Server
- `src/toolvaluator/test_server.py`: Example FastMCP server with test tools

### Key Design Patterns

#### 1. Consistent API Pattern
Both regular and chained evaluation follow the same workflow:

```python
# Step 1: Build dataset
builder = ExampleBuilder(tool_schemas)  # or ChainedExampleBuilder
builder.add_positive(...)  # or add_chain().add_step()
dataset = builder.build()

# Step 2: Evaluate with model config
result = eval_model(  # or eval_chained_model
    model_name="...",
    api_key="...",
    dataset=dataset
)
```

**Key Principle:** Dataset = what to test, Eval function = which model to test with

#### 2. Priority Systems

**System Prompts:**
- Priority: Example/Step-level > Eval-level > None
- Allows per-example customization with sensible defaults

**Mocks (Chained Evaluation):**
- Priority: Per-step mock_result > Global mocks dict > Actual execution
- Reduces side effects while allowing real tool execution when needed

#### 3. Builder Pattern
- Fluent API with method chaining
- Reduces boilerplate
- Returns `self` from all add methods

#### 4. Wildcard Arguments
- `expected_arguments={"key": None}` means "key must exist, any value accepted"
- `expected_arguments=None` means "don't care about arguments"
- `expected_arguments={}` means "expect no arguments"

## Coding Standards

### Python Version
- **Required:** Python 3.12+
- Use modern Python 3.12 features (type hints with `|`, etc.)

### Type Hints
- **Required** for all function signatures
- Use `from typing import Any` for generic types
- Use `str | None` instead of `Optional[str]`
- Use `list[dict[str, Any]]` instead of `List[Dict[str, Any]]`

Example:
```python
def eval_model(
    model_name: str,
    api_key: str,
    base_url: str | None,
    dataset: list[dspy.Example],
    verbose: bool = False,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    ...
```

### Docstrings
- **Required** for all public functions and classes
- Use Google-style docstrings
- Include Args, Returns, Raises sections where applicable

Example:
```python
def compare_arguments(
    expected: dict[str, Any] | None,
    predicted: dict[str, Any] | str,
) -> tuple[float, dict[str, Any]]:
    """
    Compare expected vs predicted tool arguments with wildcard support.

    Args:
        expected: Expected arguments (None for "don't care", {} for "no args")
        predicted: Predicted arguments from the model

    Returns:
        Tuple of (score, details_dict)
    """
```

### Code Organization
- Keep functions focused and single-purpose
- Use clear section comments for major code sections
- Group related functionality together
- Avoid functions longer than ~100 lines

### Naming Conventions
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private functions/methods: `_leading_underscore`
- Be descriptive: `eval_chained_model()` not `eval_chain()`

### Error Handling
- Use descriptive error messages
- Raise `ValueError` for invalid input
- Include context in error messages

Example:
```python
if tool not in self.tool_schemas:
    raise ValueError(
        f"Tool '{tool}' not found in tool_schemas. "
        f"Available tools: {', '.join(self.tool_schemas.keys())}"
    )
```

### No Emojis
- Do **not** use emojis in code, comments, or output messages
- Keep all text professional and emoji-free

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/toolvaluator --cov-report=term-missing

# Run specific test file
pytest tests/test_evaluator.py -v

# Run specific test
pytest tests/test_evaluator.py::test_compare_arguments_wildcard_value -v
```

### Test Organization
- Tests in `tests/` directory
- One test file per module: `test_evaluator.py`, `test_server.py`
- Use pytest fixtures for shared setup
- Test filenames: `test_*.py`
- Test functions: `test_*`

### Test Fixtures
Key fixtures in `tests/conftest.py`:
- `test_mcp_server`: FastMCP server instance for testing

### Writing Tests
- One assertion concept per test
- Use descriptive test names: `test_compare_arguments_wildcard_value`
- Include docstrings explaining what is being tested
- Test both success and failure cases

Example:
```python
def test_compare_arguments_wildcard_value():
    """Test that None value acts as wildcard (don't care about value)."""
    expected = {"arg1": "value1", "arg2": None}  # arg2 is wildcard
    predicted = {"arg1": "value1", "arg2": "any_value_works"}
    score, details = compare_arguments(expected, predicted)
    assert score == 1.0
    assert details["mismatches"] == {}
```

### Test Coverage Goals
- Aim for 80%+ coverage
- All public APIs must have tests
- Test edge cases and error conditions

## Code Quality Tools

### Formatting with Ruff

```bash
# Format all code (auto-fix)
ruff format src/ tests/

# Check formatting without fixing
ruff format --check src/ tests/
```

**Configuration:** See `pyproject.toml` section `[tool.ruff]`
- Line length: 88 characters
- Target: Python 3.12

**Rules:**
- Ruff format is the authority on code formatting
- Always run before committing
- No manual line breaks needed - ruff handles it

### Static Analysis with Ruff

```bash
# Check for issues
ruff check src/ tests/

# Auto-fix fixable issues
ruff check --fix src/ tests/
```

**Enabled Rules:**
- `E`: pycodestyle errors
- `W`: pycodestyle warnings
- `F`: pyflakes
- `I`: isort (import sorting)
- `B`: flake8-bugbear
- `C4`: flake8-comprehensions
- `UP`: pyupgrade
- `ARG`: flake8-unused-arguments
- `SIM`: flake8-simplify

**Key Suppressions:**
- `E501`: Line too long (handled by ruff format)
- `F401`: Unused imports in `__init__.py` files are allowed

### Pre-Commit Checklist

Before committing code:

1. **Format code:**
   ```bash
   ruff format src/ tests/
   ```

2. **Run static analysis:**
   ```bash
   ruff check src/ tests/
   ```

3. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

4. **Verify all pass:**
   - All tests should pass
   - No ruff errors
   - Code properly formatted

## Development Workflow

### Making Changes

1. **Understand the architecture** (see above)
2. **Read relevant code** before modifying
3. **Follow existing patterns** in the codebase
4. **Update tests** for any changes
5. **Update documentation** (README.md, docstrings)
6. **Run quality checks** (ruff + pytest)

### Adding New Features

When adding features, maintain consistency:

1. **API Consistency:** Follow the builder → dataset → eval pattern
2. **Priority Systems:** If adding configuration, consider priority levels
3. **Documentation:** Update README.md with examples
4. **Tests:** Add comprehensive test coverage
5. **Type Hints:** Fully type-annotated function signatures

### File Structure

```
toolvaluator/
├── src/toolvaluator/
│   ├── __init__.py          # Public API exports
│   ├── core.py              # DSPy signatures, utilities, metrics (350 lines)
│   ├── builders.py          # ExampleBuilder, ChainedExampleBuilder (270 lines)
│   ├── chained.py           # ChainedEvaluator, eval_chained_model (430 lines)
│   ├── evaluation.py        # eval_model, build_dataset (250 lines)
│   ├── evaluator.py         # Re-exports for backwards compatibility (55 lines)
│   ├── cli.py               # CLI tool
│   ├── init.py              # Kickstart script generator
│   └── test_server.py       # Example MCP server
├── tests/
│   ├── conftest.py          # Test fixtures
│   ├── test_evaluator.py    # Core tests
│   └── test_server.py       # Server tests
├── .github/workflows/
│   ├── release.yml          # Release automation
│   └── ci.yml               # Continuous integration
├── pyproject.toml           # Project config, dependencies, tools
├── README.md                # User documentation
├── RELEASING.md             # Release process
└── AGENTS.md                # This file
```

### Key Files to Understand

**`src/toolvaluator/__init__.py`** (40 lines):
- Exports public API from all modules
- Defines `__all__` for explicit exports
- Version is defined here: `__version__ = "..."`

**`src/toolvaluator/core.py`** (350 lines):
- Foundation layer used by all other modules
- DSPy signatures and tool calling logic
- Scoring functions and metrics
- Tool schema extraction utilities

**`src/toolvaluator/builders.py`** (270 lines):
- Dataset builder classes
- ExampleBuilder for single tool evaluation
- ChainedExampleBuilder for multi-step workflows

**`src/toolvaluator/chained.py`** (430 lines):
- ChainedEvaluator for executing multi-step workflows
- eval_chained_model() high-level evaluation function
- Mock support to avoid side effects

**`src/toolvaluator/evaluation.py`** (250 lines):
- eval_model() for single tool evaluation
- build_dataset() legacy helper function
- Latency tracking and statistics

**`src/toolvaluator/evaluator.py`** (55 lines):
- Re-exports all components for backwards compatibility
- Import from here or directly from specific modules

**`pyproject.toml`**:
- Project metadata and dependencies
- Tool configuration (ruff, pytest, coverage)
- Build system configuration (hatch)

## Dependencies

### Core Dependencies
- `dspy-ai>=2.4.0`: LLM evaluation framework
- `fastmcp>=0.2.0`: MCP server framework

### Dev Dependencies
- `pytest>=8.0.0`: Testing framework
- `pytest-asyncio>=0.23.0`: Async test support
- `pytest-cov>=4.1.0`: Coverage reporting
- `ruff>=0.8.0`: Linting and formatting
- `black>=24.0.0`: Additional formatting (deprecated, use ruff)

### Build System
- `hatchling`: Build backend
- `hatch`: Version management and building

## Common Pitfalls to Avoid

### 1. Don't Break API Consistency
- Both eval functions must have the same signature pattern
- Dataset creation must follow the builder pattern
- Don't put model config in dataset builders

### 2. Don't Skip Type Hints
- All function signatures need full type annotations
- Use `dict[str, Any]` not `dict`
- Use `str | None` not `Optional[str]`

### 3. Don't Ignore Test Failures
- All tests must pass before changes are complete
- If tests fail, fix the code or update the tests (with justification)
- Don't skip or comment out failing tests

### 4. Don't Add Emojis
- No emojis in code, comments, or output
- This is a professional tool library

### 5. Don't Forget Documentation
- Update README.md when adding features
- Update docstrings when changing behavior
- Examples should be working code

## Version Management

Versions are managed via hatch:

```bash
# Show current version
hatch version

# Bump version
hatch version patch  # 0.1.0 -> 0.1.1
hatch version minor  # 0.1.0 -> 0.2.0
hatch version major  # 0.1.0 -> 1.0.0
```

Version is stored in `src/toolvaluator/__init__.py:8` as `__version__`.

## Building

```bash
# Build wheel and source distribution
hatch build

# Output in dist/
# - toolvaluator-{version}-py3-none-any.whl
# - toolvaluator-{version}.tar.gz
```

## Questions?

If you need clarification on any aspect of the codebase:
1. Read the relevant code section
2. Check the README.md for user-facing documentation
3. Look at existing tests for examples
4. Review recent git history for patterns

Remember: **Consistency is key.** Follow existing patterns in the codebase.
