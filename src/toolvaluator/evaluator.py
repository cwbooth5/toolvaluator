"""
Core evaluation logic for testing LLM tool-calling capabilities.

This module provides functions to:
- Fetch tool schemas from FastMCP servers
- Evaluate LLM tool-calling decisions using DSPy
- Score correctness and measure latency
"""

import asyncio
import json
import statistics as stats
import time
from typing import Any

import dspy
from fastmcp import Client

# ---------------------------------------------------------------------
# 1. Fetch tool schemas from FastMCP (single source of truth)
# ---------------------------------------------------------------------


async def _fetch_tool_schemas(mcp_server) -> dict[str, dict[str, Any]]:
    """
    Connect to the FastMCP server in-memory and return a mapping:
       { tool_name: { ...tool_definition... } }

    This uses the MCP "tools/list" surface, so the structure will match
    the standard MCP Tool definition (inputSchema, description, etc.).

    Args:
        mcp_server: A FastMCP server instance

    Returns:
        Dict mapping tool names to their definitions
    """
    async with Client(mcp_server) as client:
        tools = await client.list_tools()
        # tools is typically a list of dicts with at least:
        # { "name": str, "description": str, "inputSchema": { ... } }
        by_name = {}
        for t in tools:
            # "t" might be a dict or a pydantic-like object; handle both
            if isinstance(t, dict):
                name = t.get("name")
                by_name[name] = t
            else:
                # fallback: object with attributes
                name = getattr(t, "name", None)
                by_name[name] = t
        return by_name


def get_tool_schemas_sync(mcp_server) -> dict[str, dict[str, Any]]:
    """
    Synchronous wrapper for fetching tool schemas.

    Args:
        mcp_server: A FastMCP server instance

    Returns:
        Dict mapping tool names to their definitions
    """
    return asyncio.run(_fetch_tool_schemas(mcp_server))


def extract_input_schema(tool_def: Any) -> dict[str, Any]:
    """
    Extract the JSON Schema for tool inputs from the tool definition.

    MCP protocol uses 'inputSchema' (camelCase).
    FastMCP internal formats sometimes use 'input_schema' (snake_case).

    This helper tries both.

    Args:
        tool_def: A tool definition (dict or object)

    Returns:
        The input schema as a dict
    """
    if isinstance(tool_def, dict):
        return tool_def.get("inputSchema") or tool_def.get("input_schema") or {}
    else:
        # object-like; try attributes
        return (
            getattr(tool_def, "inputSchema", None)
            or getattr(tool_def, "input_schema", {})
            or {}
        )


def extract_tool_description(tool_def: Any) -> str:
    """
    Extract the tool description from the tool definition.

    Args:
        tool_def: A tool definition (dict or object)

    Returns:
        The tool description string
    """
    if isinstance(tool_def, dict):
        return tool_def.get("description", "")
    else:
        return getattr(tool_def, "description", "")


# ---------------------------------------------------------------------
# 2. DSPy Signature and Module for generic tool-calling
# ---------------------------------------------------------------------


class GenericToolCaller(dspy.Signature):
    """
    Evaluate whether the model correctly uses a specific tool for the given query.

    Given a user query and a tool's name, description, and parameter schema,
    determine if the tool should be called and what arguments should be passed.
    """

    user_query: str = dspy.InputField(desc="The user's natural language query")
    tool_name: str = dspy.InputField(desc="The name of the tool being evaluated")
    tool_description: str = dspy.InputField(desc="Description of what the tool does")
    tool_schema_json: str = dspy.InputField(
        desc="JSON schema of the tool's input parameters"
    )

    should_call: bool = dspy.OutputField(
        desc="True if this tool should be used for the query; False if the model should answer directly without calling any tool."
    )
    arguments_json: str = dspy.OutputField(
        desc="JSON object of arguments to pass to the tool (matching the schema). Empty {} if should_call is False."
    )


class GenericToolCallerModule(dspy.Module):
    """
    Wraps GenericToolCaller and measures latency per prediction.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(GenericToolCaller)

    def forward(
        self,
        user_query: str,
        tool_name: str,
        tool_description: str,
        tool_schema: dict[str, Any],
    ):
        """
        Make a prediction for whether and how to call a tool.

        Args:
            user_query: The user's query string
            tool_name: Name of the tool being evaluated
            tool_description: Description of what the tool does
            tool_schema: The tool's input parameter schema as a dict

        Returns:
            A prediction object with should_call, arguments,
            and latency_ms attributes. Also adds tool_name for compatibility.
        """
        # Tool schema comes in as a dict from FastMCP; we serialize for the LM
        schema_json = json.dumps(tool_schema, ensure_ascii=False)

        t0 = time.perf_counter()
        pred = self.predict(
            user_query=user_query,
            tool_name=tool_name,
            tool_description=tool_description,
            tool_schema_json=schema_json,
        )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # Parse arguments_json (best-effort)
        try:
            args = json.loads(pred.arguments_json or "{}")
        except Exception:
            args = {}
        pred.arguments = args
        pred.latency_ms = latency_ms
        # Add tool_name to pred for compatibility with metric function
        pred.tool_name = tool_name
        return pred


# ---------------------------------------------------------------------
# 3. Metric helpers: argument comparison + main metric
# ---------------------------------------------------------------------


def compare_arguments(
    expected: dict[str, Any] | None, predicted: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    """
    Compare expected vs predicted arguments and return a score.

    Args:
        expected: Expected arguments dict (or None)
            - None: Don't care about arguments (score 1.0)
            - {}: Expect NO arguments (fail if model provides any)
            - {...}: Expect specific arguments
            - {"key": None}: Expect key to exist, don't care about value (wildcard)
        predicted: Predicted arguments dict

    Returns:
        Tuple of (score, details) where score is in [0,1]

    Scoring:
    - Each expected key that matches exactly: +1 point
    - Each expected key with None value (wildcard) that exists: +1 point
    - Each missing expected key: 0 points
    - Each extra unexpected key: -0.5 points (capped at 0)
    - Empty expected {} but predicted has args: 0.0
    """
    # None means "don't care about arguments"
    if expected is None:
        return 1.0, {"reason": "no_expected_args_specified"}

    # Empty dict means "expect NO arguments"
    if not expected and predicted:
        return 0.0, {
            "error": "expected_no_args_but_got_some",
            "unexpected_keys": list(predicted.keys()),
            "unexpected_args": predicted,
        }

    # Empty expected and empty predicted = perfect
    if not expected and not predicted:
        return 1.0, {"reason": "both_empty"}

    if not isinstance(predicted, dict):
        return 0.0, {"error": "predicted_arguments_not_dict", "predicted": predicted}

    # Count matches and mismatches
    total_expected = len(expected)
    correct = 0
    mismatches = {}
    extra_keys = []

    # Check expected keys
    for key, expected_val in expected.items():
        if key not in predicted:
            mismatches[key] = {"expected": expected_val, "predicted": None}
            continue
        pred_val = predicted[key]
        # If expected value is None, it's a wildcard (any value is acceptable)
        if expected_val is None:
            correct += 1  # Key exists, don't care about value
        elif pred_val == expected_val:
            correct += 1
        else:
            mismatches[key] = {"expected": expected_val, "predicted": pred_val}

    # Check for extra keys in predicted
    for key in predicted:
        if key not in expected:
            extra_keys.append(key)

    # Calculate score
    if total_expected == 0:
        # No expected args, but we already handled empty case above
        score = 1.0
    else:
        # Base score: fraction of expected keys that matched
        base_score = correct / total_expected

        # Penalty for extra keys: -0.5 points per extra key (relative to expected count)
        # This prevents score from going negative but penalizes extra arguments
        extra_penalty = (len(extra_keys) * 0.5) / total_expected
        score = max(0.0, base_score - extra_penalty)

    details = {"mismatches": mismatches}
    if extra_keys:
        details["extra_keys"] = extra_keys
        details["extra_args"] = {k: predicted[k] for k in extra_keys}

    return score, details


def tool_call_metric(example: dspy.Example, pred, _trace=None) -> float:
    """
    Metric to evaluate whether the model:
      - correctly decided to call the tool
      - chose the right tool name
      - filled arguments reasonably

    Latency is attached to pred.debug but NOT used in the score (yet).

    Args:
        example: The example with expected behavior
        pred: The model's prediction
        trace: Optional trace (unused)

    Returns:
        A correctness score in [0,1]
    """

    # 1) should_call correctness (if provided)
    should_call_score = 1.0
    if hasattr(example, "expected_should_call"):
        should_call_score = float(
            bool(pred.should_call) == bool(example.expected_should_call)
        )

    # 2) tool name correctness (if should_call is True and expected_tool_name provided)
    tool_name_score = 1.0
    if getattr(example, "expected_should_call", True) and hasattr(
        example, "expected_tool_name"
    ):
        tool_name_score = 1.0 if pred.tool_name == example.expected_tool_name else 0.0

    # 3) argument correctness
    expected_args = getattr(example, "expected_arguments", None)
    arg_score, arg_details = compare_arguments(
        expected_args,
        getattr(pred, "arguments", {}),
    )

    correctness_score = (should_call_score + tool_name_score + arg_score) / 3.0

    latency_ms = getattr(pred, "latency_ms", None)

    pred.debug = {
        "should_call_score": should_call_score,
        "tool_name_score": tool_name_score,
        "arg_score": arg_score,
        "arg_details": arg_details,
        "latency_ms": latency_ms,
        "pred_tool_name": pred.tool_name,
        "pred_arguments": getattr(pred, "arguments", {}),
    }

    return correctness_score


# ---------------------------------------------------------------------
# 4. Build a dataset of examples
# ---------------------------------------------------------------------


class ChainedEvaluator:
    """
    Evaluate chained tool calls where one tool's output feeds into the next.

    **WARNING: This actually executes tools on your MCP server!**

    This is useful for testing multi-step workflows where:
    1. Model calls first tool with arguments
    2. Tool is ACTUALLY EXECUTED
    3. Result is fed back to model
    4. Model calls second tool using the result
    5. Process continues for all steps

    Side effects:
    - Tools are executed for real
    - Data may be created/modified/deleted depending on tool behavior
    - Network calls may be made
    - Use with caution in production environments

    Usage:
        chain = ChainedEvaluator(
            tool_schemas=tool_schemas,
            mcp_server=mcp,
            model_name="gpt-4o-mini",
            api_key="your-key"
        )

        chain.add_step(
            initial_query="What's the weather where I am?",
            expected_tool="get_location",
            expected_arguments={}
        )

        chain.add_step(
            expected_tool="get_weather",
            expected_arguments={"location": None}  # Will use result from step 1
        )

        result = chain.evaluate()
    """

    def __init__(
        self,
        tool_schemas: dict[str, Any],
        mcp_server,
        model_name: str,
        api_key: str,
        base_url: str | None = None,
    ):
        """
        Initialize the chained evaluator.

        Args:
            tool_schemas: Dict mapping tool names to their definitions
            mcp_server: FastMCP server instance (for executing tools)
            model_name: Name of the model to evaluate
            api_key: API key for the model
            base_url: Optional base URL for OpenAI-compatible endpoints
        """
        self.tool_schemas = tool_schemas
        self.mcp_server = mcp_server
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.steps: list[dict[str, Any]] = []
        self.execution_history: list[dict[str, Any]] = []

    def add_step(
        self,
        expected_tool: str,
        expected_arguments: dict[str, Any] | None = None,
        initial_query: str | None = None,
    ) -> "ChainedEvaluator":
        """
        Add a step to the chain.

        Args:
            expected_tool: Name of tool that should be called in this step
            expected_arguments: Expected arguments (None for wildcards)
            initial_query: For first step only - the user's initial query

        Returns:
            Self for method chaining
        """
        if expected_tool not in self.tool_schemas:
            raise ValueError(
                f"Tool '{expected_tool}' not found in tool_schemas. "
                f"Available: {', '.join(self.tool_schemas.keys())}"
            )

        self.steps.append({
            "expected_tool": expected_tool,
            "expected_arguments": expected_arguments or {},
            "initial_query": initial_query,
        })
        return self

    async def _execute_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        """Execute a tool call on the MCP server and return the result."""
        from fastmcp import Client

        async with Client(self.mcp_server) as client:
            result = await client.call_tool(tool_name, arguments=arguments)
            # Extract text content from result
            if hasattr(result, "content") and result.content:
                content_item = result.content[0]
                if hasattr(content_item, "text"):
                    return content_item.text
            return str(result)

    def evaluate(self) -> dict[str, Any]:
        """
        Execute the chain and evaluate each step.

        Returns:
            Dict with overall score, per-step results, and execution history

        Raises:
            ValueError: If no steps defined or first step missing initial_query
        """
        import asyncio

        if not self.steps:
            raise ValueError("No steps defined. Use add_step() to add steps.")

        if not self.steps[0].get("initial_query"):
            raise ValueError("First step must have initial_query set")

        # Configure DSPy
        if self.base_url:
            model_name = (
                self.model_name
                if "/" in self.model_name
                else f"openai/{self.model_name}"
            )
            lm = dspy.LM(
                model=model_name, api_key=self.api_key, base_url=self.base_url
            )
        else:
            lm = dspy.LM(model=self.model_name, api_key=self.api_key)

        dspy.configure(lm=lm)

        tool_caller = GenericToolCallerModule()

        # Track state
        conversation_context = []
        step_results = []
        self.execution_history = []

        # Start with initial query
        current_query = self.steps[0]["initial_query"]

        for i, step in enumerate(self.steps):
            print(f"\n=== Step {i + 1}/{len(self.steps)} ===")
            print(f"Query: {current_query}")

            # Get tool metadata
            expected_tool = step["expected_tool"]
            tool_description = extract_tool_description(
                self.tool_schemas[expected_tool]
            )
            tool_schema = extract_input_schema(self.tool_schemas[expected_tool])

            # Ask model what to do
            pred = tool_caller(
                user_query=current_query,
                tool_name=expected_tool,
                tool_description=tool_description,
                tool_schema=tool_schema,
            )

            # Check if model called expected tool
            tool_correct = pred.tool_name == expected_tool
            args_score, args_details = compare_arguments(
                step["expected_arguments"], pred.arguments
            )

            print(f"Model called: {pred.tool_name}")
            print(f"With arguments: {pred.arguments}")
            print(f"Tool correct: {tool_correct}, Args score: {args_score:.2f}")

            # Execute the tool if model got it right
            tool_result = None
            if tool_correct and pred.should_call:
                print(f"Executing {pred.tool_name}...")
                try:
                    tool_result = asyncio.run(
                        self._execute_tool_call(pred.tool_name, pred.arguments)
                    )
                    print(f"Result: {tool_result}")
                except Exception as e:
                    print(f"Tool execution failed: {e}")
                    tool_result = f"Error: {e}"

            # Record step
            step_result = {
                "step": i + 1,
                "query": current_query,
                "expected_tool": expected_tool,
                "predicted_tool": pred.tool_name,
                "tool_correct": tool_correct,
                "expected_arguments": step["expected_arguments"],
                "predicted_arguments": pred.arguments,
                "args_score": args_score,
                "args_details": args_details,
                "tool_result": tool_result,
            }
            step_results.append(step_result)
            self.execution_history.append(step_result)

            # Update context for next step
            if tool_result:
                conversation_context.append({
                    "query": current_query,
                    "tool": pred.tool_name,
                    "result": tool_result,
                })

                # Build query for next step using conversation history
                if i + 1 < len(self.steps):
                    context_str = "\n".join(
                        [
                            f"Called {item['tool']}({item.get('args', '')}) → {item['result']}"
                            for item in conversation_context
                        ]
                    )
                    current_query = (
                        f"Previous context:\n{context_str}\n\n"
                        f"Continue the task to accomplish the original goal."
                    )

        # Calculate overall score
        tool_scores = [
            1.0 if r["tool_correct"] else 0.0 for r in step_results
        ]
        args_scores = [r["args_score"] for r in step_results]
        overall_score = (sum(tool_scores) + sum(args_scores)) / (
            2 * len(step_results)
        )

        return {
            "score": overall_score,
            "step_results": step_results,
            "execution_history": self.execution_history,
            "num_steps": len(self.steps),
        }


class ExampleBuilder:
    """
    Helper class to reduce boilerplate when creating evaluation examples.

    Usage:
        builder = ExampleBuilder(tool_schemas)

        # Add a positive example (should call the tool)
        builder.add(
            tool="search_docs",
            query="Find our PTO policy",
            should_call=True,
            arguments={"query": "PTO policy"}
        )

        # Add a negative example (should NOT call the tool)
        builder.add(
            tool="search_docs",
            query="What is 2+2?",
            should_call=False
        )

        # Get all examples
        examples = builder.examples
    """

    def __init__(self, tool_schemas: dict[str, Any]):
        """
        Initialize the example builder.

        Args:
            tool_schemas: Dict mapping tool names to their definitions
        """
        self.tool_schemas = tool_schemas
        self.examples: list[dspy.Example] = []

    def add(
        self,
        tool: str,
        query: str,
        should_call: bool = True,
        arguments: dict[str, Any] | None = None,
    ) -> "ExampleBuilder":
        """
        Add an evaluation example with minimal boilerplate.

        Args:
            tool: Name of the tool being tested
            query: User's natural language query
            should_call: Whether the tool should be called for this query
            arguments: Expected arguments (None for "don't care", {} for "no args")

        Returns:
            Self for method chaining

        Raises:
            ValueError: If tool is not in tool_schemas
        """
        if tool not in self.tool_schemas:
            raise ValueError(
                f"Tool '{tool}' not found in tool_schemas. "
                f"Available tools: {', '.join(self.tool_schemas.keys())}"
            )

        # Extract tool metadata
        tool_description = extract_tool_description(self.tool_schemas[tool])
        tool_schema = extract_input_schema(self.tool_schemas[tool])

        # Set expected values based on should_call
        expected_tool_name = tool if should_call else ""
        expected_arguments = arguments if arguments is not None else {}

        # Create and add the example
        example = dspy.Example(
            user_query=query,
            tool_name=tool,
            tool_description=tool_description,
            tool_schema=tool_schema,
            expected_should_call=should_call,
            expected_tool_name=expected_tool_name,
            expected_arguments=expected_arguments,
        ).with_inputs("user_query", "tool_name", "tool_description", "tool_schema")

        self.examples.append(example)
        return self

    def add_positive(
        self, tool: str, query: str, arguments: dict[str, Any] | None = None
    ) -> "ExampleBuilder":
        """
        Add a positive example (tool should be called).

        Convenience method equivalent to add(tool, query, should_call=True, arguments).
        """
        return self.add(tool, query, should_call=True, arguments=arguments)

    def add_negative(self, tool: str, query: str) -> "ExampleBuilder":
        """
        Add a negative example (tool should NOT be called).

        Convenience method equivalent to add(tool, query, should_call=False).
        """
        return self.add(tool, query, should_call=False, arguments={})

    def build(self) -> list[dspy.Example]:
        """
        Return the list of examples.

        Alias for accessing .examples directly.
        """
        return self.examples


def build_dataset(tool_schemas: dict[str, Any]) -> list[dspy.Example]:
    """
    Build a list of dspy.Example objects for evaluation.

    This uses the ExampleBuilder helper to reduce boilerplate.
    You should customize this to match your real tools & queries.

    Args:
        tool_schemas: Dict mapping tool names to their definitions

    Returns:
        List of dspy.Example objects
    """
    builder = ExampleBuilder(tool_schemas)

    # Examples for "search_docs" tool
    if "search_docs" in tool_schemas:
        builder.add_positive(
            tool="search_docs",
            query="Find our PTO policy for new hires.",
            arguments={"query": "PTO policy new hires"},
        )
        builder.add_negative(
            tool="search_docs",
            query="What is 2 + 2?",  # Should answer directly, not search
        )

    # Examples for "get_weather" tool
    if "get_weather" in tool_schemas:
        builder.add_positive(
            tool="get_weather",
            query="What's the weather like in Tokyo?",
            arguments={"location": "Tokyo", "units": "celsius"},
        )
        builder.add_positive(
            tool="get_weather",
            query="Get me the weather in New York in fahrenheit",
            arguments={"location": "New York", "units": "fahrenheit"},
        )

    # Examples for "calculate" tool
    if "calculate" in tool_schemas:
        builder.add_positive(
            tool="calculate",
            query="What is 15 multiplied by 23?",
            arguments={"operation": "multiply", "a": 15, "b": 23},
        )
        builder.add_positive(
            tool="calculate",
            query="Divide 100 by 4",
            arguments={"operation": "divide", "a": 100, "b": 4},
        )

    # Examples for "send_email" tool
    if "send_email" in tool_schemas:
        builder.add_positive(
            tool="send_email",
            query="Send an email to john@example.com with subject 'Meeting Tomorrow' and body 'Don't forget our 10am meeting'",
            arguments={
                "to": "john@example.com",
                "subject": "Meeting Tomorrow",
                "body": "Don't forget our 10am meeting",
            },
        )

    # Examples for "create_task" tool
    if "create_task" in tool_schemas:
        builder.add_positive(
            tool="create_task",
            query="Create a high priority task called 'Fix login bug' due 2025-12-01",
            arguments={
                "title": "Fix login bug",
                "priority": "high",
                "due_date": "2025-12-01",
            },
        )

    return builder.examples


# ---------------------------------------------------------------------
# 5. Evaluation driver
# ---------------------------------------------------------------------


def eval_model(
    model_name: str,
    api_key: str,
    base_url: str | None,
    dataset: list[dspy.Example],
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Configure DSPy with the given model, run the evaluation, and
    print correctness + latency stats.

    Args:
        model_name: Name of the model to evaluate
        api_key: API key for the model
        base_url: Optional base URL for OpenAI-compatible endpoints
        dataset: List of examples to evaluate on
        verbose: If True, print detailed debug information for each example

    Returns:
        Dict with 'score', 'latencies', 'predictions', and 'scores' keys
    """

    # Configure the LM
    # When base_url is provided, ensure model name has openai/ prefix for DSPy compatibility
    # (since we're using an OpenAI-compatible API)
    if base_url:
        # If using a custom base_url and model doesn't have a provider prefix,
        # add openai/ prefix for DSPy compatibility
        if "/" not in model_name:
            model_name = f"openai/{model_name}"
        lm = dspy.LM(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
        )
    else:
        # For hosted models, require provider prefix (e.g., openai/, anthropic/)
        lm = dspy.LM(
            model=model_name,
            api_key=api_key,
        )

    dspy.configure(lm=lm)

    tool_caller = GenericToolCallerModule()

    # Run predictions and collect results manually for latency tracking
    predictions = []
    scores = []
    latencies = []

    print("\nEvaluating...")
    for i, example in enumerate(dataset, 1):
        pred = tool_caller(
            user_query=example.user_query,
            tool_name=example.tool_name,
            tool_description=example.tool_description,
            tool_schema=example.tool_schema,
        )
        predictions.append(pred)

        # Calculate score
        example_score = tool_call_metric(example, pred)
        scores.append(example_score)

        # Extract latency
        dbg = getattr(pred, "debug", {})
        lat = dbg.get("latency_ms")
        if lat is not None:
            latencies.append(lat)

        # Print basic score
        print(f"  [{i}/{len(dataset)}] Score: {example_score:.3f}")

        # Print detailed debug info if verbose
        if verbose:
            print(f"    Query: {example.user_query}")
            print()
            print("    Expected behavior:")
            print(f"      should_call: {getattr(example, 'expected_should_call', 'N/A')}")
            print(f"      tool_name: {getattr(example, 'expected_tool_name', 'N/A')}")
            print(f"      arguments: {getattr(example, 'expected_arguments', {})}")
            print()
            print("    Model's tool call:")
            print(f"      should_call: {pred.should_call}")
            print(f"      tool_name: {dbg.get('pred_tool_name', 'N/A')}")
            print(f"      arguments: {dbg.get('pred_arguments', {})}")
            print()
            print("    Scores breakdown:")
            print(f"      - Should call: {dbg.get('should_call_score', 'N/A'):.3f}")
            print(f"      - Tool name: {dbg.get('tool_name_score', 'N/A'):.3f}")
            print(f"      - Arguments: {dbg.get('arg_score', 'N/A'):.3f}")

            arg_details = dbg.get("arg_details", {})

            if arg_details.get("mismatches"):
                print()
                print("    Argument mismatches:")
                for key, details in arg_details["mismatches"].items():
                    print(
                        f"      - {key}: expected={details['expected']}, got={details['predicted']}"
                    )

            if arg_details.get("extra_keys"):
                print()
                print("    Extra unexpected arguments:")
                for key in arg_details["extra_keys"]:
                    print(f"      - {key}: {arg_details['extra_args'][key]}")

            if arg_details.get("error"):
                print()
                print(f"    Error: {arg_details['error']}")
            print()
            print(f"    Latency: {lat:.1f}ms" if lat else "    Latency: N/A")
            print()
            print("    " + "=" * 70)
            print()

    overall_score = sum(scores) / len(scores) if scores else 0.0
    print(f"\nModel: {model_name}")
    print(f"Correctness score (0-1): {overall_score:.3f}")

    if latencies:
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        p50 = stats.median(latencies_sorted)
        p95 = latencies_sorted[int(n * 0.95) - 1] if n > 1 else latencies_sorted[0]
        print(f"Latency stats (ms) over {n} calls:")
        print(f"  mean : {stats.mean(latencies_sorted):.1f}")
        print(f"  p50  : {p50:.1f}")
        print(f"  p95  : {p95:.1f}")
        print(f"  max  : {latencies_sorted[-1]:.1f}")
    else:
        print("No latency data collected.")

    return {
        "score": overall_score,
        "latencies": latencies,
        "predictions": predictions,
        "scores": scores,
    }
