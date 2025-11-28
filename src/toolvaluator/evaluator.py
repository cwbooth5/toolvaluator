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


# ---------------------------------------------------------------------
# 2. DSPy Signature and Module for generic tool-calling
# ---------------------------------------------------------------------


class GenericToolCaller(dspy.Signature):
    """
    Decide whether to call the given tool for this user query,
    and if so, fill in the tool name and arguments.

    The tool schema is passed as a JSON string so the model can "read"
    it and construct proper arguments.
    """

    user_query: str = dspy.InputField()
    tool_schema_json: str = dspy.InputField()

    should_call: bool = dspy.OutputField(
        desc="True if the tool should be used; False if the model should answer directly."
    )
    tool_name: str = dspy.OutputField(
        desc="The name of the tool to call, or empty string if should_call is false."
    )
    arguments_json: str = dspy.OutputField(
        desc="JSON object of arguments that match the tool's parameters schema."
    )


class GenericToolCallerModule(dspy.Module):
    """
    Wraps GenericToolCaller and measures latency per prediction.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(GenericToolCaller)

    def forward(self, user_query: str, tool_schema: dict[str, Any]):
        """
        Make a prediction for whether and how to call a tool.

        Args:
            user_query: The user's query string
            tool_schema: The tool's input schema as a dict

        Returns:
            A prediction object with should_call, tool_name, arguments,
            and latency_ms attributes
        """
        # Tool schema comes in as a dict from FastMCP; we serialize for the LM
        schema_json = json.dumps(tool_schema, ensure_ascii=False)

        t0 = time.perf_counter()
        pred = self.predict(
            user_query=user_query,
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
        predicted: Predicted arguments dict

    Returns:
        Tuple of (score, details) where score is in [0,1]

    Simple version: fraction of expected keys that match exactly.
    - If expected is None or empty, treat as full score (1.0).
    """
    if not expected:
        return 1.0, {"reason": "no_expected_args_specified"}

    if not isinstance(predicted, dict):
        return 0.0, {"error": "predicted_arguments_not_dict", "predicted": predicted}

    total = len(expected)
    correct = 0
    mismatches = {}

    for key, expected_val in expected.items():
        if key not in predicted:
            mismatches[key] = {"expected": expected_val, "predicted": None}
            continue
        pred_val = predicted[key]
        if pred_val == expected_val:
            correct += 1
        else:
            mismatches[key] = {"expected": expected_val, "predicted": pred_val}

    score = correct / total if total > 0 else 1.0
    return score, {"mismatches": mismatches}


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


def build_dataset(tool_schemas: dict[str, Any]) -> list[dspy.Example]:
    """
    Build a list of dspy.Example objects for evaluation.

    You should customize this to match your real tools & queries.

    Each example should have:
      - user_query
      - tool_schema: the *input* schema for the intended tool
      - expected_should_call (bool)
      - expected_tool_name (str)
      - expected_arguments (dict)

    Args:
        tool_schemas: Dict mapping tool names to their definitions

    Returns:
        List of dspy.Example objects
    """

    examples: list[dspy.Example] = []

    # Examples for "search_docs" tool
    if "search_docs" in tool_schemas:
        search_docs_schema = extract_input_schema(tool_schemas["search_docs"])

        examples.append(
            dspy.Example(
                user_query="Find our PTO policy for new hires.",
                tool_schema=search_docs_schema,
                expected_should_call=True,
                expected_tool_name="search_docs",
                expected_arguments={"query": "PTO policy new hires"},
            ).with_inputs("user_query", "tool_schema")
        )

        examples.append(
            dspy.Example(
                user_query="What is 2 + 2?",
                tool_schema=search_docs_schema,
                expected_should_call=False,  # model should answer directly, not search
                expected_tool_name="",
                expected_arguments={},
            ).with_inputs("user_query", "tool_schema")
        )

    # Examples for "get_weather" tool
    if "get_weather" in tool_schemas:
        weather_schema = extract_input_schema(tool_schemas["get_weather"])

        examples.append(
            dspy.Example(
                user_query="What's the weather like in Tokyo?",
                tool_schema=weather_schema,
                expected_should_call=True,
                expected_tool_name="get_weather",
                expected_arguments={"location": "Tokyo", "units": "celsius"},
            ).with_inputs("user_query", "tool_schema")
        )

        examples.append(
            dspy.Example(
                user_query="Get me the weather in New York in fahrenheit",
                tool_schema=weather_schema,
                expected_should_call=True,
                expected_tool_name="get_weather",
                expected_arguments={"location": "New York", "units": "fahrenheit"},
            ).with_inputs("user_query", "tool_schema")
        )

    # Examples for "calculate" tool
    if "calculate" in tool_schemas:
        calc_schema = extract_input_schema(tool_schemas["calculate"])

        examples.append(
            dspy.Example(
                user_query="What is 15 multiplied by 23?",
                tool_schema=calc_schema,
                expected_should_call=True,
                expected_tool_name="calculate",
                expected_arguments={"operation": "multiply", "a": 15, "b": 23},
            ).with_inputs("user_query", "tool_schema")
        )

        examples.append(
            dspy.Example(
                user_query="Divide 100 by 4",
                tool_schema=calc_schema,
                expected_should_call=True,
                expected_tool_name="calculate",
                expected_arguments={"operation": "divide", "a": 100, "b": 4},
            ).with_inputs("user_query", "tool_schema")
        )

    # Examples for "send_email" tool
    if "send_email" in tool_schemas:
        email_schema = extract_input_schema(tool_schemas["send_email"])

        examples.append(
            dspy.Example(
                user_query="Send an email to john@example.com with subject 'Meeting Tomorrow' and body 'Don't forget our 10am meeting'",
                tool_schema=email_schema,
                expected_should_call=True,
                expected_tool_name="send_email",
                expected_arguments={
                    "to": "john@example.com",
                    "subject": "Meeting Tomorrow",
                    "body": "Don't forget our 10am meeting",
                },
            ).with_inputs("user_query", "tool_schema")
        )

    # Examples for "create_task" tool
    if "create_task" in tool_schemas:
        task_schema = extract_input_schema(tool_schemas["create_task"])

        examples.append(
            dspy.Example(
                user_query="Create a high priority task called 'Fix login bug' due 2025-12-01",
                tool_schema=task_schema,
                expected_should_call=True,
                expected_tool_name="create_task",
                expected_arguments={
                    "title": "Fix login bug",
                    "priority": "high",
                    "due_date": "2025-12-01",
                },
            ).with_inputs("user_query", "tool_schema")
        )

    return examples


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
            if dbg.get("arg_details", {}).get("mismatches"):
                print()
                print("    Argument mismatches:")
                for key, details in dbg["arg_details"]["mismatches"].items():
                    print(
                        f"      - {key}: expected={details['expected']}, got={details['predicted']}"
                    )
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
