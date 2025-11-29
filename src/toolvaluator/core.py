"""
Core evaluation components for LLM tool-calling.

This module provides the foundational components:
- Tool schema extraction utilities
- DSPy signatures for tool calling
- Scoring functions for evaluating tool call correctness
"""

import asyncio
import json
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
        system_prompt: str | None = None,
    ):
        """
        Make a prediction for whether and how to call a tool.

        Args:
            user_query: The user's query string
            tool_name: Name of the tool being evaluated
            tool_description: Description of what the tool does
            tool_schema: The tool's input parameter schema as a dict
            system_prompt: Optional system prompt to prepend to the context

        Returns:
            A prediction object with should_call, arguments,
            and latency_ms attributes. Also adds tool_name for compatibility.
        """
        # Tool schema comes in as a dict from FastMCP; we serialize for the LM
        schema_json = json.dumps(tool_schema, ensure_ascii=False)

        # If system prompt is provided, prepend it to the user query
        effective_query = user_query
        if system_prompt:
            effective_query = f"{system_prompt}\n\n{user_query}"

        t0 = time.perf_counter()
        pred = self.predict(
            user_query=effective_query,
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
