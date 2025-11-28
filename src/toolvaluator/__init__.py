"""
Toolvaluator: An MCP tool schema evaluation framework.

This package provides utilities to evaluate how well language models
use FastMCP tools by measuring correctness and latency.
"""

__version__ = "0.1.0"

from .evaluator import (
    GenericToolCallerModule,
    build_dataset,
    compare_arguments,
    eval_model,
    extract_input_schema,
    extract_tool_description,
    get_tool_schemas_sync,
    tool_call_metric,
)

__all__ = [
    "__version__",
    "GenericToolCallerModule",
    "build_dataset",
    "compare_arguments",
    "eval_model",
    "extract_input_schema",
    "extract_tool_description",
    "get_tool_schemas_sync",
    "tool_call_metric",
]
