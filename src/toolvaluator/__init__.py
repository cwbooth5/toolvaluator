"""
Toolvaluator: An MCP tool schema evaluation framework.

This package provides utilities to evaluate how well language models
use FastMCP tools by measuring correctness and latency.

Module Structure:
- core: DSPy signatures, utilities, and metrics
- builders: ExampleBuilder and ChainedExampleBuilder classes
- chained: ChainedEvaluator and eval_chained_model function
- evaluation: eval_model function and build_dataset helper
- evaluator: Re-exports all components (backwards compatibility)

You can import from the package root or from specific modules:
    from toolvaluator import eval_model  # Main package import
    from toolvaluator.builders import ExampleBuilder  # Direct module import
"""

__version__ = "1.0.0"

from .evaluator import (
    ChainedEvaluator,
    ChainedExampleBuilder,
    ExampleBuilder,
    GenericToolCallerModule,
    build_dataset,
    compare_arguments,
    eval_chained_model,
    eval_model,
    extract_input_schema,
    extract_tool_description,
    get_tool_schemas_sync,
    tool_call_metric,
)

__all__ = [
    "__version__",
    "ChainedEvaluator",
    "ChainedExampleBuilder",
    "ExampleBuilder",
    "GenericToolCallerModule",
    "build_dataset",
    "compare_arguments",
    "eval_chained_model",
    "eval_model",
    "extract_input_schema",
    "extract_tool_description",
    "get_tool_schemas_sync",
    "tool_call_metric",
]
