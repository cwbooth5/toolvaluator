"""
Core evaluation logic for testing LLM tool-calling capabilities.

This module re-exports all evaluation components for backwards compatibility.
The actual implementations have been refactored into separate modules:
- core: DSPy signatures, utilities, and metrics
- builders: ExampleBuilder and ChainedExampleBuilder classes
- chained: ChainedEvaluator and eval_chained_model
- evaluation: eval_model and build_dataset

You can import directly from these modules or from this module for compatibility.
"""

# Re-export from core
# Re-export from builders
from .builders import ChainedExampleBuilder, ExampleBuilder

# Re-export from chained
from .chained import ChainedEvaluator, eval_chained_model
from .core import (
    GenericToolCaller,
    GenericToolCallerModule,
    compare_arguments,
    extract_input_schema,
    extract_tool_description,
    get_tool_schemas_sync,
    tool_call_metric,
)

# Re-export from evaluation
from .evaluation import build_dataset, eval_model

__all__ = [
    # Core utilities
    "get_tool_schemas_sync",
    "extract_input_schema",
    "extract_tool_description",
    # DSPy components
    "GenericToolCaller",
    "GenericToolCallerModule",
    # Metrics
    "compare_arguments",
    "tool_call_metric",
    # Builders
    "ExampleBuilder",
    "ChainedExampleBuilder",
    # Chained evaluation
    "ChainedEvaluator",
    "eval_chained_model",
    # Regular evaluation
    "eval_model",
    "build_dataset",
]
