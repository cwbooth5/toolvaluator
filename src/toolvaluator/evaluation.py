"""
Regular (single tool) evaluation functions.

This module provides functionality for evaluating single tool call decisions:
- eval_model: Evaluate a model on a dataset of single tool examples
- build_dataset: Legacy helper for building test datasets
"""

import statistics as stats
from typing import Any

import dspy

from .builders import ExampleBuilder
from .core import GenericToolCallerModule, tool_call_metric


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


def eval_model(
    model_name: str,
    api_key: str,
    base_url: str | None,
    dataset: list[dspy.Example],
    verbose: bool = False,
    system_prompt: str | None = None,
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
        system_prompt: Optional default system prompt for all examples
                      (can be overridden per-example)

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
        # Priority: example-level system_prompt > eval-level system_prompt > None
        example_system_prompt = getattr(example, "system_prompt", None)
        effective_system_prompt = (
            example_system_prompt
            if example_system_prompt is not None
            else system_prompt
        )

        pred = tool_caller(
            user_query=example.user_query,
            tool_name=example.tool_name,
            tool_description=example.tool_description,
            tool_schema=example.tool_schema,
            system_prompt=effective_system_prompt,
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
            print(
                f"      should_call: {getattr(example, 'expected_should_call', 'N/A')}"
            )
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
