"""
Command-line interface for toolvaluator.

This module provides the CLI entry point for evaluating LLM tool usage.
You could run this in your CI/build system to burn the evaluation into
your process.
"""

import argparse
import importlib
import os
import sys

from .evaluator import build_dataset, eval_model, get_tool_schemas_sync


def main():
    """Main CLI entry point for toolvaluator."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM tool usage and latency with DSPy + FastMCP."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name (OpenAI or OpenAI-compatible).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the model. Defaults to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional base URL for OpenAI-compatible endpoints (e.g. local OSS server).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum acceptable correctness score. If overall score is below, exit 1.",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="server",
        help="Python module containing your FastMCP server (default: 'server'). "
        "The module should expose a FastMCP instance named 'mcp'.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed debug information for each example.",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key and not args.base_url:
        print(
            "ERROR: No API key provided and no base-url specified.",
            file=sys.stderr,
        )
        print(
            "Set OPENAI_API_KEY or pass --api-key or --base-url for local models.",
            file=sys.stderr,
        )
        sys.exit(1)

    # We need to use this so we can use parts of the FastMCP client lib.
    try:
        server_module = importlib.import_module(args.server)
        mcp = server_module.mcp
    except ImportError as e:
        print(
            f"ERROR: Could not import server module '{args.server}': {e}",
            file=sys.stderr,
        )
        print(
            "Make sure your FastMCP server module is in PYTHONPATH.",
            file=sys.stderr,
        )
        sys.exit(1)
    except AttributeError:
        print(
            f"ERROR: Module '{args.server}' does not have an 'mcp' attribute.",
            file=sys.stderr,
        )
        print(
            "Your FastMCP server module should expose a FastMCP instance named 'mcp'.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Fetching tool schemas from FastMCP...")
    tool_schemas = get_tool_schemas_sync(mcp)
    if not tool_schemas:
        print(
            "ERROR: No tools found on the FastMCP server.",
            file=sys.stderr,
        )
        print(
            "Check your server import / @mcp.tool definitions.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Discovered tools: {', '.join(tool_schemas.keys())}")

    dataset = build_dataset(tool_schemas)
    if not dataset:
        print(
            "ERROR: Dataset is empty. Add some examples in build_dataset().",
            file=sys.stderr,
        )
        sys.exit(1)

    result = eval_model(
        model_name=args.model,
        api_key=api_key,
        base_url=args.base_url,
        dataset=dataset,
        verbose=args.verbose,
    )

    score = result["score"]
    if score < args.min_score:
        print(f"\nFAIL: score {score:.3f} < min-score {args.min_score:.3f}")
        sys.exit(1)
    else:
        print(f"\nPASS: score {score:.3f} >= min-score {args.min_score:.3f}")
        sys.exit(0)


if __name__ == "__main__":
    main()
