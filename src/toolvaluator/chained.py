"""
Chained tool evaluation for multi-step workflows.

This module provides functionality for evaluating sequences of tool calls:
- ChainedEvaluator: Executes and evaluates multi-step tool workflows
- eval_chained_model: High-level function for chained evaluation
"""

import asyncio
from typing import Any

import dspy
from fastmcp import Client

from .core import (
    GenericToolCallerModule,
    compare_arguments,
    extract_input_schema,
    extract_tool_description,
)


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
        # Without mocks (tools execute for real)
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

    Usage with mocks (no side effects):
        # Global mocks - apply to all steps
        chain = ChainedEvaluator(
            tool_schemas=tool_schemas,
            mcp_server=mcp,
            model_name="gpt-4o-mini",
            api_key="your-key",
            mocks={
                "get_location": "San Francisco",
                "calculate": lambda args: str(args["a"] * args["b"])  # Callable mock
            }
        )

        # Per-step mocks - override global mocks for specific steps
        chain.add_step(
            initial_query="Calculate 5 * 10",
            expected_tool="calculate",
            expected_arguments={"operation": "multiply", "a": 5, "b": 10},
            mock_result="50"  # Use this instead of executing or global mock
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
        mocks: dict[str, Any] | None = None,
        verbose: bool = False,
        system_prompt: str | None = None,
    ):
        """
        Initialize the chained evaluator.

        Args:
            tool_schemas: Dict mapping tool names to their definitions
            mcp_server: FastMCP server instance (for executing tools)
            model_name: Name of the model to evaluate
            api_key: API key for the model
            base_url: Optional base URL for OpenAI-compatible endpoints
            mocks: Optional dict of tool_name -> mock_result or callable
                   Use to avoid executing tools (reduces side effects)
                   Can be string result or callable(args) -> result
            verbose: If True, print detailed debug information for each step
            system_prompt: Optional default system prompt for all steps
                          (can be overridden per-step)
        """
        self.tool_schemas = tool_schemas
        self.mcp_server = mcp_server
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.mocks = mocks or {}
        self.verbose = verbose
        self.system_prompt = system_prompt
        self.steps: list[dict[str, Any]] = []
        self.execution_history: list[dict[str, Any]] = []

    def add_step(
        self,
        expected_tool: str,
        expected_arguments: dict[str, Any] | None = None,
        initial_query: str | None = None,
        mock_result: str | None = None,
    ) -> "ChainedEvaluator":
        """
        Add a step to the chain.

        Args:
            expected_tool: Name of tool that should be called in this step
            expected_arguments: Expected arguments (None for wildcards)
            initial_query: Optional query for this step. Required for first step.
                          For subsequent steps, if provided, will be used as the
                          query along with previous context. If not provided for
                          subsequent steps, uses generic continuation message.
            mock_result: Optional mock result to use instead of executing tool
                        Overrides any global mock for this specific step

        Returns:
            Self for method chaining
        """
        if expected_tool not in self.tool_schemas:
            raise ValueError(
                f"Tool '{expected_tool}' not found in tool_schemas. "
                f"Available: {', '.join(self.tool_schemas.keys())}"
            )

        self.steps.append(
            {
                "expected_tool": expected_tool,
                "expected_arguments": expected_arguments or {},
                "initial_query": initial_query,
                "mock_result": mock_result,
            }
        )
        return self

    async def _execute_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        """Execute a tool call on the MCP server and return the result."""
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
            lm = dspy.LM(model=model_name, api_key=self.api_key, base_url=self.base_url)
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

            # Priority: step-level system_prompt > eval-level system_prompt > None
            step_system_prompt = step.get("system_prompt")
            effective_system_prompt = (
                step_system_prompt
                if step_system_prompt is not None
                else self.system_prompt
            )

            # Ask model what to do
            pred = tool_caller(
                user_query=current_query,
                tool_name=expected_tool,
                tool_description=tool_description,
                tool_schema=tool_schema,
                system_prompt=effective_system_prompt,
            )

            # Check if model called expected tool
            tool_correct = pred.tool_name == expected_tool
            args_score, args_details = compare_arguments(
                step["expected_arguments"], pred.arguments
            )

            print(f"Model called: {pred.tool_name}")
            print(f"With arguments: {pred.arguments}")
            print(f"Tool correct: {tool_correct}, Args score: {args_score:.2f}")

            # Verbose output
            if self.verbose:
                print()
                print("    Expected behavior:")
                print(f"      tool: {expected_tool}")
                print(f"      arguments: {step['expected_arguments']}")
                print()
                print("    Model's tool call:")
                print(f"      should_call: {pred.should_call}")
                print(f"      tool: {pred.tool_name}")
                print(f"      arguments: {pred.arguments}")
                print()
                print("    Scores:")
                print(f"      Tool correct: {1.0 if tool_correct else 0.0:.3f}")
                print(f"      Arguments: {args_score:.3f}")

                if args_details.get("mismatches"):
                    print()
                    print("    Argument mismatches:")
                    for key, details in args_details["mismatches"].items():
                        print(
                            f"      - {key}: expected={details['expected']}, got={details['predicted']}"
                        )

                if args_details.get("extra_keys"):
                    print()
                    print("    Extra unexpected arguments:")
                    for key in args_details["extra_keys"]:
                        print(f"      - {key}: {args_details['extra_args'][key]}")

                if args_details.get("error"):
                    print()
                    print(f"    Error: {args_details['error']}")

                latency_ms = getattr(pred, "latency_ms", None)
                if latency_ms:
                    print()
                    print(f"    Latency: {latency_ms:.1f}ms")
                print()

            # Execute the tool if model got it right
            tool_result = None
            if tool_correct and pred.should_call:
                # Check for mocks (per-step overrides global)
                mock_result = step.get("mock_result")
                global_mock = self.mocks.get(pred.tool_name)

                if mock_result is not None:
                    # Per-step mock takes priority
                    tool_result = mock_result
                    print(f"Using mock result for {pred.tool_name}: {tool_result}")
                elif global_mock is not None:
                    # Global mock
                    if callable(global_mock):
                        tool_result = global_mock(pred.arguments)
                    else:
                        tool_result = str(global_mock)
                    print(f"Using global mock for {pred.tool_name}: {tool_result}")
                else:
                    # Actually execute the tool
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
                conversation_context.append(
                    {
                        "query": current_query,
                        "tool": pred.tool_name,
                        "result": tool_result,
                    }
                )

                # Build query for next step using conversation history
                if i + 1 < len(self.steps):
                    next_step = self.steps[i + 1]
                    context_str = "\n".join(
                        [
                            f"Called {item['tool']}({item.get('args', '')}) → {item['result']}"
                            for item in conversation_context
                        ]
                    )

                    # Check if next step has its own initial_query
                    if next_step.get("initial_query"):
                        # Use the step's query with context
                        current_query = (
                            f"Previous context:\n{context_str}\n\n"
                            f"{next_step['initial_query']}"
                        )
                    else:
                        # Use generic continuation message
                        current_query = (
                            f"Previous context:\n{context_str}\n\n"
                            f"Continue the task to accomplish the original goal."
                        )

        # Calculate overall score
        tool_scores = [1.0 if r["tool_correct"] else 0.0 for r in step_results]
        args_scores = [r["args_score"] for r in step_results]
        overall_score = (sum(tool_scores) + sum(args_scores)) / (2 * len(step_results))

        return {
            "score": overall_score,
            "step_results": step_results,
            "execution_history": self.execution_history,
            "num_steps": len(self.steps),
        }


def eval_chained_model(
    model_name: str,
    api_key: str,
    dataset: list[dict[str, Any]],
    base_url: str | None = None,
    verbose: bool = False,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate chained tool calls following the same pattern as eval_model().

    This function takes a dataset of chained examples and evaluates them,
    providing the same workflow as eval_model() but for multi-step chains.

    Args:
        model_name: Name of the model to evaluate
        api_key: API key for the model
        dataset: List of chain configurations from ChainedExampleBuilder.build()
        base_url: Optional base URL for OpenAI-compatible endpoints
        verbose: If True, print detailed debug information for each step
        system_prompt: Optional default system prompt for all steps
                      (can be overridden per-step)

    Returns:
        Dict with:
            - score: Overall score across all chains (0-1)
            - chain_results: List of results for each chain
            - num_chains: Total number of chains evaluated
            - num_steps: Total number of steps across all chains

    Usage:
        builder = ChainedExampleBuilder(tool_schemas, mcp_server)
        builder.add_chain(
            mocks={"calculate": lambda args: str(args["a"] * args["b"])}
        ).add_step(
            initial_query="Calculate 5 * 10",
            expected_tool="calculate",
            expected_arguments={"operation": "multiply", "a": 5, "b": 10}
        ).add_step(
            expected_tool="calculate",
            expected_arguments={"operation": "add", "a": None, "b": 25}
        )

        dataset = builder.build()

        # Test with different models
        result1 = eval_chained_model(
            model_name="gpt-4o-mini",
            api_key="key",
            dataset=dataset
        )

        result2 = eval_chained_model(
            model_name="claude-3-sonnet",
            api_key="key",
            dataset=dataset
        )
    """
    if not dataset:
        raise ValueError(
            "Dataset is empty. Use ChainedExampleBuilder to create chains."
        )

    print(f"\nEvaluating {len(dataset)} chain(s)...")

    all_chain_results = []
    all_scores = []
    total_steps = 0

    for chain_idx, chain_config in enumerate(dataset, 1):
        print(f"\n{'=' * 70}")
        print(f"Chain {chain_idx}/{len(dataset)}")
        print(f"{'=' * 70}")

        # Create evaluator for this chain
        chain = ChainedEvaluator(
            tool_schemas=chain_config["tool_schemas"],
            mcp_server=chain_config["mcp_server"],
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            mocks=chain_config.get("mocks", {}),
            verbose=verbose,
            system_prompt=system_prompt,
        )

        # Add steps
        for step in chain_config["steps"]:
            chain.add_step(
                expected_tool=step["expected_tool"],
                expected_arguments=step["expected_arguments"],
                initial_query=step.get("initial_query"),
                mock_result=step.get("mock_result"),
            )

        # Evaluate chain
        result = chain.evaluate()
        all_chain_results.append(result)
        all_scores.append(result["score"])
        total_steps += result["num_steps"]

    # Calculate overall statistics
    overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    print(f"\n{'=' * 70}")
    print("Overall Results")
    print(f"{'=' * 70}")
    print(f"Chains evaluated: {len(dataset)}")
    print(f"Total steps: {total_steps}")
    print(f"Overall score: {overall_score:.3f}")

    return {
        "score": overall_score,
        "chain_results": all_chain_results,
        "num_chains": len(dataset),
        "num_steps": total_steps,
    }
