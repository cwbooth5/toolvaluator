"""
Builder classes for creating evaluation datasets.

This module provides fluent APIs for building evaluation datasets:
- ExampleBuilder: For single tool call evaluation
- ChainedExampleBuilder: For multi-step chained tool evaluation
"""

from typing import Any

import dspy

from .core import extract_input_schema, extract_tool_description


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
        system_prompt: str | None = None,
    ) -> "ExampleBuilder":
        """
        Add an evaluation example with minimal boilerplate.

        Args:
            tool: Name of the tool being tested
            query: User's natural language query
            should_call: Whether the tool should be called for this query
            arguments: Expected arguments (None for "don't care", {} for "no args")
            system_prompt: Optional system prompt for this example (overrides eval-level)

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
            system_prompt=system_prompt,
        ).with_inputs("user_query", "tool_name", "tool_description", "tool_schema")

        self.examples.append(example)
        return self

    def add_positive(
        self,
        tool: str,
        query: str,
        arguments: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> "ExampleBuilder":
        """
        Add a positive example (tool should be called).

        Convenience method equivalent to add(tool, query, should_call=True, arguments).
        """
        return self.add(
            tool,
            query,
            should_call=True,
            arguments=arguments,
            system_prompt=system_prompt,
        )

    def add_negative(
        self, tool: str, query: str, system_prompt: str | None = None
    ) -> "ExampleBuilder":
        """
        Add a negative example (tool should NOT be called).

        Convenience method equivalent to add(tool, query, should_call=False).
        """
        return self.add(
            tool, query, should_call=False, arguments={}, system_prompt=system_prompt
        )

    def build(self) -> list[dspy.Example]:
        """
        Return the list of examples.

        Alias for accessing .examples directly.
        """
        return self.examples


class ChainedExampleBuilder:
    """
    Helper to build chained evaluation examples using a consistent API pattern.

    This provides the same workflow as ExampleBuilder but for multi-step chains:
    1. Create builder
    2. Add chain(s) with steps
    3. Build dataset
    4. Pass dataset and model config to eval_chained_model()

    Usage:
        builder = ChainedExampleBuilder(tool_schemas, mcp_server)

        # Add a chain (sequence of tool calls)
        builder.add_chain(
            mocks={"calculate": lambda args: str(args["a"] * args["b"])}
        ).add_step(
            initial_query="Calculate 5 * 10, then add 25",
            expected_tool="calculate",
            expected_arguments={"operation": "multiply", "a": 5, "b": 10}
        ).add_step(
            expected_tool="calculate",
            expected_arguments={"operation": "add", "a": None, "b": 25}
        )

        # Build dataset
        dataset = builder.build()

        # Evaluate with different models (model config in eval function!)
        result = eval_chained_model(
            model_name="gpt-4o-mini",
            api_key="your-key",
            dataset=dataset
        )
    """

    def __init__(self, tool_schemas: dict[str, Any], mcp_server):
        """
        Initialize the chained example builder.

        Args:
            tool_schemas: Dict mapping tool names to their definitions
            mcp_server: FastMCP server instance (for executing tools)
        """
        self.tool_schemas = tool_schemas
        self.mcp_server = mcp_server
        self.chains: list[dict[str, Any]] = []
        self._current_chain: dict[str, Any] | None = None

    def add_chain(
        self,
        mocks: dict[str, Any] | None = None,
    ) -> "ChainedExampleBuilder":
        """
        Start a new chain with its configuration.

        Args:
            mocks: Optional dict of tool_name -> mock_result or callable

        Returns:
            Self for method chaining
        """
        # Save previous chain if exists
        if self._current_chain and self._current_chain.get("steps"):
            self.chains.append(self._current_chain)

        # Start new chain
        self._current_chain = {
            "tool_schemas": self.tool_schemas,
            "mcp_server": self.mcp_server,
            "mocks": mocks or {},
            "steps": [],
        }
        return self

    def add_step(
        self,
        expected_tool: str,
        expected_arguments: dict[str, Any] | None = None,
        initial_query: str | None = None,
        mock_result: str | None = None,
        system_prompt: str | None = None,
    ) -> "ChainedExampleBuilder":
        """
        Add a step to the current chain.

        Args:
            expected_tool: Name of tool that should be called in this step
            expected_arguments: Expected arguments (None for wildcards)
            initial_query: Optional query for this step. Required for first step.
                          For subsequent steps, if provided, will be used as the
                          query along with previous context. If not provided for
                          subsequent steps, uses generic continuation message.
            mock_result: Optional mock result to use instead of executing tool
            system_prompt: Optional system prompt for this step (overrides eval-level)

        Returns:
            Self for method chaining
        """
        if self._current_chain is None:
            raise ValueError("Must call add_chain() before add_step()")

        if expected_tool not in self.tool_schemas:
            raise ValueError(
                f"Tool '{expected_tool}' not found in tool_schemas. "
                f"Available: {', '.join(self.tool_schemas.keys())}"
            )

        self._current_chain["steps"].append(
            {
                "expected_tool": expected_tool,
                "expected_arguments": expected_arguments or {},
                "initial_query": initial_query,
                "mock_result": mock_result,
                "system_prompt": system_prompt,
            }
        )
        return self

    def build(self) -> list[dict[str, Any]]:
        """
        Build and return the list of chained examples.

        Returns:
            List of chain configurations ready for eval_chained_model()
        """
        # Save current chain if exists
        if self._current_chain and self._current_chain.get("steps"):
            self.chains.append(self._current_chain)
            self._current_chain = None

        return self.chains
