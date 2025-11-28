"""Tests for the evaluator module."""

from toolvaluator.evaluator import (
    ChainedEvaluator,
    ExampleBuilder,
    compare_arguments,
    extract_input_schema,
    get_tool_schemas_sync,
)


def test_extract_input_schema_from_dict():
    """Test extracting input schema from a dict-based tool definition."""
    tool_def = {
        "name": "test_tool",
        "inputSchema": {"type": "object", "properties": {"arg1": {"type": "string"}}},
    }
    schema = extract_input_schema(tool_def)
    assert schema == {"type": "object", "properties": {"arg1": {"type": "string"}}}


def test_extract_input_schema_snake_case():
    """Test extracting input schema with snake_case key."""
    tool_def = {
        "name": "test_tool",
        "input_schema": {"type": "object", "properties": {"arg1": {"type": "string"}}},
    }
    schema = extract_input_schema(tool_def)
    assert schema == {"type": "object", "properties": {"arg1": {"type": "string"}}}


def test_extract_input_schema_empty():
    """Test extracting input schema when none exists."""
    tool_def = {"name": "test_tool"}
    schema = extract_input_schema(tool_def)
    assert schema == {}


def test_compare_arguments_exact_match():
    """Test comparing arguments with exact match."""
    expected = {"arg1": "value1", "arg2": 42}
    predicted = {"arg1": "value1", "arg2": 42}
    score, details = compare_arguments(expected, predicted)
    assert score == 1.0
    assert details["mismatches"] == {}


def test_compare_arguments_partial_match():
    """Test comparing arguments with partial match."""
    expected = {"arg1": "value1", "arg2": 42}
    predicted = {"arg1": "value1", "arg2": 99}
    score, details = compare_arguments(expected, predicted)
    assert score == 0.5
    assert "arg2" in details["mismatches"]


def test_compare_arguments_missing_key():
    """Test comparing arguments with missing key."""
    expected = {"arg1": "value1", "arg2": 42}
    predicted = {"arg1": "value1"}
    score, details = compare_arguments(expected, predicted)
    assert score == 0.5
    assert details["mismatches"]["arg2"]["predicted"] is None


def test_compare_arguments_none_expected():
    """Test comparing when expected is None (don't care about args)."""
    expected = None
    predicted = {"arg1": "value1"}
    score, details = compare_arguments(expected, predicted)
    assert score == 1.0
    assert details["reason"] == "no_expected_args_specified"


def test_compare_arguments_empty_expected_empty_predicted():
    """Test comparing when both expected and predicted are empty."""
    expected = {}
    predicted = {}
    score, details = compare_arguments(expected, predicted)
    assert score == 1.0
    assert details["reason"] == "both_empty"


def test_compare_arguments_empty_expected_but_predicted_has_args():
    """Test that empty expected {} fails when model provides arguments."""
    expected = {}
    predicted = {"arg1": "value1", "arg2": "value2"}
    score, details = compare_arguments(expected, predicted)
    assert score == 0.0
    assert details["error"] == "expected_no_args_but_got_some"
    assert "arg1" in details["unexpected_keys"]
    assert "arg2" in details["unexpected_keys"]


def test_compare_arguments_invalid_predicted():
    """Test comparing when predicted is not a dict."""
    expected = {"arg1": "value1"}
    predicted = "not a dict"
    score, details = compare_arguments(expected, predicted)
    assert score == 0.0
    assert "error" in details


def test_get_tool_schemas_sync(test_mcp_server):
    """Test fetching tool schemas from the test server."""
    schemas = get_tool_schemas_sync(test_mcp_server)
    assert isinstance(schemas, dict)
    assert len(schemas) > 0
    # Check that our test server tools are present
    assert "search_docs" in schemas
    assert "get_weather" in schemas
    assert "calculate" in schemas


def test_tool_schemas_have_required_fields(test_mcp_server):
    """Test that tool schemas have the required MCP fields."""
    schemas = get_tool_schemas_sync(test_mcp_server)
    for _tool_name, tool_def in schemas.items():
        # Check for name field
        if isinstance(tool_def, dict):
            assert "name" in tool_def or hasattr(tool_def, "name")
        else:
            assert hasattr(tool_def, "name")


def test_compare_arguments_with_extra_keys():
    """Test that extra keys in predicted are penalized."""
    expected = {"arg1": "value1", "arg2": 42}
    predicted = {"arg1": "value1", "arg2": 42, "extra1": "foo", "extra2": "bar"}
    score, details = compare_arguments(expected, predicted)

    # Base score: 2/2 = 1.0
    # Penalty: 2 extra keys * 0.5 / 2 expected = 0.5
    # Final: 1.0 - 0.5 = 0.5
    assert score == 0.5
    assert details["extra_keys"] == ["extra1", "extra2"]
    assert details["extra_args"] == {"extra1": "foo", "extra2": "bar"}


def test_compare_arguments_partial_match_with_extra():
    """Test partial match with extra keys."""
    expected = {"arg1": "value1", "arg2": 42}
    predicted = {"arg1": "value1", "arg2": 99, "extra": "foo"}
    score, details = compare_arguments(expected, predicted)

    # Base score: 1/2 = 0.5 (only arg1 matched)
    # Penalty: 1 extra key * 0.5 / 2 expected = 0.25
    # Final: 0.5 - 0.25 = 0.25
    assert score == 0.25
    assert "arg2" in details["mismatches"]
    assert details["extra_keys"] == ["extra"]


def test_compare_arguments_wildcard_value():
    """Test that None value acts as wildcard (don't care about value)."""
    expected = {"arg1": "value1", "arg2": None}  # arg2 is wildcard
    predicted = {"arg1": "value1", "arg2": "any_value_works"}
    score, details = compare_arguments(expected, predicted)
    assert score == 1.0
    assert details["mismatches"] == {}


def test_compare_arguments_wildcard_missing():
    """Test that wildcard still fails if key is missing."""
    expected = {"arg1": "value1", "arg2": None}  # arg2 is wildcard
    predicted = {"arg1": "value1"}  # arg2 missing
    score, details = compare_arguments(expected, predicted)
    assert score == 0.5  # Only arg1 matched, arg2 missing
    assert "arg2" in details["mismatches"]
    assert details["mismatches"]["arg2"]["predicted"] is None


def test_compare_arguments_all_wildcards():
    """Test all wildcard values."""
    expected = {"arg1": None, "arg2": None, "arg3": None}
    predicted = {"arg1": "foo", "arg2": 42, "arg3": [1, 2, 3]}
    score, details = compare_arguments(expected, predicted)
    assert score == 1.0
    assert details["mismatches"] == {}


def test_compare_arguments_mixed_wildcard_and_specific():
    """Test mix of wildcard and specific values."""
    expected = {"arg1": "value1", "arg2": None, "arg3": 42}
    predicted = {"arg1": "value1", "arg2": "any_value", "arg3": 42}
    score, details = compare_arguments(expected, predicted)
    assert score == 1.0
    assert details["mismatches"] == {}


def test_example_builder_add_positive(test_mcp_server):
    """Test ExampleBuilder.add_positive creates correct example."""
    from toolvaluator import get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    builder = ExampleBuilder(tool_schemas)

    builder.add_positive(
        tool="search_docs", query="Find documentation", arguments={"query": "test"}
    )

    assert len(builder.examples) == 1
    example = builder.examples[0]
    assert example.user_query == "Find documentation"
    assert example.tool_name == "search_docs"
    assert example.expected_should_call is True
    assert example.expected_tool_name == "search_docs"
    assert example.expected_arguments == {"query": "test"}


def test_example_builder_add_negative(test_mcp_server):
    """Test ExampleBuilder.add_negative creates correct example."""
    from toolvaluator import get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    builder = ExampleBuilder(tool_schemas)

    builder.add_negative(tool="search_docs", query="What is 2+2?")

    assert len(builder.examples) == 1
    example = builder.examples[0]
    assert example.user_query == "What is 2+2?"
    assert example.expected_should_call is False
    assert example.expected_tool_name == ""
    assert example.expected_arguments == {}


def test_example_builder_method_chaining(test_mcp_server):
    """Test ExampleBuilder supports method chaining."""
    from toolvaluator import get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    builder = ExampleBuilder(tool_schemas)

    builder.add_positive(
        tool="search_docs", query="Query 1", arguments={"query": "test1"}
    ).add_negative(tool="search_docs", query="Query 2").add_positive(
        tool="get_weather", query="Query 3", arguments={"location": "Tokyo"}
    )

    assert len(builder.examples) == 3


def test_example_builder_invalid_tool(test_mcp_server):
    """Test ExampleBuilder raises error for invalid tool."""
    import pytest

    from toolvaluator import get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    builder = ExampleBuilder(tool_schemas)

    with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
        builder.add_positive(tool="nonexistent", query="test", arguments={"arg": "val"})


def test_example_builder_build_method(test_mcp_server):
    """Test ExampleBuilder.build() returns examples list."""
    from toolvaluator import get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    builder = ExampleBuilder(tool_schemas)

    builder.add_positive(tool="search_docs", query="test", arguments={"query": "test"})

    examples = builder.build()
    assert examples == builder.examples
    assert len(examples) == 1


def test_chained_evaluator_with_static_mock(test_mcp_server):
    """Test ChainedEvaluator with static mock result."""
    from toolvaluator import get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)

    # Create evaluator with static mock for calculate tool
    chain = ChainedEvaluator(
        tool_schemas=tool_schemas,
        mcp_server=test_mcp_server,
        model_name="gpt-4o-mini",
        api_key="test-key",
        mocks={"calculate": "50"},  # Static mock result
    )

    chain.add_step(
        initial_query="Calculate 5 * 10",
        expected_tool="calculate",
        expected_arguments={"operation": "multiply", "a": 5, "b": 10},
    )

    assert len(chain.steps) == 1
    assert chain.mocks["calculate"] == "50"


def test_chained_evaluator_with_callable_mock(test_mcp_server):
    """Test ChainedEvaluator with callable mock."""
    from toolvaluator import get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)

    # Create evaluator with callable mock
    def mock_calculate(args):
        """Mock calculator that just returns sum."""
        return str(args.get("a", 0) + args.get("b", 0))

    chain = ChainedEvaluator(
        tool_schemas=tool_schemas,
        mcp_server=test_mcp_server,
        model_name="gpt-4o-mini",
        api_key="test-key",
        mocks={"calculate": mock_calculate},
    )

    chain.add_step(
        initial_query="Add 10 and 20",
        expected_tool="calculate",
        expected_arguments={"operation": "add", "a": 10, "b": 20},
    )

    assert len(chain.steps) == 1
    assert callable(chain.mocks["calculate"])
    # Test the mock function directly
    assert chain.mocks["calculate"]({"a": 10, "b": 20}) == "30"


def test_chained_evaluator_per_step_mock_override(test_mcp_server):
    """Test per-step mock_result overrides global mock."""
    from toolvaluator import get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)

    # Create evaluator with global mock
    chain = ChainedEvaluator(
        tool_schemas=tool_schemas,
        mcp_server=test_mcp_server,
        model_name="gpt-4o-mini",
        api_key="test-key",
        mocks={"calculate": "999"},  # Global mock
    )

    # Add step with per-step mock override
    chain.add_step(
        initial_query="Calculate 5 + 10",
        expected_tool="calculate",
        expected_arguments={"operation": "add", "a": 5, "b": 10},
        mock_result="15",  # Per-step override
    )

    assert len(chain.steps) == 1
    assert chain.steps[0]["mock_result"] == "15"
    assert chain.mocks["calculate"] == "999"  # Global mock unchanged


def test_chained_evaluator_multiple_steps_with_mocks(test_mcp_server):
    """Test chained evaluator with multiple steps using different mocks."""
    from toolvaluator import get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)

    chain = ChainedEvaluator(
        tool_schemas=tool_schemas,
        mcp_server=test_mcp_server,
        model_name="gpt-4o-mini",
        api_key="test-key",
        mocks={
            "calculate": lambda args: str(args["a"] * args["b"]),
            "search_docs": "Found documentation about X",
        },
    )

    chain.add_step(
        initial_query="Calculate 5 * 10",
        expected_tool="calculate",
        expected_arguments={"operation": "multiply", "a": 5, "b": 10},
    )

    chain.add_step(
        expected_tool="search_docs",
        expected_arguments={"query": "result"},
        mock_result="Custom result for this step",  # Override global mock
    )

    assert len(chain.steps) == 2
    assert chain.steps[0]["mock_result"] is None  # Uses global mock
    assert chain.steps[1]["mock_result"] == "Custom result for this step"


def test_chained_evaluator_no_mocks(test_mcp_server):
    """Test ChainedEvaluator without any mocks (will execute tools)."""
    from toolvaluator import get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)

    # Create evaluator without mocks
    chain = ChainedEvaluator(
        tool_schemas=tool_schemas,
        mcp_server=test_mcp_server,
        model_name="gpt-4o-mini",
        api_key="test-key",
    )

    chain.add_step(
        initial_query="Calculate 5 + 5",
        expected_tool="calculate",
        expected_arguments={"operation": "add", "a": 5, "b": 5},
    )

    assert len(chain.steps) == 1
    assert chain.mocks == {}
    assert chain.steps[0]["mock_result"] is None


def test_chained_evaluator_mixed_mocks_and_real_execution(test_mcp_server):
    """Test mixing mocked and unmocked tools in same chain."""
    from toolvaluator import get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)

    # Mock only calculate, leave search_docs to execute
    chain = ChainedEvaluator(
        tool_schemas=tool_schemas,
        mcp_server=test_mcp_server,
        model_name="gpt-4o-mini",
        api_key="test-key",
        mocks={"calculate": "100"},  # Only mock calculate
    )

    chain.add_step(
        initial_query="Calculate 10 * 10",
        expected_tool="calculate",
        expected_arguments={"operation": "multiply", "a": 10, "b": 10},
    )

    chain.add_step(
        expected_tool="search_docs",
        expected_arguments={"query": "test"},
        # No mock_result, will execute tool
    )

    assert len(chain.steps) == 2
    assert "calculate" in chain.mocks
    assert "search_docs" not in chain.mocks
