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


def test_build_dataset(test_mcp_server):
    """Test build_dataset creates examples for available tools."""
    from toolvaluator import build_dataset, get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    dataset = build_dataset(tool_schemas)

    # Should create multiple examples
    assert len(dataset) > 0

    # All examples should have required fields
    for example in dataset:
        assert hasattr(example, "user_query")
        assert hasattr(example, "tool_name")
        assert hasattr(example, "expected_should_call")


def test_chained_example_builder_basic(test_mcp_server):
    """Test ChainedExampleBuilder basic functionality."""
    from toolvaluator import ChainedExampleBuilder, get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    builder = ChainedExampleBuilder(tool_schemas, test_mcp_server)

    builder.add_chain(mocks={"calculate": "50"}).add_step(
        initial_query="Calculate 5 * 10",
        expected_tool="calculate",
        expected_arguments={"operation": "multiply", "a": 5, "b": 10},
    )

    dataset = builder.build()
    assert len(dataset) == 1
    assert len(dataset[0]["steps"]) == 1
    assert dataset[0]["mocks"]["calculate"] == "50"


def test_chained_example_builder_multiple_chains(test_mcp_server):
    """Test ChainedExampleBuilder with multiple chains."""
    from toolvaluator import ChainedExampleBuilder, get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    builder = ChainedExampleBuilder(tool_schemas, test_mcp_server)

    # First chain
    builder.add_chain().add_step(
        initial_query="Query 1",
        expected_tool="calculate",
        expected_arguments={"operation": "add", "a": 1, "b": 2},
    )

    # Second chain
    builder.add_chain().add_step(
        initial_query="Query 2",
        expected_tool="search_docs",
        expected_arguments={"query": "test"},
    )

    dataset = builder.build()
    assert len(dataset) == 2
    assert len(dataset[0]["steps"]) == 1
    assert len(dataset[1]["steps"]) == 1


def test_chained_example_builder_requires_add_chain_first(test_mcp_server):
    """Test ChainedExampleBuilder raises error if add_step called before add_chain."""
    import pytest

    from toolvaluator import ChainedExampleBuilder, get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    builder = ChainedExampleBuilder(tool_schemas, test_mcp_server)

    with pytest.raises(ValueError, match="Must call add_chain"):
        builder.add_step(
            initial_query="test", expected_tool="calculate", expected_arguments={}
        )


def test_chained_example_builder_invalid_tool(test_mcp_server):
    """Test ChainedExampleBuilder raises error for invalid tool."""
    import pytest

    from toolvaluator import ChainedExampleBuilder, get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    builder = ChainedExampleBuilder(tool_schemas, test_mcp_server)

    builder.add_chain()
    with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
        builder.add_step(
            initial_query="test", expected_tool="nonexistent", expected_arguments={}
        )


def test_example_builder_with_system_prompt(test_mcp_server):
    """Test ExampleBuilder with system prompt."""
    from toolvaluator import ExampleBuilder, get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    builder = ExampleBuilder(tool_schemas)

    builder.add_positive(
        tool="search_docs",
        query="Find documentation",
        arguments={"query": "test"},
        system_prompt="You are a helpful assistant.",
    )

    assert len(builder.examples) == 1
    example = builder.examples[0]
    assert hasattr(example, "system_prompt")
    assert example.system_prompt == "You are a helpful assistant."


def test_chained_evaluator_add_step_invalid_tool(test_mcp_server):
    """Test ChainedEvaluator.add_step validates tool existence."""
    import pytest

    from toolvaluator import ChainedEvaluator, get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    chain = ChainedEvaluator(
        tool_schemas=tool_schemas,
        mcp_server=test_mcp_server,
        model_name="gpt-4o-mini",
        api_key="test-key",
    )

    with pytest.raises(ValueError, match="Tool 'invalid_tool' not found"):
        chain.add_step(
            initial_query="test", expected_tool="invalid_tool", expected_arguments={}
        )


def test_chained_evaluator_evaluate_no_steps(test_mcp_server):
    """Test ChainedEvaluator.evaluate raises error if no steps."""
    import pytest

    from toolvaluator import ChainedEvaluator, get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    chain = ChainedEvaluator(
        tool_schemas=tool_schemas,
        mcp_server=test_mcp_server,
        model_name="gpt-4o-mini",
        api_key="test-key",
    )

    with pytest.raises(ValueError, match="No steps defined"):
        chain.evaluate()


def test_chained_evaluator_evaluate_missing_initial_query(test_mcp_server):
    """Test ChainedEvaluator.evaluate raises error if first step has no initial_query."""
    import pytest

    from toolvaluator import ChainedEvaluator, get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    chain = ChainedEvaluator(
        tool_schemas=tool_schemas,
        mcp_server=test_mcp_server,
        model_name="gpt-4o-mini",
        api_key="test-key",
    )

    chain.add_step(
        expected_tool="calculate",
        expected_arguments={"operation": "add", "a": 1, "b": 2},
    )

    with pytest.raises(ValueError, match="First step must have initial_query"):
        chain.evaluate()


def test_generic_tool_caller_module_with_system_prompt():
    """Test GenericToolCallerModule prepends system prompt."""
    from unittest.mock import Mock, patch

    from toolvaluator.core import GenericToolCallerModule

    # Mock the DSPy Predict class
    with patch("toolvaluator.core.dspy.Predict") as MockPredict:
        mock_predict_instance = Mock()
        mock_result = Mock()
        mock_result.arguments_json = '{"arg": "value"}'
        mock_result.should_call = True
        mock_predict_instance.return_value = mock_result
        MockPredict.return_value = mock_predict_instance

        module = GenericToolCallerModule()

        # Call with system prompt
        module.forward(
            user_query="What is the weather?",
            tool_name="get_weather",
            tool_description="Get weather info",
            tool_schema={"type": "object", "properties": {}},
            system_prompt="You are a weather assistant.",
        )

        # Verify system prompt was prepended
        call_args = mock_predict_instance.call_args
        assert call_args is not None
        assert "You are a weather assistant." in call_args[1]["user_query"]
        assert "What is the weather?" in call_args[1]["user_query"]


def test_generic_tool_caller_module_without_system_prompt():
    """Test GenericToolCallerModule works without system prompt."""
    from unittest.mock import Mock, patch

    from toolvaluator.core import GenericToolCallerModule

    with patch("toolvaluator.core.dspy.Predict") as MockPredict:
        mock_predict_instance = Mock()
        mock_result = Mock()
        mock_result.arguments_json = '{"arg": "value"}'
        mock_result.should_call = True
        mock_predict_instance.return_value = mock_result
        MockPredict.return_value = mock_predict_instance

        module = GenericToolCallerModule()

        module.forward(
            user_query="What is the weather?",
            tool_name="get_weather",
            tool_description="Get weather info",
            tool_schema={"type": "object", "properties": {}},
        )

        # Verify query is used as-is
        call_args = mock_predict_instance.call_args
        assert call_args is not None
        assert call_args[1]["user_query"] == "What is the weather?"


def test_generic_tool_caller_module_invalid_json():
    """Test GenericToolCallerModule handles invalid JSON in arguments."""
    from unittest.mock import Mock, patch

    from toolvaluator.core import GenericToolCallerModule

    with patch("toolvaluator.core.dspy.Predict") as MockPredict:
        mock_predict_instance = Mock()
        mock_result = Mock()
        mock_result.arguments_json = "invalid json{"
        mock_result.should_call = True
        mock_predict_instance.return_value = mock_result
        MockPredict.return_value = mock_predict_instance

        module = GenericToolCallerModule()

        result = module.forward(
            user_query="test",
            tool_name="test_tool",
            tool_description="test",
            tool_schema={},
        )

        # Should return empty dict for invalid JSON
        assert result.arguments == {}


def test_tool_call_metric_should_call_mismatch():
    """Test tool_call_metric when should_call doesn't match."""
    from unittest.mock import Mock

    import dspy

    from toolvaluator.core import tool_call_metric

    example = dspy.Example(
        expected_should_call=True, expected_tool_name="test_tool", expected_arguments={}
    )

    pred = Mock()
    pred.should_call = False
    pred.tool_name = "test_tool"
    pred.arguments = {}
    pred.latency_ms = 100

    score = tool_call_metric(example, pred)

    # should_call wrong (0), tool_name correct (1), args correct (1) = 2/3
    assert abs(score - 0.666) < 0.01


def test_tool_call_metric_tool_name_mismatch():
    """Test tool_call_metric when tool name doesn't match."""
    from unittest.mock import Mock

    import dspy

    from toolvaluator.core import tool_call_metric

    example = dspy.Example(
        expected_should_call=True,
        expected_tool_name="correct_tool",
        expected_arguments={},
    )

    pred = Mock()
    pred.should_call = True
    pred.tool_name = "wrong_tool"
    pred.arguments = {}
    pred.latency_ms = 100

    score = tool_call_metric(example, pred)

    # should_call correct (1), tool_name wrong (0), args correct (1) = 2/3
    assert abs(score - 0.666) < 0.01


def test_extract_tool_description_from_object():
    """Test extract_tool_description with object-like tool def."""
    from toolvaluator.core import extract_tool_description

    class ToolDef:
        description = "Test description"

    tool_def = ToolDef()
    desc = extract_tool_description(tool_def)
    assert desc == "Test description"


def test_extract_tool_description_missing():
    """Test extract_tool_description when description is missing."""
    from toolvaluator.core import extract_tool_description

    tool_def = {"name": "test"}
    desc = extract_tool_description(tool_def)
    assert desc == ""


def test_extract_input_schema_from_object():
    """Test extract_input_schema with object-like tool def."""
    from toolvaluator.core import extract_input_schema

    class ToolDef:
        inputSchema = {"type": "object"}

    tool_def = ToolDef()
    schema = extract_input_schema(tool_def)
    assert schema == {"type": "object"}


def test_eval_model_basic(test_mcp_server):
    """Test eval_model with mocked DSPy LM."""
    from unittest.mock import MagicMock, patch

    from toolvaluator import ExampleBuilder, eval_model, get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    builder = ExampleBuilder(tool_schemas)

    builder.add_positive(
        tool="calculate",
        query="What is 5 + 10?",
        arguments={"operation": "add", "a": 5, "b": 10},
    )

    dataset = builder.build()

    # Mock DSPy LM and predictions
    with (
        patch("toolvaluator.evaluation.dspy.LM") as MockLM,
        patch("toolvaluator.evaluation.dspy.configure"),
    ):
        mock_lm = MagicMock()
        MockLM.return_value = mock_lm

        with patch("toolvaluator.core.dspy.Predict") as MockPredict:
            mock_predict = MagicMock()
            mock_result = MagicMock()
            mock_result.should_call = True
            mock_result.arguments_json = '{"operation": "add", "a": 5, "b": 10}'
            mock_result.arguments = {"operation": "add", "a": 5, "b": 10}
            mock_result.tool_name = "calculate"
            mock_predict.return_value = mock_result
            MockPredict.return_value = mock_predict

            result = eval_model(
                model_name="gpt-4o-mini",
                api_key="test-key",
                base_url=None,
                dataset=dataset,
                verbose=False,
            )

            # Verify result structure
            assert "score" in result
            assert "latencies" in result
            assert "predictions" in result
            assert "scores" in result
            assert len(result["predictions"]) == 1


def test_eval_model_with_base_url(test_mcp_server):
    """Test eval_model with custom base_url."""
    from unittest.mock import MagicMock, patch

    from toolvaluator import ExampleBuilder, eval_model, get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    builder = ExampleBuilder(tool_schemas)
    builder.add_positive(
        tool="calculate",
        query="What is 2 * 3?",
        arguments={"operation": "multiply", "a": 2, "b": 3},
    )
    dataset = builder.build()

    with (
        patch("toolvaluator.evaluation.dspy.LM") as MockLM,
        patch("toolvaluator.evaluation.dspy.configure"),
    ):
        mock_lm = MagicMock()
        MockLM.return_value = mock_lm

        with patch("toolvaluator.core.dspy.Predict") as MockPredict:
            mock_predict = MagicMock()
            mock_result = MagicMock()
            mock_result.should_call = True
            mock_result.arguments_json = '{"operation": "multiply", "a": 2, "b": 3}'
            mock_result.arguments = {"operation": "multiply", "a": 2, "b": 3}
            mock_result.tool_name = "calculate"
            mock_predict.return_value = mock_result
            MockPredict.return_value = mock_predict

            eval_model(
                model_name="custom-model",
                api_key="test-key",
                base_url="http://localhost:8000",
                dataset=dataset,
            )

            # Should add openai/ prefix when base_url is provided
            MockLM.assert_called_once()
            call_args = MockLM.call_args
            assert "openai/custom-model" in str(call_args)


def test_eval_model_with_system_prompt(test_mcp_server):
    """Test eval_model respects system prompts at different levels."""
    from unittest.mock import MagicMock, patch

    from toolvaluator import ExampleBuilder, eval_model, get_tool_schemas_sync

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    builder = ExampleBuilder(tool_schemas)

    # Example with its own system prompt
    builder.add_positive(
        tool="calculate",
        query="Add 1 and 2",
        arguments={"operation": "add", "a": 1, "b": 2},
        system_prompt="Example-level prompt",
    )

    # Example without system prompt (will use eval-level)
    builder.add_positive(
        tool="calculate",
        query="Add 3 and 4",
        arguments={"operation": "add", "a": 3, "b": 4},
    )

    dataset = builder.build()

    with (
        patch("toolvaluator.evaluation.dspy.LM"),
        patch("toolvaluator.evaluation.dspy.configure"),
    ):
        with patch("toolvaluator.core.dspy.Predict") as MockPredict:
            mock_predict = MagicMock()
            mock_result = MagicMock()
            mock_result.should_call = True
            mock_result.arguments_json = '{"operation": "add", "a": 1, "b": 2}'
            mock_result.arguments = {"operation": "add", "a": 1, "b": 2}
            mock_result.tool_name = "calculate"
            mock_predict.return_value = mock_result
            MockPredict.return_value = mock_predict

            eval_model(
                model_name="gpt-4o-mini",
                api_key="test-key",
                base_url=None,
                dataset=dataset,
                system_prompt="Eval-level prompt",
            )

            # Verify calls were made with system prompts
            assert mock_predict.call_count == 2


def test_eval_chained_model_basic(test_mcp_server):
    """Test eval_chained_model with mocked DSPy LM."""
    from unittest.mock import MagicMock, patch

    from toolvaluator import (
        ChainedExampleBuilder,
        eval_chained_model,
        get_tool_schemas_sync,
    )

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    builder = ChainedExampleBuilder(tool_schemas, test_mcp_server)

    builder.add_chain(mocks={"calculate": "15"}).add_step(
        initial_query="Calculate 5 + 10",
        expected_tool="calculate",
        expected_arguments={"operation": "add", "a": 5, "b": 10},
    )

    dataset = builder.build()

    with (
        patch("toolvaluator.chained.dspy.LM") as MockLM,
        patch("toolvaluator.chained.dspy.configure"),
    ):
        mock_lm = MagicMock()
        MockLM.return_value = mock_lm

        with patch("toolvaluator.core.dspy.Predict") as MockPredict:
            mock_predict = MagicMock()
            mock_result = MagicMock()
            mock_result.should_call = True
            mock_result.arguments_json = '{"operation": "add", "a": 5, "b": 10}'
            mock_result.arguments = {"operation": "add", "a": 5, "b": 10}
            mock_result.tool_name = "calculate"
            mock_predict.return_value = mock_result
            MockPredict.return_value = mock_predict

            result = eval_chained_model(
                model_name="gpt-4o-mini",
                api_key="test-key",
                dataset=dataset,
            )

            # Verify result structure
            assert "score" in result
            assert "chain_results" in result
            assert "num_chains" in result
            assert "num_steps" in result
            assert result["num_chains"] == 1
            assert result["num_steps"] == 1


def test_eval_chained_model_empty_dataset():
    """Test eval_chained_model raises error for empty dataset."""
    import pytest

    from toolvaluator import eval_chained_model

    with pytest.raises(ValueError, match="Dataset is empty"):
        eval_chained_model(
            model_name="gpt-4o-mini",
            api_key="test-key",
            dataset=[],
        )


def test_eval_chained_model_with_system_prompt(test_mcp_server):
    """Test eval_chained_model with system prompts."""
    from unittest.mock import MagicMock, patch

    from toolvaluator import (
        ChainedExampleBuilder,
        eval_chained_model,
        get_tool_schemas_sync,
    )

    tool_schemas = get_tool_schemas_sync(test_mcp_server)
    builder = ChainedExampleBuilder(tool_schemas, test_mcp_server)

    builder.add_chain(mocks={"calculate": "10"}).add_step(
        initial_query="Calculate 5 + 5",
        expected_tool="calculate",
        expected_arguments={"operation": "add", "a": 5, "b": 5},
        system_prompt="Step-level prompt",
    )

    dataset = builder.build()

    with (
        patch("toolvaluator.chained.dspy.LM"),
        patch("toolvaluator.chained.dspy.configure"),
    ):
        with patch("toolvaluator.core.dspy.Predict") as MockPredict:
            mock_predict = MagicMock()
            mock_result = MagicMock()
            mock_result.should_call = True
            mock_result.arguments_json = '{"operation": "add", "a": 5, "b": 5}'
            mock_result.arguments = {"operation": "add", "a": 5, "b": 5}
            mock_result.tool_name = "calculate"
            mock_predict.return_value = mock_result
            MockPredict.return_value = mock_predict

            result = eval_chained_model(
                model_name="gpt-4o-mini",
                api_key="test-key",
                dataset=dataset,
                system_prompt="Eval-level prompt",
            )

            assert result["num_chains"] == 1
