"""Tests for the evaluator module."""

from toolvaluator.evaluator import (
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


def test_compare_arguments_no_expected():
    """Test comparing when no expected arguments."""
    expected = None
    predicted = {"arg1": "value1"}
    score, details = compare_arguments(expected, predicted)
    assert score == 1.0
    assert details["reason"] == "no_expected_args_specified"


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
