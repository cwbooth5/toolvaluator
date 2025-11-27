"""Tests for the test MCP server."""

import pytest

from toolvaluator.test_server import (
    _calculate_impl as calculate,
)
from toolvaluator.test_server import (
    _create_task_impl as create_task,
)
from toolvaluator.test_server import (
    _get_weather_impl as get_weather,
)
from toolvaluator.test_server import (
    _search_docs_impl as search_docs,
)
from toolvaluator.test_server import (
    _send_email_impl as send_email,
)


def test_search_docs():
    """Test the search_docs tool."""
    result = search_docs("test query")
    assert isinstance(result, str)
    assert "test query" in result


def test_get_weather():
    """Test the get_weather tool."""
    result = get_weather("San Francisco")
    assert isinstance(result, dict)
    assert result["location"] == "San Francisco"
    assert "temperature" in result
    assert result["units"] == "celsius"


def test_get_weather_custom_units():
    """Test the get_weather tool with custom units."""
    result = get_weather("New York", units="fahrenheit")
    assert result["location"] == "New York"
    assert result["units"] == "fahrenheit"


def test_calculate_add():
    """Test the calculate tool with addition."""
    result = calculate("add", 5, 3)
    assert result == 8


def test_calculate_subtract():
    """Test the calculate tool with subtraction."""
    result = calculate("subtract", 10, 4)
    assert result == 6


def test_calculate_multiply():
    """Test the calculate tool with multiplication."""
    result = calculate("multiply", 6, 7)
    assert result == 42


def test_calculate_divide():
    """Test the calculate tool with division."""
    result = calculate("divide", 20, 4)
    assert result == 5


def test_calculate_divide_by_zero():
    """Test the calculate tool with division by zero."""
    result = calculate("divide", 10, 0)
    assert result == float("inf")


def test_calculate_invalid_operation():
    """Test the calculate tool with invalid operation."""
    with pytest.raises(ValueError, match="Unknown operation"):
        calculate("invalid", 1, 2)


def test_send_email():
    """Test the send_email tool."""
    result = send_email(to="test@example.com", subject="Test Subject", body="Test body")
    assert isinstance(result, dict)
    assert result["status"] == "sent"
    assert result["to"] == "test@example.com"
    assert result["subject"] == "Test Subject"
    assert result["cc"] == []


def test_send_email_with_cc():
    """Test the send_email tool with CC recipients."""
    result = send_email(
        to="test@example.com",
        subject="Test",
        body="Body",
        cc=["cc1@example.com", "cc2@example.com"],
    )
    assert result["cc"] == ["cc1@example.com", "cc2@example.com"]


def test_create_task():
    """Test the create_task tool."""
    result = create_task("Test task")
    assert isinstance(result, dict)
    assert result["title"] == "Test task"
    assert result["priority"] == "medium"
    assert result["status"] == "open"


def test_create_task_with_all_params():
    """Test the create_task tool with all parameters."""
    result = create_task(
        title="Important task",
        description="This is important",
        priority="high",
        due_date="2025-12-31",
    )
    assert result["title"] == "Important task"
    assert result["description"] == "This is important"
    assert result["priority"] == "high"
    assert result["due_date"] == "2025-12-31"
