"""Pytest configuration and fixtures for toolvaluator tests."""

import pytest

from toolvaluator.test_server import mcp


@pytest.fixture
def test_mcp_server():
    """Provide the test MCP server instance."""
    return mcp


@pytest.fixture
def sample_tool_schema():
    """Provide a sample tool schema for testing."""
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
        },
        "required": ["query"],
    }
