"""
Test FastMCP server for evaluating the toolvaluator framework.

This server provides example tools that can be used to test the evaluation
framework itself.
"""

from fastmcp import FastMCP

mcp = FastMCP("Test Server")


# Implementation functions (testable)
def _search_docs_impl(query: str) -> str:
    """Implementation of search_docs."""
    return f"Found documents matching '{query}'"


def _get_weather_impl(location: str, units: str = "celsius") -> dict:
    """Implementation of get_weather."""
    return {
        "location": location,
        "temperature": 22,
        "units": units,
        "conditions": "partly cloudy",
    }


def _calculate_impl(operation: str, a: float, b: float) -> float:
    """Implementation of calculate."""
    ops = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float("inf"),
    }
    if operation not in ops:
        raise ValueError(f"Unknown operation: {operation}")
    return ops[operation](a, b)


def _send_email_impl(to: str, subject: str, body: str, cc: list[str] = None) -> dict:  # noqa: ARG001
    """Implementation of send_email."""
    return {
        "status": "sent",
        "to": to,
        "subject": subject,
        "cc": cc or [],
        "message_id": "test-message-123",
    }


def _create_task_impl(
    title: str, description: str = "", priority: str = "medium", due_date: str = None
) -> dict:
    """Implementation of create_task."""
    return {
        "id": "task-456",
        "title": title,
        "description": description,
        "priority": priority,
        "due_date": due_date,
        "status": "open",
    }


# MCP tool wrappers
@mcp.tool()
def search_docs(query: str) -> str:
    """
    Search through company documentation.

    Args:
        query: The search query to find relevant documents

    Returns:
        A summary of matching documents
    """
    return _search_docs_impl(query)


@mcp.tool()
def get_weather(location: str, units: str = "celsius") -> dict:
    """
    Get current weather for a location.

    Args:
        location: City name or location to get weather for
        units: Temperature units ('celsius' or 'fahrenheit')

    Returns:
        Weather information including temperature and conditions
    """
    return _get_weather_impl(location, units)


@mcp.tool()
def calculate(operation: str, a: float, b: float) -> float:
    """
    Perform a mathematical calculation.

    Args:
        operation: The operation to perform ('add', 'subtract', 'multiply', 'divide')
        a: First number
        b: Second number

    Returns:
        Result of the calculation
    """
    return _calculate_impl(operation, a, b)


@mcp.tool()
def send_email(to: str, subject: str, body: str, cc: list[str] = None) -> dict:
    """
    Send an email message.

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Email body content
        cc: Optional list of CC recipients

    Returns:
        Status of the email send operation
    """
    return _send_email_impl(to, subject, body, cc)


@mcp.tool()
def create_task(
    title: str, description: str = "", priority: str = "medium", due_date: str = None
) -> dict:
    """
    Create a new task in the task management system.

    Args:
        title: Task title
        description: Detailed task description
        priority: Priority level ('low', 'medium', 'high')
        due_date: Due date in YYYY-MM-DD format

    Returns:
        The created task with its ID
    """
    return _create_task_impl(title, description, priority, due_date)


if __name__ == "__main__":
    # This allows running the server standalone for testing
    mcp.run()
