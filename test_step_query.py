"""
Demonstration of per-step queries in chained evaluation.
This shows how initial_query can be used for steps after the first one.
"""

from toolvaluator import ChainedEvaluator, get_tool_schemas_sync
from toolvaluator.test_server import mcp


def test_step_queries():
    """Test that per-step queries are properly sent to the model."""
    tool_schemas = get_tool_schemas_sync(mcp)

    # Create a chain with specific queries for each step
    chain = ChainedEvaluator(
        tool_schemas=tool_schemas,
        mcp_server=mcp,
        model_name="openai/gpt-4o-mini",
        api_key="test-key",
        mocks={
            "get_location": "San Francisco",
            "get_weather": "Sunny, 72F",
        },
        verbose=True,
    )

    # Step 1: Get location
    chain.add_step(
        initial_query="What's my location?",
        expected_tool="get_location",
        expected_arguments={},
    )

    # Step 2: Get weather with a SPECIFIC query for this step
    # This query will now be included along with the context
    chain.add_step(
        initial_query="Now get the weather forecast for the next 5 days",
        expected_tool="get_weather",
        expected_arguments={"location": None},  # Wildcard - any location works
    )

    print("=" * 70)
    print("Testing per-step queries in chained evaluation")
    print("=" * 70)
    print()
    print("Expected behavior:")
    print("- Step 1: Uses 'What's my location?' as query")
    print("- Step 2: Should receive:")
    print("  1. Previous context (location result)")
    print("  2. The step-specific query: 'Now get the weather forecast...'")
    print()
    print("This ensures the model gets both context AND step-specific instructions.")
    print()

    # Run evaluation (this will print queries sent to model due to verbose=True)
    result = chain.evaluate()

    print()
    print("=" * 70)
    print("Result:")
    print(f"Overall score: {result['score']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    test_step_queries()
