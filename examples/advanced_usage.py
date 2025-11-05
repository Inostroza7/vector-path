"""
Advanced usage example for vector-path.

This example demonstrates:
- Dynamic path addition
- Custom configuration
- Handling edge cases
- Integration patterns
"""

import os
from vector_path import VectorPath, Path, create_vector_path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def example_dynamic_paths():
    """Demonstrate adding paths dynamically."""
    print("\n" + "="*60)
    print("Example 1: Dynamic Path Addition")
    print("="*60)

    # Start with minimal paths
    initial_paths = [
        Path(
            name="greeting",
            utterances=["hello", "hi", "hey"]
        ),
        Path(
            name="farewell",
            utterances=["goodbye", "bye", "see you"]
        )
    ]

    router = VectorPath(paths=initial_paths, score_threshold=0.7)
    print(f"Initial paths: {[p.name for p in router.paths]}")

    # Test before adding new path
    result = router("I need help with billing")
    print(f"\nQuery: 'I need help with billing'")
    print(f"Result: {result.name} (no match expected)")

    # Dynamically add a new path
    print("\n➕ Adding 'billing' path...")
    billing_path = Path(
        name="billing",
        utterances=[
            "I need help with billing",
            "invoice question",
            "payment issue",
            "subscription cost"
        ]
    )
    router.add_path(billing_path)

    # Test after adding new path
    result = router("I need help with billing")
    print(f"\nQuery: 'I need help with billing'")
    print(f"Result: {result.name} (score: {result.score:.4f})")


def example_custom_configuration():
    """Demonstrate custom configuration options."""
    print("\n" + "="*60)
    print("Example 2: Custom Configuration")
    print("="*60)

    paths = [
        Path(name="urgent", utterances=["emergency", "urgent", "critical"]),
        Path(name="normal", utterances=["question", "help", "inquiry"])
    ]

    # VectorPath with strict threshold
    strict_router = VectorPath(
        paths=paths,
        score_threshold=0.85,  # High threshold - fewer matches
        top_k=3,
        model="text-embedding-3-small"
    )

    # VectorPath with lenient threshold
    lenient_router = VectorPath(
        paths=paths,
        score_threshold=0.6,   # Low threshold - more matches
        top_k=3,
        model="text-embedding-3-small"
    )

    test_query = "I have a problem"

    print(f"\nQuery: '{test_query}'")
    print("\nStrict VectorPath (threshold=0.85):")
    strict_result = strict_router(test_query)
    print(f"  Result: {strict_result.name or 'No match'} (score: {strict_result.score:.4f})")

    print("\nLenient VectorPath (threshold=0.6):")
    lenient_result = lenient_router(test_query)
    print(f"  Result: {lenient_result.name or 'No match'} (score: {lenient_result.score:.4f})")


def example_handling_edge_cases():
    """Demonstrate handling edge cases."""
    print("\n" + "="*60)
    print("Example 3: Handling Edge Cases")
    print("="*60)

    paths = [
        Path(name="code", utterances=["python", "javascript", "programming"]),
        Path(name="design", utterances=["UI", "UX", "interface"])
    ]

    router = VectorPath(paths=paths, score_threshold=0.7)

    # Test various edge cases
    edge_cases = [
        "",  # Empty string
        "xyzabc123",  # Nonsense
        "I want to learn about quantum physics",  # Off-topic
        "python programming",  # Should match "code"
    ]

    for query in edge_cases:
        result = router(query)
        match_status = "✅ MATCH" if result.name else "❌ NO MATCH"
        print(f"\n{match_status}")
        print(f"  Query: '{query}'")
        print(f"  Path: {result.name or 'None'}")
        print(f"  Score: {result.score:.4f}")


def example_helper_function():
    """Demonstrate the create_vector_path helper."""
    print("\n" + "="*60)
    print("Example 4: Using Helper Function")
    print("="*60)

    paths = [
        Path(name="sales", utterances=["pricing", "buy", "purchase"]),
        Path(name="support", utterances=["help", "issue", "problem"])
    ]

    # Quick VectorPath creation with helper function
    router = create_vector_path(
        paths=paths,
        score_threshold=0.75,
        model="text-embedding-3-small"
    )

    print("VectorPath created with helper function:")
    print(router.get_path_info())

    # Test routing
    result = router.evaluate_query("I want to buy your product", verbose=True)


def example_metadata_usage():
    """Demonstrate using path metadata."""
    print("\n" + "="*60)
    print("Example 5: Path Metadata")
    print("="*60)

    paths = [
        Path(
            name="vip_support",
            utterances=["VIP help", "priority support", "urgent assistance"],
            description="High-priority support for VIP customers",
            metadata={"priority": "high", "sla": "1 hour", "tier": "premium"}
        ),
        Path(
            name="standard_support",
            utterances=["help needed", "question", "inquiry"],
            description="Standard support",
            metadata={"priority": "normal", "sla": "24 hours", "tier": "standard"}
        )
    ]

    router = VectorPath(paths=paths, score_threshold=0.7)

    query = "I need urgent help"
    result = router(query)

    if result.name:
        # Find the matched path to access metadata
        matched_path = next(p for p in router.paths if p.name == result.name)
        print(f"\nQuery: '{query}'")
        print(f"Matched Path: {matched_path.name}")
        print(f"Description: {matched_path.description}")
        print(f"Metadata:")
        for key, value in matched_path.metadata.items():
            print(f"  - {key}: {value}")


def main():
    """Run all advanced examples."""
    print("\n" + "🚀 " + "="*58)
    print("    Vector-Path: Advanced Usage Examples")
    print("="*60)

    try:
        example_dynamic_paths()
        example_custom_configuration()
        example_handling_edge_cases()
        example_helper_function()
        example_metadata_usage()

        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have set up your .env file with Azure OpenAI credentials.")


if __name__ == "__main__":
    main()
