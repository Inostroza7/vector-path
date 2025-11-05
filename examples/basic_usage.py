"""
Basic usage example for vector-path.

This example demonstrates how to create a simple semantic path
with predefined paths and route user queries.
"""

import os
from vector_path import VectorPath, Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def main():
    """Demonstrate basic vector-path usage."""

    # Define paths with example utterances
    paths = [
        Path(
            name="politics",
            utterances=[
                "what's your opinion on the president?",
                "thoughts on the government?",
                "what do you think about the election?",
                "who should I vote for?"
            ],
            description="Political discussions and questions"
        ),
        Path(
            name="chitchat",
            utterances=[
                "how's the weather today?",
                "what's up?",
                "how are you doing?",
                "nice to meet you",
                "good morning"
            ],
            description="Casual conversation and greetings"
        ),
        Path(
            name="technical_support",
            utterances=[
                "my computer won't turn on",
                "I'm having issues with my software",
                "how do I reset my password?",
                "the app keeps crashing",
                "can't connect to the internet"
            ],
            description="Technical support and troubleshooting"
        )
    ]

    # Create the semantic path
    print("🔨 Creating semantic VectorPath...")
    router = VectorPath(
        paths=paths,
        score_threshold=0.7,  # Minimum similarity score to match a path
        top_k=5,              # Consider top 5 most similar utterances
        model="text-embedding-3-small"
    )

    # Display VectorPath information
    print(router.get_path_info())

    # Test queries
    test_queries = [
        "hey there, how are you?",
        "what do you think about the new tax policy?",
        "my laptop won't boot up",
        "tell me a joke",  # This might not match any path
    ]

    print("\n" + "="*60)
    print("Testing semantic routing:")
    print("="*60)

    for query in test_queries:
        result = router.evaluate_query(query, verbose=True)
        print("-" * 60)

    # Example: Check if a specific path was selected
    query = "good morning!"
    result = router(query)

    if result.name:
        print(f"\n✅ Query '{query}' matched path: {result.name}")
        print(f"   Confidence score: {result.score:.4f}")
    else:
        print(f"\n❌ Query '{query}' didn't match any path")
        print(f"   Best score: {result.score:.4f} (below threshold: {router.score_threshold})")


if __name__ == "__main__":
    main()
