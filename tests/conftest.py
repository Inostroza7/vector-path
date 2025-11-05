"""Pytest configuration and fixtures for vector_path tests."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_embedding_provider():
    """
    Mock EmbeddingProvider to avoid making real API calls during tests.

    Returns a mock provider that returns deterministic embeddings.
    """
    from vector_path.providers import EmbeddingProvider

    class MockProvider(EmbeddingProvider):
        """Mock provider with deterministic embeddings."""

        def __init__(self):
            self._model = "mock-model"
            self._dimension = 1536

        def get_embedding(self, text: str) -> np.ndarray:
            """Generate deterministic embedding based on text."""
            embedding = np.array([0.1] * self._dimension)
            # Vary first dimension by text length for diversity
            embedding[0] = len(text) / 100.0
            return embedding

        def get_embeddings_batch(self, texts: list) -> np.ndarray:
            """Generate batch embeddings."""
            return np.array([self.get_embedding(text) for text in texts])

        @property
        def embedding_dimension(self) -> int:
            return self._dimension

        @property
        def model_name(self) -> str:
            return f"mock/{self._model}"

    return MockProvider()


@pytest.fixture
def mock_azure_client(monkeypatch):
    """
    Mock Azure OpenAI client to avoid making real API calls during tests.

    Returns a mock client that returns deterministic embeddings.
    (Mantenido para compatibilidad con tests existentes)
    """
    mock_client = Mock()

    # Mock embedding response
    def mock_embeddings_create(input, model):
        """Create mock embeddings with deterministic values."""
        embeddings_data = []
        for i, text in enumerate(input if isinstance(input, list) else [input]):
            # Create a simple deterministic embedding based on text length
            # This ensures consistent test behavior
            embedding = [0.1] * 1536  # Standard embedding dimension
            embedding[0] = len(text) / 100.0  # Vary first dimension by length

            mock_data = Mock()
            mock_data.embedding = embedding
            embeddings_data.append(mock_data)

        mock_response = Mock()
        mock_response.data = embeddings_data
        return mock_response

    mock_client.embeddings.create = Mock(side_effect=mock_embeddings_create)

    return mock_client


@pytest.fixture
def sample_paths():
    """
    Sample paths for testing.

    Returns a list of Path objects representing different conversation topics.
    """
    from vector_path import Path

    return [
        Path(
            name="politics",
            utterances=[
                "what's your opinion on the president?",
                "thoughts on the government?",
                "what do you think about the election?"
            ],
            description="Political discussions"
        ),
        Path(
            name="chitchat",
            utterances=[
                "how's the weather today?",
                "what's up?",
                "how are you doing?"
            ],
            description="Casual conversation"
        ),
        Path(
            name="technical_support",
            utterances=[
                "my computer won't turn on",
                "I'm having issues with my software",
                "how do I reset my password?"
            ],
            description="Technical support queries"
        )
    ]


@pytest.fixture
def mock_vector_path(mock_embedding_provider, sample_paths):
    """
    Create a VectorPath instance with mocked embedding provider.

    This fixture is ready to use for testing without making real API calls.
    """
    from vector_path import VectorPath

    return VectorPath(
        paths=sample_paths,
        provider=mock_embedding_provider,
        score_threshold=0.7,
        top_k=3,
    )


@pytest.fixture
def mock_vector_path_legacy(mock_azure_client, sample_paths, monkeypatch):
    """
    Create a VectorPath instance with mocked Azure client (legacy method).

    Mantenido para tests de compatibilidad hacia atrás.
    """
    from vector_path import VectorPath

    # Patch AzureOpenAI to return our mock
    def mock_azure_openai_init(*args, **kwargs):
        return mock_azure_client

    monkeypatch.setattr("vector_path.providers.AzureOpenAI", mock_azure_openai_init)

    return VectorPath(
        paths=sample_paths,
        score_threshold=0.7,
        top_k=3,
        api_key="test_key",
        azure_endpoint="https://test.openai.azure.com/",
        api_version="2024-02-01"
    )


@pytest.fixture
def simple_embeddings():
    """
    Simple embeddings for testing similarity calculations.

    Returns a set of numpy arrays representing embeddings.
    """
    return {
        "vec1": np.array([1.0, 0.0, 0.0]),
        "vec2": np.array([0.0, 1.0, 0.0]),
        "vec3": np.array([1.0, 1.0, 0.0]) / np.sqrt(2),  # 45 degree angle
        "identical": np.array([1.0, 0.0, 0.0]),
    }
