"""Tests for the core VectorPath functionality."""

import pytest
import numpy as np
from vector_path import VectorPath, Path, PathChoice, create_vector_path


class TestPath:
    """Tests for the Path dataclass."""

    def test_path_creation(self):
        """Test creating a valid Path."""
        path = Path(
            name="test",
            utterances=["hello", "hi"],
            description="Test path"
        )
        assert path.name == "test"
        assert len(path.utterances) == 2
        assert path.description == "Test path"

    def test_path_without_utterances_raises_error(self):
        """Test that Path requires at least one utterance."""
        with pytest.raises(ValueError, match="debe tener al menos una utterance"):
            Path(name="test", utterances=[])

    def test_path_with_metadata(self):
        """Test Path with custom metadata."""
        path = Path(
            name="test",
            utterances=["hello"],
            metadata={"priority": "high", "category": "support"}
        )
        assert path.metadata["priority"] == "high"
        assert path.metadata["category"] == "support"


class TestPathChoice:
    """Tests for the PathChoice dataclass."""

    def test_pathchoice_with_match(self):
        """Test PathChoice when a path matches."""
        choice = PathChoice(
            name="politics",
            score=0.85,
            scores={"politics": 0.85, "chitchat": 0.42}
        )
        assert choice.name == "politics"
        assert choice.score == 0.85
        assert len(choice.scores) == 2

    def test_pathchoice_without_match(self):
        """Test PathChoice when no path matches."""
        choice = PathChoice(
            name=None,
            score=0.65,
            scores={"politics": 0.65, "chitchat": 0.55}
        )
        assert choice.name is None
        assert choice.score == 0.65


class TestVectorPathInit:
    """Tests for VectorPath initialization."""

    def test_vectorpath_creation(self, mock_vector_path):
        """Test creating a VectorPath instance."""
        assert isinstance(mock_vector_path, VectorPath)
        assert len(mock_vector_path.paths) == 3
        assert mock_vector_path.score_threshold == 0.7
        assert mock_vector_path.top_k == 3

    def test_vectorpath_builds_index(self, mock_vector_path):
        """Test that initialization builds the embedding index."""
        # Should have embeddings for all utterances across all paths
        total_utterances = sum(len(path.utterances) for path in mock_vector_path.paths)
        assert len(mock_vector_path.embeddings) == total_utterances
        assert len(mock_vector_path.path_names) == total_utterances
        assert len(mock_vector_path.utterances) == total_utterances


class TestVectorPathRouting:
    """Tests for VectorPath routing functionality."""

    def test_call_returns_pathchoice(self, mock_vector_path):
        """Test that calling the router returns a PathChoice."""
        result = mock_vector_path("what's the weather like?")
        assert isinstance(result, PathChoice)

    def test_empty_router_returns_none(self, mock_embedding_provider):
        """Test routing with no paths returns None."""
        router = VectorPath(paths=[], provider=mock_embedding_provider)
        result = router("test query")

        assert result.name is None
        assert result.score == 0.0
        assert result.scores == {}

    def test_evaluate_query_verbose(self, mock_vector_path, capsys):
        """Test evaluate_query with verbose output."""
        result = mock_vector_path.evaluate_query("how's the weather?", verbose=True)

        captured = capsys.readouterr()
        assert "Query:" in captured.out
        assert "Scores por path:" in captured.out
        assert isinstance(result, PathChoice)


class TestVectorPathMethods:
    """Tests for VectorPath helper methods."""

    def test_cosine_similarity_identical_vectors(self, mock_vector_path):
        """Test cosine similarity with identical vectors."""
        vec = np.array([1.0, 0.0, 0.0])
        similarity = mock_vector_path._cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_orthogonal_vectors(self, mock_vector_path):
        """Test cosine similarity with orthogonal vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = mock_vector_path._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0, abs=1e-6)

    def test_cosine_similarity_opposite_vectors(self, mock_vector_path):
        """Test cosine similarity with opposite vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        similarity = mock_vector_path._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(-1.0, abs=1e-6)

    def test_cosine_similarity_zero_vector(self, mock_vector_path):
        """Test cosine similarity with zero vector."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        similarity = mock_vector_path._cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_add_path(self, mock_vector_path):
        """Test adding a new path dynamically."""
        initial_path_count = len(mock_vector_path.paths)

        new_path = Path(
            name="sales",
            utterances=["pricing", "buy now", "purchase"]
        )
        mock_vector_path.add_path(new_path)

        assert len(mock_vector_path.paths) == initial_path_count + 1
        assert "sales" in [p.name for p in mock_vector_path.paths]

    def test_get_path_info(self, mock_vector_path, capsys):
        """Test get_path_info returns formatted information."""
        info = mock_vector_path.get_path_info()

        assert "VectorPath con" in info
        assert "paths:" in info
        assert "Proveedor:" in info  # Changed from "Modelo:"
        assert "Threshold:" in info
        assert "Top-K:" in info


class TestAggregateScores:
    """Tests for score aggregation logic."""

    def test_aggregate_scores_single_path(self, mock_vector_path):
        """Test score aggregation with a single path."""
        # Mock similarities for testing
        similarities = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        mock_vector_path.path_names = ["test"] * 5

        aggregated = mock_vector_path._aggregate_scores(similarities)

        assert "test" in aggregated
        # Should average top_k (3) scores: (0.9 + 0.8 + 0.7) / 3
        assert aggregated["test"] == pytest.approx(0.8, abs=0.01)

    def test_aggregate_scores_multiple_paths(self, mock_vector_path):
        """Test score aggregation with multiple paths."""
        similarities = np.array([0.9, 0.5, 0.8, 0.4, 0.7, 0.3])
        mock_vector_path.path_names = ["path1", "path2"] * 3

        aggregated = mock_vector_path._aggregate_scores(similarities)

        assert "path1" in aggregated
        assert "path2" in aggregated


class TestCreateVectorPath:
    """Tests for the create_vector_path helper function."""

    def test_create_vector_path_helper(self, sample_paths, mock_embedding_provider):
        """Test the create_vector_path helper function."""
        router = create_vector_path(
            paths=sample_paths,
            provider=mock_embedding_provider,
            score_threshold=0.8,
        )

        assert isinstance(router, VectorPath)
        assert router.score_threshold == 0.8
        assert router.provider == mock_embedding_provider
        assert len(router.paths) == 3
