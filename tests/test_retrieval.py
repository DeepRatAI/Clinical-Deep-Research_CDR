"""
Tests for CDR Retrieval Layer.

Tests for PubMed, ClinicalTrials.gov clients, embeddings, and retrieval components.

Rewritten for public API + deterministic mocks per ADR-005.

NCBI Rate Limits (for CI/documentation):
-----------------------------------------
All NCBI E-utilities calls in this module are MOCKED using unittest.mock.patch
to avoid network calls during testing. This ensures:

1. CI environments don't require NCBI API keys
2. Tests don't hit NCBI rate limits (3 req/sec without key, 10 req/sec with key)
3. Tests are deterministic and reproducible

For production usage, configure:
- NCBI_API_KEY: Increases rate limit from 3 to 10 requests/second
- NCBI_EMAIL: Required for identification per NCBI guidelines

Reference: https://www.ncbi.nlm.nih.gov/books/NBK25497/#chapter2.Usage_Guidelines_and_Requiremen
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from cdr.core.enums import RecordSource
from cdr.core.schemas import Record


# =============================================================================
# PUBMED CLIENT TESTS
# =============================================================================


class TestPubMedClient:
    """Tests for PubMedClient using public API."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for PubMed client."""
        mock = MagicMock()
        mock.retrieval.ncbi_api_key = "test_api_key"
        mock.retrieval.ncbi_email = "test@example.com"
        mock.retrieval.pubmed_max_results = 100
        return mock

    @pytest.fixture
    def client(self, mock_settings):
        """Create PubMed client instance with mocked settings."""
        from cdr.retrieval.pubmed_client import PubMedClient

        with patch("cdr.retrieval.pubmed_client.get_settings", return_value=mock_settings):
            with patch("cdr.retrieval.pubmed_client.get_tracer") as mock_tracer:
                mock_span = MagicMock()
                mock_span.__enter__ = lambda s: s
                mock_span.__exit__ = lambda s, *a: None
                mock_tracer.return_value.span.return_value = mock_span
                return PubMedClient()

    def test_client_initialization(self, client):
        """Test client initializes with correct configuration."""
        assert client._api_key == "test_api_key"
        assert client._email == "test@example.com"
        assert client._max_results == 100
        assert client._request_interval == 0.1  # With API key

    def test_client_initialization_without_api_key(self, mock_settings):
        """Test client rate limits properly without API key."""
        from cdr.retrieval.pubmed_client import PubMedClient

        mock_settings.retrieval.ncbi_api_key = None
        with patch("cdr.retrieval.pubmed_client.get_settings", return_value=mock_settings):
            with patch("cdr.retrieval.pubmed_client.get_tracer") as mock_tracer:
                mock_span = MagicMock()
                mock_span.__enter__ = lambda s: s
                mock_span.__exit__ = lambda s, *a: None
                mock_tracer.return_value.span.return_value = mock_span
                client = PubMedClient()
                assert client._request_interval == 0.34  # Without API key

    def test_build_params_includes_api_key(self, client):
        """Test _build_params includes API key when configured."""
        params = client._build_params(term="test query")

        assert params["db"] == "pubmed"
        assert params["retmode"] == "xml"
        assert params["api_key"] == "test_api_key"
        assert params["email"] == "test@example.com"
        assert params["term"] == "test query"

    def test_build_params_omits_none_values(self, client):
        """Test _build_params omits None values."""
        params = client._build_params(term="test", extra=None)

        assert "term" in params
        assert "extra" not in params

    def test_search_filters_to_filter_string(self):
        """Test PubMedSearchFilters.to_filter_string()."""
        from cdr.retrieval.pubmed_client import PubMedSearchFilters

        filters = PubMedSearchFilters(
            publication_types=["Randomized Controlled Trial", "Meta-Analysis"],
            humans_only=True,
            english_only=True,
        )

        filter_str = filters.to_filter_string()

        assert '"Randomized Controlled Trial"[pt]' in filter_str
        assert '"Meta-Analysis"[pt]' in filter_str
        assert '"Humans"[mh]' in filter_str
        assert '"English"[la]' in filter_str

    def test_search_filters_empty(self):
        """Test PubMedSearchFilters with no filters returns empty string."""
        from cdr.retrieval.pubmed_client import PubMedSearchFilters

        filters = PubMedSearchFilters()
        assert filters.to_filter_string() == ""


# =============================================================================
# CLINICALTRIALS CLIENT TESTS
# =============================================================================


class TestClinicalTrialsClient:
    """Tests for ClinicalTrialsClient using public API."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for CT.gov client."""
        mock = MagicMock()
        mock.retrieval.clinical_trials_max_results = 50
        return mock

    @pytest.fixture
    def client(self, mock_settings):
        """Create ClinicalTrials client instance with mocked settings."""
        from cdr.retrieval.ct_client import ClinicalTrialsClient

        with patch("cdr.retrieval.ct_client.get_settings", return_value=mock_settings):
            with patch("cdr.retrieval.ct_client.get_tracer") as mock_tracer:
                mock_span = MagicMock()
                mock_span.__enter__ = lambda s: s
                mock_span.__exit__ = lambda s, *a: None
                mock_tracer.return_value.span.return_value = mock_span
                return ClinicalTrialsClient()

    def test_client_initialization(self, client, mock_settings):
        """Test client initializes correctly."""
        assert client._max_results == 50

    def test_ct_search_filters_to_params_study_type(self):
        """Test CTSearchFilters.to_params() with study type."""
        from cdr.retrieval.ct_client import CTSearchFilters

        filters = CTSearchFilters(study_type="INTERVENTIONAL")
        params = filters.to_params()

        assert "aggFilters" in params
        assert "studyType:int" in params["aggFilters"]

    def test_ct_search_filters_to_params_phases(self):
        """Test CTSearchFilters.to_params() with phases."""
        from cdr.retrieval.ct_client import CTSearchFilters

        filters = CTSearchFilters(phases=["PHASE2", "PHASE3"])
        params = filters.to_params()

        assert "aggFilters" in params
        assert "phase:2" in params["aggFilters"]
        assert "phase:3" in params["aggFilters"]

    def test_ct_search_filters_to_params_statuses(self):
        """Test CTSearchFilters.to_params() with statuses."""
        from cdr.retrieval.ct_client import CTSearchFilters

        filters = CTSearchFilters(study_statuses=["COMPLETED", "RECRUITING"])
        params = filters.to_params()

        assert "aggFilters" in params
        assert "status:com" in params["aggFilters"]
        assert "status:rec" in params["aggFilters"]

    def test_ct_search_filters_empty(self):
        """Test CTSearchFilters with no filters returns empty params."""
        from cdr.retrieval.ct_client import CTSearchFilters

        filters = CTSearchFilters()
        params = filters.to_params()

        assert params == {}

    def test_ct_search_filters_combined(self):
        """Test CTSearchFilters with multiple filter types."""
        from cdr.retrieval.ct_client import CTSearchFilters

        filters = CTSearchFilters(
            study_type="INTERVENTIONAL",
            phases=["PHASE3"],
            study_statuses=["COMPLETED"],
        )
        params = filters.to_params()

        assert "aggFilters" in params
        agg = params["aggFilters"]
        assert "studyType:int" in agg
        assert "phase:3" in agg
        assert "status:com" in agg


# =============================================================================
# EMBEDDER TESTS
# =============================================================================


class TestEmbedder:
    """Tests for Embedder class using public API with deterministic mocks."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for embedder."""
        mock = MagicMock()
        mock.retrieval.embedding_model = "test-model"
        mock.retrieval.embedding_batch_size = 32
        return mock

    @pytest.fixture
    def mock_model(self):
        """Create deterministic mock for SentenceTransformer."""
        mock = MagicMock()
        mock.get_sentence_embedding_dimension.return_value = 384
        mock.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4] * 96)  # 384-dim
        return mock

    @pytest.fixture
    def embedder(self, mock_settings, mock_model):
        """Create embedder with mocked model."""
        from cdr.retrieval.embedder import Embedder

        with patch("cdr.retrieval.embedder.get_settings", return_value=mock_settings):
            with patch("cdr.retrieval.embedder.get_tracer") as mock_tracer:
                mock_span = MagicMock()
                mock_span.__enter__ = lambda s: s
                mock_span.__exit__ = lambda s, *a: None
                mock_span.set_attribute = lambda k, v: None
                mock_tracer.return_value.span.return_value = mock_span

                with patch("cdr.retrieval.embedder.SentenceTransformer", return_value=mock_model):
                    with patch("cdr.retrieval.embedder.get_embedding_cache") as mock_cache:
                        mock_cache.return_value.get.return_value = None
                        return Embedder()

    def test_embedder_initialization(self, embedder):
        """Test embedder initializes with correct configuration."""
        assert embedder._model_name == "test-model"
        assert embedder._batch_size == 32
        assert embedder._dimension == 384

    def test_embedder_dimension_property(self, embedder):
        """Test dimension property returns correct value."""
        assert embedder.dimension == 384

    def test_embedder_model_name_property(self, embedder):
        """Test model_name property returns correct value."""
        assert embedder.model_name == "test-model"

    def test_embed_returns_numpy_array(self, embedder):
        """Test embed() returns numpy array."""
        with patch.object(embedder, "_cache") as mock_cache:
            mock_cache.get.return_value = None
            with patch.object(embedder, "_tracer") as mock_tracer:
                mock_span = MagicMock()
                mock_span.__enter__ = lambda s: s
                mock_span.__exit__ = lambda s, *a: None
                mock_span.set_attribute = lambda k, v: None
                mock_tracer.span.return_value = mock_span

                embedding = embedder.embed("Test text", use_cache=False)

                assert isinstance(embedding, np.ndarray)
                assert len(embedding) == 384

    def test_embed_batch_returns_list(self, embedder):
        """Test embed_batch() returns list of arrays."""
        with patch.object(embedder._model, "encode") as mock_encode:
            mock_encode.return_value = np.array(
                [
                    [0.1] * 384,
                    [0.2] * 384,
                    [0.3] * 384,
                ]
            )
            with patch.object(embedder, "_tracer") as mock_tracer:
                mock_span = MagicMock()
                mock_span.__enter__ = lambda s: s
                mock_span.__exit__ = lambda s, *a: None
                mock_span.set_attribute = lambda k, v: None
                mock_tracer.span.return_value = mock_span

                embeddings = embedder.embed_batch(["text1", "text2", "text3"], use_cache=False)

                assert len(embeddings) == 3
                mock_encode.assert_called_once()

    def test_embedder_raises_on_missing_library(self, mock_settings):
        """Test embedder raises error when sentence-transformers not installed."""
        from cdr.retrieval.embedder import Embedder

        with patch("cdr.retrieval.embedder.SENTENCE_TRANSFORMERS_AVAILABLE", False):
            with patch("cdr.retrieval.embedder.get_settings", return_value=mock_settings):
                with pytest.raises(Exception) as exc_info:
                    Embedder()
                assert "sentence-transformers" in str(exc_info.value)


# =============================================================================
# BM25 RETRIEVER TESTS
# =============================================================================


class TestBM25Retriever:
    """Tests for BM25Retriever."""

    @pytest.fixture
    def retriever(self):
        """Create BM25 retriever with test corpus."""
        from cdr.retrieval.bm25 import BM25Retriever

        corpus = [
            "Diabetes mellitus type 2 treatment with metformin",
            "Hypertension management in elderly patients",
            "Metformin efficacy in prediabetes",
            "Cardiovascular outcomes in diabetes patients",
        ]
        doc_ids = [str(i) for i in range(len(corpus))]

        retriever = BM25Retriever()
        retriever.index_documents(doc_ids, corpus)
        return retriever

    def test_search_basic(self, retriever):
        """Test basic BM25 search."""
        results = retriever.search("metformin diabetes", top_k=2)

        assert len(results) == 2
        # Results are (doc_id, score) tuples - doc_ids are "0", "2" etc.
        doc_ids = [r[0] for r in results]
        assert "0" in doc_ids or "2" in doc_ids  # metformin docs

    def test_search_scores(self, retriever):
        """Test that search returns scores."""
        results = retriever.search("hypertension elderly", top_k=3)

        for doc_id, score in results:
            assert isinstance(doc_id, str)
            assert isinstance(score, float)
            assert score >= 0

    def test_search_empty_query(self, retriever):
        """Test search with empty query."""
        results = retriever.search("", top_k=5)
        assert len(results) == 0

    def test_tokenize(self, retriever):
        """Test tokenization."""
        tokens = retriever._tokenize("Type 2 Diabetes Mellitus")

        assert "type" in tokens
        assert "diabetes" in tokens
        assert "mellitus" in tokens
        # Note: single-character tokens like "2" are filtered out


# =============================================================================
# RERANKER TESTS
# =============================================================================


class TestReranker:
    """Tests for Reranker class using public API."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for reranker."""
        mock = MagicMock()
        mock.retrieval.reranker_model = "test-reranker"
        mock.retrieval.reranker_top_k = 10
        return mock

    @pytest.fixture
    def mock_model(self):
        """Create deterministic mock for CrossEncoder."""
        mock = MagicMock()
        mock.predict.return_value = np.array([0.9, 0.3, 0.7, 0.1])
        return mock

    @pytest.fixture
    def reranker(self, mock_settings, mock_model):
        """Create reranker with mocked model."""
        from cdr.retrieval.reranker import Reranker

        with patch("cdr.retrieval.reranker.get_settings", return_value=mock_settings):
            with patch("cdr.retrieval.reranker.get_tracer") as mock_tracer:
                mock_span = MagicMock()
                mock_span.__enter__ = lambda s: s
                mock_span.__exit__ = lambda s, *a: None
                mock_span.set_attribute = lambda k, v: None
                mock_tracer.return_value.span.return_value = mock_span

                with patch("cdr.retrieval.reranker.CrossEncoder", return_value=mock_model):
                    return Reranker()

    def test_reranker_initialization(self, reranker):
        """Test reranker initializes correctly."""
        assert reranker._model_name == "test-reranker"

    def test_reranker_model_name_property(self, reranker):
        """Test model_name property."""
        assert reranker.model_name == "test-reranker"

    def test_score_single_pair(self, reranker):
        """Test score() method for single query-text pair."""
        with patch.object(reranker._model, "predict", return_value=np.array([0.85])):
            score = reranker.score(
                "diabetes treatment", "Type 2 diabetes mellitus treatment options"
            )

            assert isinstance(score, float)
            assert score == 0.85

    def test_score_batch(self, reranker):
        """Test score_batch() method."""
        texts = ["doc1", "doc2", "doc3"]
        with patch.object(reranker._model, "predict", return_value=np.array([0.9, 0.5, 0.3])):
            scores = reranker.score_batch("query", texts)

            assert len(scores) == 3
            assert scores[0] == 0.9
            assert scores[1] == 0.5
            assert scores[2] == 0.3

    def test_score_batch_empty(self, reranker):
        """Test score_batch() with empty texts returns empty list."""
        scores = reranker.score_batch("query", [])
        assert scores == []

    def test_rerank_with_tuples(self, reranker, mock_settings):
        """Test rerank() with (doc_id, data) tuples - current API."""
        candidates = [
            ("doc1", "Treatment of type 2 diabetes"),
            ("doc2", "Hypertension guidelines"),
            ("doc3", "Diabetes medication efficacy"),
            ("doc4", "Cancer prevention"),
        ]

        with patch.object(reranker._model, "predict", return_value=np.array([0.9, 0.3, 0.7, 0.1])):
            with patch.object(reranker, "_tracer") as mock_tracer:
                mock_span = MagicMock()
                mock_span.__enter__ = lambda s: s
                mock_span.__exit__ = lambda s, *a: None
                mock_span.set_attribute = lambda k, v: None
                mock_tracer.span.return_value = mock_span

                reranked = reranker.rerank("diabetes treatment", candidates, top_k=2)

                # Should return (doc_id, data, score) tuples sorted by score
                assert len(reranked) == 2
                assert reranked[0][0] == "doc1"  # Highest score (0.9)
                assert reranked[0][2] == 0.9
                assert reranked[1][0] == "doc3"  # Second highest (0.7)

    def test_rerank_empty_candidates(self, reranker):
        """Test rerank() with empty candidates returns empty list."""
        reranked = reranker.rerank("query", [])
        assert reranked == []

    def test_reranker_raises_on_missing_library(self, mock_settings):
        """Test reranker raises error when sentence-transformers not installed."""
        from cdr.retrieval.reranker import Reranker

        with patch("cdr.retrieval.reranker.CROSS_ENCODER_AVAILABLE", False):
            with patch("cdr.retrieval.reranker.get_settings", return_value=mock_settings):
                with pytest.raises(Exception) as exc_info:
                    Reranker()
                assert "sentence-transformers" in str(exc_info.value)


# =============================================================================
# QDRANT STORE TESTS
# =============================================================================


class TestQdrantStore:
    """Tests for QdrantStore using public API."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for Qdrant store."""
        mock = MagicMock()
        mock.vector_store.qdrant_url = "http://localhost:6333"
        mock.vector_store.qdrant_api_key = None
        mock.vector_store.qdrant_collection = "test_collection"
        mock.vector_store.qdrant_vector_size = 384
        return mock

    @pytest.fixture
    def mock_client(self):
        """Create deterministic mock for QdrantClient."""
        mock = MagicMock()
        mock.get_collection.return_value = MagicMock()
        return mock

    @pytest.fixture
    def store(self, mock_settings, mock_client):
        """Create Qdrant store with mocked client."""
        from cdr.retrieval.qdrant_store import QdrantStore

        with patch("cdr.retrieval.qdrant_store.get_settings", return_value=mock_settings):
            with patch("cdr.retrieval.qdrant_store.get_tracer") as mock_tracer:
                mock_span = MagicMock()
                mock_span.__enter__ = lambda s: s
                mock_span.__exit__ = lambda s, *a: None
                mock_span.set_attribute = lambda k, v: None
                mock_tracer.return_value.span.return_value = mock_span

                with patch("cdr.retrieval.qdrant_store.QdrantClient", return_value=mock_client):
                    return QdrantStore()

    def test_store_initialization(self, store, mock_settings):
        """Test store initializes with correct configuration."""
        assert store._url == "http://localhost:6333"
        assert store._default_collection == "test_collection"
        assert store._vector_size == 384

    def test_store_initialization_with_custom_url(self, mock_settings, mock_client):
        """Test store accepts custom URL."""
        from cdr.retrieval.qdrant_store import QdrantStore

        with patch("cdr.retrieval.qdrant_store.get_settings", return_value=mock_settings):
            with patch("cdr.retrieval.qdrant_store.get_tracer") as mock_tracer:
                mock_span = MagicMock()
                mock_span.__enter__ = lambda s: s
                mock_span.__exit__ = lambda s, *a: None
                mock_tracer.return_value.span.return_value = mock_span

                with patch("cdr.retrieval.qdrant_store.QdrantClient", return_value=mock_client):
                    store = QdrantStore(url="http://custom:6333")
                    assert store._url == "http://custom:6333"

    def test_collection_exists_true(self, store, mock_client):
        """Test collection_exists() returns True when collection exists."""
        mock_client.get_collection.return_value = MagicMock()

        exists = store.collection_exists("test_collection")

        assert exists is True
        mock_client.get_collection.assert_called_with("test_collection")

    def test_collection_exists_false(self, store, mock_client):
        """Test collection_exists() returns False when collection doesn't exist."""
        mock_client.get_collection.side_effect = Exception("Collection not found")

        exists = store.collection_exists("nonexistent")

        assert exists is False

    def test_create_collection(self, store, mock_client):
        """Test create_collection() creates collection with correct params."""
        with patch.object(store, "_tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_span.__enter__ = lambda s: s
            mock_span.__exit__ = lambda s, *a: None
            mock_span.set_attribute = lambda k, v: None
            mock_tracer.span.return_value = mock_span

            store.create_collection("new_collection", vector_size=768, distance="cosine")

            mock_client.create_collection.assert_called_once()
            call_kwargs = mock_client.create_collection.call_args
            assert call_kwargs[1]["collection_name"] == "new_collection"

    def test_create_collection_recreate(self, store, mock_client):
        """Test create_collection() with recreate=True deletes first."""
        with patch.object(store, "_tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_span.__enter__ = lambda s: s
            mock_span.__exit__ = lambda s, *a: None
            mock_span.set_attribute = lambda k, v: None
            mock_tracer.span.return_value = mock_span

            store.create_collection("test_collection", recreate=True)

            mock_client.delete_collection.assert_called_once_with("test_collection")
            mock_client.create_collection.assert_called_once()

    def test_upsert_vectors(self, store, mock_client):
        """Test upsert() inserts vectors with payloads."""
        ids = ["id1", "id2"]
        embeddings = [[0.1] * 384, [0.2] * 384]
        payloads = [{"text": "doc1"}, {"text": "doc2"}]

        with patch.object(store, "_tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_span.__enter__ = lambda s: s
            mock_span.__exit__ = lambda s, *a: None
            mock_span.set_attribute = lambda k, v: None
            mock_tracer.span.return_value = mock_span

            store.upsert(ids, embeddings, payloads, collection_name="test_collection")

            mock_client.upsert.assert_called_once()

    def test_store_raises_on_missing_library(self, mock_settings):
        """Test store raises error when qdrant-client not installed."""
        from cdr.retrieval.qdrant_store import QdrantStore

        with patch("cdr.retrieval.qdrant_store.QDRANT_AVAILABLE", False):
            with patch("cdr.retrieval.qdrant_store.get_settings", return_value=mock_settings):
                with pytest.raises(Exception) as exc_info:
                    QdrantStore()
                assert "qdrant-client" in str(exc_info.value)


# =============================================================================
# INTEGRATION TESTS (with mocked external services)
# =============================================================================


class TestRetrievalIntegration:
    """Integration tests for retrieval pipeline with deterministic mocks."""

    def test_bm25_to_reranker_pipeline(self):
        """Test BM25 retrieval feeding into reranker."""
        from cdr.retrieval.bm25 import BM25Retriever

        # Setup BM25
        corpus = [
            "Diabetes mellitus type 2 treatment with metformin",
            "Hypertension management in elderly patients",
            "Metformin efficacy in prediabetes prevention",
            "Cardiovascular outcomes in diabetes patients",
        ]
        doc_ids = [f"doc{i}" for i in range(len(corpus))]

        retriever = BM25Retriever()
        retriever.index_documents(doc_ids, corpus)

        # BM25 search
        bm25_results = retriever.search("diabetes metformin", top_k=4)

        # Verify BM25 returns expected format
        assert len(bm25_results) > 0
        for doc_id, score in bm25_results:
            assert doc_id in doc_ids
            assert isinstance(score, float)

        # Convert to reranker format (doc_id, text)
        candidates = [
            (doc_id, corpus[int(doc_id.replace("doc", ""))]) for doc_id, _ in bm25_results
        ]

        # Mock reranker for deterministic test
        with patch("cdr.retrieval.reranker.get_settings") as mock_settings:
            mock_settings.return_value.retrieval.reranker_model = "test"
            mock_settings.return_value.retrieval.reranker_top_k = 10

            with patch("cdr.retrieval.reranker.get_tracer") as mock_tracer:
                mock_span = MagicMock()
                mock_span.__enter__ = lambda s: s
                mock_span.__exit__ = lambda s, *a: None
                mock_span.set_attribute = lambda k, v: None
                mock_tracer.return_value.span.return_value = mock_span

                with patch("cdr.retrieval.reranker.CrossEncoder") as mock_cross:
                    mock_model = MagicMock()
                    # Return deterministic scores matching candidate order
                    mock_model.predict.return_value = np.array([0.95, 0.6, 0.8, 0.3])
                    mock_cross.return_value = mock_model

                    from cdr.retrieval.reranker import Reranker

                    reranker = Reranker()
                    reranked = reranker.rerank("diabetes metformin", candidates, top_k=2)

                    # Verify reranker output format
                    assert len(reranked) == 2
                    assert len(reranked[0]) == 3  # (doc_id, text, score)
                    assert reranked[0][2] >= reranked[1][2]  # Sorted by score

    def test_hybrid_search_concept(self):
        """Test concept of hybrid BM25 + dense retrieval combination."""
        # This tests the data flow without actual embedding model

        # BM25 results
        bm25_results = [
            ("doc1", 2.5),
            ("doc2", 1.8),
            ("doc3", 1.2),
        ]

        # Simulated dense retrieval results
        dense_results = [
            ("doc2", 0.95),
            ("doc4", 0.88),
            ("doc1", 0.75),
        ]

        # Merge strategy: reciprocal rank fusion
        def reciprocal_rank_fusion(results_list, k=60):
            """Combine multiple ranked lists using RRF."""
            scores = {}
            for results in results_list:
                for rank, (doc_id, _) in enumerate(results, start=1):
                    if doc_id not in scores:
                        scores[doc_id] = 0.0
                    scores[doc_id] += 1.0 / (k + rank)
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)

        combined = reciprocal_rank_fusion([bm25_results, dense_results])

        # Verify fusion works correctly
        assert len(combined) == 4  # doc1, doc2, doc3, doc4
        # doc2 should rank high (appears in both)
        doc_ids = [doc_id for doc_id, _ in combined]
        assert "doc2" in doc_ids[:2]  # Top 2

    def test_embedding_cache_concept(self):
        """Test embedding caching concept."""
        # Test cache key generation
        model_name = "all-MiniLM-L6-v2"
        text = "Test text for embedding"

        cache_key = f"embed:{model_name}:{hash(text)}"

        assert "embed:" in cache_key
        assert model_name in cache_key
        assert str(hash(text)) in cache_key

        # Same text produces same key
        cache_key2 = f"embed:{model_name}:{hash(text)}"
        assert cache_key == cache_key2

        # Different text produces different key
        cache_key3 = f"embed:{model_name}:{hash('Different text')}"
        assert cache_key != cache_key3


# =============================================================================
# FULL-TEXT CLIENT TESTS (MOCKED - NO NETWORK)
# =============================================================================


class TestFullTextClientMocked:
    """Tests for FullTextClient using mocks (no network access).

    HIGH-2 verification: Full-text fallback with mocked PMC responses.
    Refs: CDR_Integral_Audit_2026-01-20.md HIGH-2, NCBI guidelines
    """

    def _run_async(self, coro):
        """Run async coroutine in a portable way."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coro)

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for FullTextClient."""
        mock = MagicMock()
        mock.ncbi_api_key = "test_api_key"
        mock.ncbi_email = "test@example.com"
        return mock

    @pytest.fixture
    def mock_tracer(self):
        """Mock tracer that returns a context manager span."""

        class MockSpan:
            def __init__(self):
                self.attributes = {}

            def set_attribute(self, key, value):
                self.attributes[key] = value

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        mock = MagicMock()
        mock.start_span.return_value = MockSpan()
        return mock

    @pytest.fixture
    def client(self, mock_settings, mock_tracer):
        """Create FullTextClient with mocked settings."""
        from cdr.retrieval.fulltext_client import FullTextClient

        with patch("cdr.retrieval.fulltext_client.get_settings", return_value=mock_settings):
            with patch("cdr.retrieval.fulltext_client.get_tracer", return_value=mock_tracer):
                return FullTextClient()

    def test_client_initialization(self, client):
        """Test client initializes with correct configuration."""
        assert client.api_key == "test_api_key"
        assert client.tool_email == "test@example.com"
        assert client.tool_name == "CDR-ClinicalDeepResearch"

    def test_rate_limit_enforced(self, client):
        """Test rate limiting is enforced between requests."""
        import time

        # First call sets _last_request_time
        start = time.time()
        client._rate_limit()
        first_elapsed = time.time() - start

        # Second immediate call should be delayed
        start = time.time()
        client._rate_limit()
        second_elapsed = time.time() - start

        # First call is fast, second should be delayed by MIN_REQUEST_INTERVAL
        assert first_elapsed < 0.1
        # Second call waits the remainder of interval
        assert second_elapsed >= 0.3 or first_elapsed >= 0.3

    def test_get_pmcid_from_pmid_success(self, mock_settings, mock_tracer):
        """Test successful PMID to PMCID conversion."""
        from cdr.retrieval.fulltext_client import FullTextClient

        # Mock successful ID converter response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"records": [{"pmid": "12345678", "pmcid": "PMC7654321"}]}
        mock_response.raise_for_status = MagicMock()

        with patch("cdr.retrieval.fulltext_client.get_settings", return_value=mock_settings):
            with patch("cdr.retrieval.fulltext_client.get_tracer", return_value=mock_tracer):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.get = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    client = FullTextClient()
                    # Skip rate limiting for test
                    client._last_request_time = 0

                    result = self._run_async(client._get_pmcid_from_pmid("12345678"))

                    assert result == "PMC7654321"

    def test_get_pmcid_from_pmid_not_in_pmc(self, mock_settings, mock_tracer):
        """Test PMID not in PMC returns None."""
        from cdr.retrieval.fulltext_client import FullTextClient

        # Mock response without PMCID
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"records": [{"pmid": "12345678"}]}
        mock_response.raise_for_status = MagicMock()

        with patch("cdr.retrieval.fulltext_client.get_settings", return_value=mock_settings):
            with patch("cdr.retrieval.fulltext_client.get_tracer", return_value=mock_tracer):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.get = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    client = FullTextClient()
                    client._last_request_time = 0

                    result = self._run_async(client._get_pmcid_from_pmid("99999999"))

                    assert result is None

    def test_get_pmcid_handles_network_error(self, mock_settings, mock_tracer):
        """Test graceful handling of network errors."""
        from cdr.retrieval.fulltext_client import FullTextClient

        with patch("cdr.retrieval.fulltext_client.get_settings", return_value=mock_settings):
            with patch("cdr.retrieval.fulltext_client.get_tracer", return_value=mock_tracer):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.get = AsyncMock(side_effect=Exception("Network error"))
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    client = FullTextClient()
                    client._last_request_time = 0

                    # Should return None, not raise
                    result = self._run_async(client._get_pmcid_from_pmid("12345678"))

                    assert result is None

    def test_check_oa_availability_success(self, mock_settings, mock_tracer):
        """Test OA availability check returns links."""
        from cdr.retrieval.fulltext_client import FullTextClient

        # Mock OA service XML response
        oa_xml = """<?xml version="1.0"?>
        <OA>
            <record id="PMC7654321">
                <link format="tgz" href="https://pmc.ncbi.nlm.nih.gov/oa/package/PMC7654321.tar.gz"/>
                <link format="pdf" href="https://pmc.ncbi.nlm.nih.gov/oa/pdf/PMC7654321.pdf"/>
            </record>
        </OA>"""

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = oa_xml
        mock_response.raise_for_status = MagicMock()

        with patch("cdr.retrieval.fulltext_client.get_settings", return_value=mock_settings):
            with patch("cdr.retrieval.fulltext_client.get_tracer", return_value=mock_tracer):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.get = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    client = FullTextClient()
                    client._last_request_time = 0

                    result = self._run_async(client._check_oa_availability("PMC7654321"))

                    assert result is not None
                    assert "tgz" in result
                    assert "pdf" in result

    def test_check_oa_not_available(self, mock_settings, mock_tracer):
        """Test OA not available returns None."""
        from cdr.retrieval.fulltext_client import FullTextClient

        # Mock OA service error response
        oa_xml = """<?xml version="1.0"?>
        <OA>
            <error code="idIsNotOpenAccess">PMC7654321 is not in Open Access subset</error>
        </OA>"""

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = oa_xml
        mock_response.raise_for_status = MagicMock()

        with patch("cdr.retrieval.fulltext_client.get_settings", return_value=mock_settings):
            with patch("cdr.retrieval.fulltext_client.get_tracer", return_value=mock_tracer):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.get = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    client = FullTextClient()
                    client._last_request_time = 0

                    result = self._run_async(client._check_oa_availability("PMC7654321"))

                    assert result is None

    def test_parse_jats_xml_extracts_sections(self, client):
        """Test JATS XML parsing extracts standard sections."""
        # JATS XML wrapped in OAI-PMH response format (as returned by PMC)
        jats_xml = """<?xml version="1.0"?>
        <OAI-PMH>
            <GetRecord>
                <record>
                    <metadata>
                        <article>
                            <front>
                                <article-meta>
                                    <abstract><p>This is the abstract text.</p></abstract>
                                </article-meta>
                            </front>
                            <body>
                                <sec>
                                    <title>Introduction</title>
                                    <p>This is the introduction.</p>
                                </sec>
                                <sec>
                                    <title>Methods</title>
                                    <p>These are the methods.</p>
                                    <p>More methods details.</p>
                                </sec>
                                <sec>
                                    <title>Results</title>
                                    <p>These are the results.</p>
                                </sec>
                                <sec>
                                    <title>Discussion</title>
                                    <p>This is the discussion.</p>
                                </sec>
                            </body>
                        </article>
                    </metadata>
                </record>
            </GetRecord>
        </OAI-PMH>"""

        sections = client._parse_jats_xml(jats_xml)

        assert "abstract" in sections
        assert "This is the abstract text." in sections["abstract"]
        assert "introduction" in sections
        assert "methods" in sections
        assert "These are the methods." in sections["methods"]
        assert "More methods details." in sections["methods"]
        assert "results" in sections
        assert "discussion" in sections

    def test_parse_jats_xml_handles_malformed(self, client):
        """Test JATS parsing handles malformed XML gracefully."""
        malformed_xml = "<not valid xml<<<"

        sections = client._parse_jats_xml(malformed_xml)

        assert sections == {}

    def test_get_full_text_complete_flow_success(self, mock_settings, mock_tracer):
        """Test complete full-text retrieval flow with mocks."""
        from cdr.retrieval.fulltext_client import FullTextClient

        # Mock ID converter response
        id_response = MagicMock()
        id_response.status_code = 200
        id_response.json.return_value = {"records": [{"pmid": "12345678", "pmcid": "PMC7654321"}]}
        id_response.raise_for_status = MagicMock()

        # Mock OA service response
        oa_xml = """<?xml version="1.0"?>
        <OA><record id="PMC7654321">
            <link format="pdf" href="https://example.com/pdf"/>
        </record></OA>"""
        oa_response = MagicMock()
        oa_response.status_code = 200
        oa_response.text = oa_xml
        oa_response.raise_for_status = MagicMock()

        # Mock full-text XML response (OAI-PMH wrapped format)
        fulltext_xml = """<?xml version="1.0"?>
        <OAI-PMH>
            <GetRecord>
                <record>
                    <metadata>
                        <article>
                            <front><article-meta>
                                <abstract><p>Study abstract.</p></abstract>
                            </article-meta></front>
                            <body>
                                <sec><title>Methods</title><p>Study methods.</p></sec>
                                <sec><title>Results</title><p>Study results.</p></sec>
                            </body>
                        </article>
                    </metadata>
                </record>
            </GetRecord>
        </OAI-PMH>"""
        ft_response = MagicMock()
        ft_response.status_code = 200
        ft_response.text = fulltext_xml
        ft_response.raise_for_status = MagicMock()

        # Create mock that returns different responses for different URLs
        call_count = [0]

        async def mock_get(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return id_response
            elif call_count[0] == 2:
                return oa_response
            else:
                return ft_response

        with patch("cdr.retrieval.fulltext_client.get_settings", return_value=mock_settings):
            with patch("cdr.retrieval.fulltext_client.get_tracer", return_value=mock_tracer):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.get = mock_get
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    client = FullTextClient()
                    # Disable rate limiting for test
                    client._rate_limit = MagicMock()

                    result = self._run_async(
                        client.get_full_text(
                            record_id="rec_001",
                            pmid="12345678",
                            abstract="Fallback abstract",
                        )
                    )

                    assert result.record_id == "rec_001"
                    assert result.pmid == "12345678"
                    assert result.pmcid == "PMC7654321"
                    assert result.source == "pmc_fulltext"
                    assert result.is_open_access is True
                    assert result.full_text is not None
                    assert "Study methods." in result.full_text
                    assert result.sections is not None
                    assert "methods" in result.sections

    def test_get_full_text_fallback_to_abstract(self, mock_settings, mock_tracer):
        """Test fallback to abstract when PMC not available."""
        from cdr.retrieval.fulltext_client import FullTextClient

        # Mock ID converter - not in PMC
        id_response = MagicMock()
        id_response.status_code = 200
        id_response.json.return_value = {"records": [{"pmid": "99999999"}]}  # No PMCID
        id_response.raise_for_status = MagicMock()

        with patch("cdr.retrieval.fulltext_client.get_settings", return_value=mock_settings):
            with patch("cdr.retrieval.fulltext_client.get_tracer", return_value=mock_tracer):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.get = AsyncMock(return_value=id_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    client = FullTextClient()
                    client._rate_limit = MagicMock()

                    result = self._run_async(
                        client.get_full_text(
                            record_id="rec_002",
                            pmid="99999999",
                            abstract="This is the fallback abstract content.",
                        )
                    )

                    assert result.record_id == "rec_002"
                    assert result.source == "abstract_fallback"
                    assert result.full_text == "This is the fallback abstract content."
                    assert result.retrieval_reason == "PMID not in PMC"

    def test_get_full_text_no_pmid_no_abstract(self, mock_settings, mock_tracer):
        """Test not_retrieved when no PMID and no abstract."""
        from cdr.retrieval.fulltext_client import FullTextClient

        with patch("cdr.retrieval.fulltext_client.get_settings", return_value=mock_settings):
            with patch("cdr.retrieval.fulltext_client.get_tracer", return_value=mock_tracer):
                client = FullTextClient()
                client._rate_limit = MagicMock()

                result = self._run_async(
                    client.get_full_text(
                        record_id="rec_003",
                        pmid=None,
                        abstract=None,
                    )
                )

                assert result.record_id == "rec_003"
                assert result.source == "not_retrieved"
                assert result.full_text is None
                assert "No PMID" in result.retrieval_reason

    def test_fulltext_result_dataclass_fields(self):
        """Test FullTextResult dataclass has all required fields."""
        from cdr.retrieval.fulltext_client import FullTextResult

        result = FullTextResult(record_id="test_rec")

        assert result.record_id == "test_rec"
        assert result.pmid is None
        assert result.pmcid is None
        assert result.full_text is None
        assert result.sections is None
        assert result.source == "not_retrieved"
        assert result.retrieval_reason == ""
        assert result.is_open_access is False
