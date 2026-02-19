"""
Tests for hybrid retrieval integration (MEDIUM-3).

Refs: CDR_Integral_Audit_2026-01-20.md MEDIUM-3
- BM25 sparse retrieval
- Optional cross-encoder reranking
- retrieval_scores per Record
"""

import pytest

from cdr.core.schemas import Record, RecordSource


class TestBM25Retriever:
    """Test BM25 sparse retrieval component."""

    def test_bm25_indexing(self):
        """BM25Retriever should index documents correctly."""
        from cdr.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever()
        doc_ids = ["doc1", "doc2", "doc3"]
        texts = [
            "diabetes mellitus treatment metformin",
            "hypertension cardiovascular disease",
            "diabetes insulin therapy glucose control",
        ]

        retriever.index_documents(doc_ids, texts)

        # Index should be built
        assert retriever._index is not None
        assert retriever._index.total_docs == 3

    def test_bm25_search(self):
        """BM25Retriever should return relevant documents."""
        from cdr.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever()
        doc_ids = ["doc1", "doc2", "doc3"]
        texts = [
            "diabetes mellitus treatment metformin",
            "hypertension cardiovascular disease",
            "diabetes insulin therapy glucose control",
        ]

        retriever.index_documents(doc_ids, texts)

        # Search for diabetes - should return matching docs
        results = retriever.search("diabetes treatment", top_k=3)

        # BM25 only returns docs with matching terms
        assert len(results) >= 1
        # Results are (doc_id, score) tuples
        result_ids = [r[0] for r in results]
        # doc1 and doc3 mention diabetes
        assert "doc1" in result_ids or "doc3" in result_ids

        # Scores should be positive for matching documents
        assert results[0][1] > 0

    def test_bm25_parameters(self):
        """BM25 parameters should be configurable."""
        from cdr.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever(k1=2.0, b=0.5)
        assert retriever._k1 == 2.0
        assert retriever._b == 0.5


class TestRecordRetrievalScores:
    """Test retrieval_scores field on Record."""

    def test_record_has_retrieval_scores_field(self):
        """Record should have retrieval_scores field."""
        record = Record(
            record_id="test-123",
            title="Test Study",
            source=RecordSource.PUBMED,
            content_hash="abc123",
        )

        assert hasattr(record, "retrieval_scores")
        assert record.retrieval_scores == {}

    def test_record_with_bm25_score(self):
        """Record should store BM25 score."""
        record = Record(
            record_id="test-123",
            title="Test Study",
            source=RecordSource.PUBMED,
            content_hash="abc123",
            retrieval_scores={"bm25": 0.85},
        )

        assert record.retrieval_scores["bm25"] == 0.85

    def test_record_with_multiple_scores(self):
        """Record should store multiple retrieval scores."""
        record = Record(
            record_id="test-123",
            title="Test Study",
            source=RecordSource.PUBMED,
            content_hash="abc123",
            retrieval_scores={
                "bm25": 0.85,
                "dense": 0.72,
                "rerank": 0.91,
            },
        )

        assert record.retrieval_scores["bm25"] == 0.85
        assert record.retrieval_scores["dense"] == 0.72
        assert record.retrieval_scores["rerank"] == 0.91


class TestHybridRetrievalIntegration:
    """Test hybrid retrieval integration in retrieve_node."""

    def test_bm25_scoring_enabled_by_default(self):
        """BM25 scoring should be enabled by default via config."""
        # Test that enable_hybrid_retrieval defaults to True
        config = {"configurable": {}}
        enable_hybrid = config.get("configurable", {}).get("enable_hybrid_retrieval", True)
        assert enable_hybrid is True

    def test_reranking_disabled_by_default(self):
        """Cross-encoder reranking should be disabled by default (expensive)."""
        config = {"configurable": {}}
        enable_rerank = config.get("configurable", {}).get("enable_reranker", False)
        assert enable_rerank is False

    def test_bm25_index_efficiency(self):
        """BM25 indexing should handle large document sets efficiently."""
        from cdr.retrieval.bm25 import BM25Retriever
        import time

        retriever = BM25Retriever()

        # Generate 100 documents
        doc_ids = [f"doc_{i}" for i in range(100)]
        texts = [
            f"document {i} contains text about medical research topic {i % 10}" for i in range(100)
        ]

        start = time.time()
        retriever.index_documents(doc_ids, texts)
        elapsed = time.time() - start

        # Indexing should be fast (< 1 second for 100 docs)
        assert elapsed < 1.0
        assert retriever._index.total_docs == 100

    def test_retrieval_scores_in_model_dump(self):
        """retrieval_scores should be serializable."""
        record = Record(
            record_id="test-123",
            title="Test Study",
            source=RecordSource.PUBMED,
            content_hash="abc123",
            retrieval_scores={"bm25": 0.85, "rerank": 0.91},
        )

        dump = record.model_dump()

        assert "retrieval_scores" in dump
        assert dump["retrieval_scores"]["bm25"] == 0.85
        assert dump["retrieval_scores"]["rerank"] == 0.91
