"""
Cross-Encoder Reranker

Reranking using cross-encoder models for improved precision.
"""

from typing import Any

import numpy as np

from cdr.config import get_settings
from cdr.core.exceptions import RetrievalError
from cdr.observability import get_tracer

try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


class Reranker:
    """
    Cross-encoder reranker for improving retrieval precision.

    Usage:
        reranker = Reranker()
        reranked = reranker.rerank(query, candidates, top_k=10)
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        """
        Initialize reranker.

        Args:
            model_name: Cross-encoder model name.
            device: Device to use.
        """
        if not CROSS_ENCODER_AVAILABLE:
            raise RetrievalError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

        settings = get_settings()
        self._model_name = model_name or settings.retrieval.reranker_model
        self._tracer = get_tracer("cdr.retrieval.reranker")

        with self._tracer.span("load_model", attributes={"model": self._model_name}) as span:
            try:
                self._model = CrossEncoder(self._model_name, device=device)
            except Exception as e:
                raise RetrievalError(f"Failed to load reranker: {e}") from e

    @property
    def model_name(self) -> str:
        """Model name."""
        return self._model_name

    def score(self, query: str, text: str) -> float:
        """
        Score a single query-text pair.

        Args:
            query: Query text.
            text: Document text.

        Returns:
            Relevance score.
        """
        return float(self._model.predict([(query, text)])[0])

    def score_batch(self, query: str, texts: list[str]) -> list[float]:
        """
        Score multiple texts against a query.

        Args:
            query: Query text.
            texts: Document texts.

        Returns:
            List of relevance scores.
        """
        if not texts:
            return []

        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, Any]],
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[str, Any, float]]:
        """
        Rerank candidates based on relevance to query.

        Args:
            query: Query text.
            candidates: List of (doc_id, data) tuples where data can be text or dict with 'text' key.
            top_k: Number of results to return.
            score_threshold: Minimum score threshold.

        Returns:
            List of (doc_id, data, score) tuples sorted by score.
        """
        if not candidates:
            return []

        settings = get_settings()
        top_k = top_k or settings.retrieval.reranker_top_k

        with self._tracer.span(
            "rerank", attributes={"count": len(candidates), "top_k": top_k}
        ) as span:
            # Extract texts
            texts = []
            for doc_id, data in candidates:
                if isinstance(data, str):
                    texts.append(data)
                elif isinstance(data, dict) and "text" in data:
                    texts.append(data["text"])
                else:
                    texts.append(str(data))

            # Score
            scores = self.score_batch(query, texts)

            # Combine results
            results = [(doc_id, data, score) for (doc_id, data), score in zip(candidates, scores)]

            # Filter by threshold
            if score_threshold is not None:
                results = [(d, t, s) for d, t, s in results if s >= score_threshold]

            # Sort by score descending
            results.sort(key=lambda x: x[2], reverse=True)

            # Limit to top_k
            results = results[:top_k]

            span.set_attribute("result_count", len(results))
            if results:
                span.set_attribute("max_score", results[0][2])
                span.set_attribute("min_score", results[-1][2])

            return results

    def rerank_records(
        self,
        query: str,
        records: list[Any],
        text_field: str = "abstract",
        id_field: str = "id",
        top_k: int | None = None,
    ) -> list[tuple[Any, float]]:
        """
        Rerank record objects.

        Args:
            query: Query text.
            records: List of record objects (dicts or Pydantic models).
            text_field: Field to use as text.
            id_field: Field to use as ID.
            top_k: Number of results.

        Returns:
            List of (record, score) tuples.
        """
        if not records:
            return []

        # Extract texts and IDs
        texts = []
        for record in records:
            if hasattr(record, text_field):
                text = getattr(record, text_field) or ""
            elif isinstance(record, dict):
                text = record.get(text_field, "")
            else:
                text = ""
            texts.append(text)

        # Score
        scores = self.score_batch(query, texts)

        # Combine and sort
        results = list(zip(records, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            results = results[:top_k]

        return results


# Global reranker instance
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """Get global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def reset_reranker() -> None:
    """Reset global reranker (for testing)."""
    global _reranker
    _reranker = None
