"""
Qdrant Vector Store

Vector storage and retrieval using Qdrant.
"""

from typing import Any
from uuid import uuid4

import numpy as np

from cdr.config import get_settings
from cdr.core.exceptions import VectorStoreError
from cdr.observability import get_tracer

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PointStruct,
        VectorParams,
        Filter,
        FieldCondition,
        MatchValue,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class QdrantStore:
    """
    Qdrant vector store for dense retrieval.

    Usage:
        store = QdrantStore()
        store.upsert("collection", ids, embeddings, payloads)
        results = store.search("collection", query_embedding, top_k=10)
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        """
        Initialize Qdrant store.

        Args:
            url: Qdrant server URL.
            api_key: Qdrant API key.
            collection_name: Default collection name.
        """
        if not QDRANT_AVAILABLE:
            raise VectorStoreError("qdrant-client not installed. Run: pip install qdrant-client")

        settings = get_settings()
        self._url = url or settings.vector_store.qdrant_url
        self._api_key = api_key or settings.vector_store.qdrant_api_key
        self._default_collection = collection_name or settings.vector_store.qdrant_collection
        self._vector_size = settings.vector_store.qdrant_vector_size

        self._tracer = get_tracer("cdr.retrieval.qdrant")

        # Initialize client
        with self._tracer.span("init_client", attributes={"url": self._url}) as span:
            try:
                self._client = QdrantClient(
                    url=self._url,
                    api_key=self._api_key,
                    timeout=30.0,
                )
            except Exception as e:
                raise VectorStoreError(f"Failed to connect to Qdrant: {e}") from e

    def create_collection(
        self,
        collection_name: str | None = None,
        vector_size: int | None = None,
        distance: str = "cosine",
        recreate: bool = False,
    ) -> None:
        """
        Create or recreate a collection.

        Args:
            collection_name: Collection name.
            vector_size: Vector dimension.
            distance: Distance metric (cosine, euclid, dot).
            recreate: Delete existing collection first.
        """
        collection = collection_name or self._default_collection
        size = vector_size or self._vector_size

        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }

        with self._tracer.span("create_collection", attributes={"collection": collection}) as span:
            try:
                if recreate:
                    try:
                        self._client.delete_collection(collection)
                    except Exception:
                        pass  # Collection might not exist

                self._client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=size,
                        distance=distance_map.get(distance, Distance.COSINE),
                    ),
                )
                span.set_attribute("vector_size", size)

            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise VectorStoreError(f"Failed to create collection: {e}") from e

    def collection_exists(self, collection_name: str | None = None) -> bool:
        """Check if collection exists."""
        collection = collection_name or self._default_collection
        try:
            self._client.get_collection(collection)
            return True
        except Exception:
            return False

    def upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray | list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
        collection_name: str | None = None,
    ) -> None:
        """
        Insert or update vectors.

        Args:
            ids: Vector IDs.
            embeddings: Vector embeddings.
            payloads: Optional metadata payloads.
            collection_name: Collection name.
        """
        collection = collection_name or self._default_collection

        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        if payloads is None:
            payloads = [{} for _ in ids]

        if len(ids) != len(embeddings) or len(ids) != len(payloads):
            raise ValueError("ids, embeddings, and payloads must have same length")

        with self._tracer.span(
            "upsert", attributes={"collection": collection, "count": len(ids)}
        ) as span:
            try:
                points = [
                    PointStruct(
                        id=id_,
                        vector=emb,
                        payload=payload,
                    )
                    for id_, emb, payload in zip(ids, embeddings, payloads)
                ]

                self._client.upsert(
                    collection_name=collection,
                    points=points,
                )

            except Exception as e:
                raise VectorStoreError(f"Upsert failed: {e}") from e

    def search(
        self,
        query_embedding: np.ndarray | list[float],
        top_k: int = 10,
        collection_name: str | None = None,
        filter_conditions: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[str, dict[str, Any], float]]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector.
            top_k: Number of results.
            collection_name: Collection name.
            filter_conditions: Payload filters.
            score_threshold: Minimum score threshold.

        Returns:
            List of (id, payload, score) tuples.
        """
        collection = collection_name or self._default_collection

        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        with self._tracer.span(
            "search", attributes={"collection": collection, "top_k": top_k}
        ) as span:
            try:
                # Build filter
                query_filter = None
                if filter_conditions:
                    conditions = [
                        FieldCondition(key=k, match=MatchValue(value=v))
                        for k, v in filter_conditions.items()
                    ]
                    query_filter = Filter(must=conditions)

                results = self._client.search(
                    collection_name=collection,
                    query_vector=query_embedding,
                    limit=top_k,
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                )

                output = [(str(hit.id), hit.payload or {}, hit.score) for hit in results]

                span.set_attribute("result_count", len(output))
                return output

            except Exception as e:
                raise VectorStoreError(f"Search failed: {e}") from e

    def delete(
        self,
        ids: list[str],
        collection_name: str | None = None,
    ) -> None:
        """Delete vectors by ID."""
        collection = collection_name or self._default_collection

        with self._tracer.span(
            "delete", attributes={"collection": collection, "count": len(ids)}
        ) as span:
            try:
                self._client.delete(
                    collection_name=collection,
                    points_selector=ids,
                )
            except Exception as e:
                raise VectorStoreError(f"Delete failed: {e}") from e

    def count(self, collection_name: str | None = None) -> int:
        """Get number of vectors in collection."""
        collection = collection_name or self._default_collection
        try:
            info = self._client.get_collection(collection)
            return info.points_count
        except Exception:
            return 0

    def get_by_id(
        self,
        id_: str,
        collection_name: str | None = None,
    ) -> tuple[list[float], dict[str, Any]] | None:
        """Get vector and payload by ID."""
        collection = collection_name or self._default_collection
        try:
            results = self._client.retrieve(
                collection_name=collection,
                ids=[id_],
                with_vectors=True,
            )
            if results:
                point = results[0]
                return (point.vector, point.payload or {})
            return None
        except Exception:
            return None

    def close(self) -> None:
        """Close client connection."""
        try:
            self._client.close()
        except Exception:
            pass


# Global store instance
_store: QdrantStore | None = None


def get_qdrant_store() -> QdrantStore:
    """Get global Qdrant store instance."""
    global _store
    if _store is None:
        _store = QdrantStore()
    return _store


def reset_qdrant_store() -> None:
    """Reset global store (for testing)."""
    global _store
    if _store:
        _store.close()
    _store = None
