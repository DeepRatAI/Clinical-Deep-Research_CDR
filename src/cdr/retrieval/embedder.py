"""
Embedder

Text embedding using sentence-transformers.
"""

from typing import Any

import numpy as np

from cdr.config import get_settings
from cdr.core.exceptions import EmbeddingError
from cdr.observability import get_tracer
from cdr.storage.cache import cached, get_embedding_cache

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class Embedder:
    """
    Text embedder using sentence-transformers.

    Usage:
        embedder = Embedder()
        embedding = embedder.embed("Some text")
        embeddings = embedder.embed_batch(["text1", "text2"])
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """
        Initialize embedder.

        Args:
            model_name: Model name (default: all-MiniLM-L6-v2).
            device: Device to use (cpu, cuda, etc.).
            batch_size: Batch size for encoding.
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise EmbeddingError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

        settings = get_settings()
        self._model_name = model_name or settings.retrieval.embedding_model
        self._batch_size = batch_size or settings.retrieval.embedding_batch_size

        self._tracer = get_tracer("cdr.retrieval.embedder")
        self._cache = get_embedding_cache()

        # Load model
        with self._tracer.span("load_model", attributes={"model": self._model_name}) as span:
            try:
                self._model = SentenceTransformer(self._model_name, device=device)
                self._dimension = self._model.get_sentence_embedding_dimension()
                span.set_attribute("dimension", self._dimension)
            except Exception as e:
                raise EmbeddingError(f"Failed to load model: {e}") from e

    @property
    def model_name(self) -> str:
        """Model name."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self._dimension

    def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Embed single text.

        Args:
            text: Text to embed.
            use_cache: Whether to use cache.

        Returns:
            Embedding vector as numpy array.
        """
        if use_cache:
            cache_key = f"embed:{self._model_name}:{hash(text)}"
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                return np.array(cached_result)

        with self._tracer.span("embed") as span:
            try:
                embedding = self._model.encode(
                    text,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

                if use_cache:
                    self._cache.set(cache_key, embedding.tolist())

                return embedding

            except Exception as e:
                raise EmbeddingError(f"Embedding failed: {e}") from e

    def embed_batch(
        self, texts: list[str], use_cache: bool = True, show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed batch of texts.

        Args:
            texts: List of texts to embed.
            use_cache: Whether to use cache.
            show_progress: Show progress bar.

        Returns:
            Embeddings as 2D numpy array (n_texts, dimension).
        """
        if not texts:
            return np.array([]).reshape(0, self._dimension)

        with self._tracer.span("embed_batch", attributes={"count": len(texts)}) as span:
            # Check cache for each text
            embeddings = []
            texts_to_encode = []
            text_indices = []

            if use_cache:
                for i, text in enumerate(texts):
                    cache_key = f"embed:{self._model_name}:{hash(text)}"
                    cached_result = self._cache.get(cache_key)
                    if cached_result is not None:
                        embeddings.append((i, np.array(cached_result)))
                    else:
                        texts_to_encode.append(text)
                        text_indices.append(i)
            else:
                texts_to_encode = texts
                text_indices = list(range(len(texts)))

            # Encode uncached texts
            if texts_to_encode:
                try:
                    new_embeddings = self._model.encode(
                        texts_to_encode,
                        batch_size=self._batch_size,
                        convert_to_numpy=True,
                        show_progress_bar=show_progress,
                    )

                    # Cache new embeddings
                    if use_cache:
                        for text, emb in zip(texts_to_encode, new_embeddings):
                            cache_key = f"embed:{self._model_name}:{hash(text)}"
                            self._cache.set(cache_key, emb.tolist())

                    # Add to results
                    for idx, emb in zip(text_indices, new_embeddings):
                        embeddings.append((idx, emb))

                except Exception as e:
                    raise EmbeddingError(f"Batch embedding failed: {e}") from e

            # Sort by original index and stack
            embeddings.sort(key=lambda x: x[0])
            result = np.stack([emb for _, emb in embeddings])

            span.set_attribute("encoded_count", len(texts_to_encode))
            span.set_attribute("cached_count", len(texts) - len(texts_to_encode))

            return result

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Returns:
            Similarity score in [-1, 1].
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)

        # Cosine similarity
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def most_similar(
        self, query: str, candidates: list[str], top_k: int = 5
    ) -> list[tuple[int, str, float]]:
        """
        Find most similar texts to query.

        Args:
            query: Query text.
            candidates: Candidate texts.
            top_k: Number of results.

        Returns:
            List of (index, text, score) tuples.
        """
        if not candidates:
            return []

        query_emb = self.embed(query)
        cand_embs = self.embed_batch(candidates)

        # Compute similarities
        similarities = np.dot(cand_embs, query_emb) / (
            np.linalg.norm(cand_embs, axis=1) * np.linalg.norm(query_emb)
        )

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(idx), candidates[idx], float(similarities[idx])) for idx in top_indices]


# Global embedder instance
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Get global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def reset_embedder() -> None:
    """Reset global embedder (for testing)."""
    global _embedder
    _embedder = None
