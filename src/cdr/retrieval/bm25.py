"""
BM25 Retriever

Sparse retrieval using BM25 algorithm.
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from cdr.observability import get_tracer


@dataclass
class BM25Index:
    """BM25 index for a document collection."""

    # BM25 parameters
    k1: float = 1.5  # Term frequency saturation
    b: float = 0.75  # Document length normalization

    # Index data
    doc_ids: list[str] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    doc_texts: list[str] = field(default_factory=list)

    # Inverted index: term -> [(doc_idx, term_freq), ...]
    inverted_index: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    # Document frequencies: term -> count of docs containing term
    doc_freqs: dict[str, int] = field(default_factory=dict)

    # Corpus stats
    total_docs: int = 0
    avg_doc_length: float = 0.0


class BM25Retriever:
    """
    BM25 sparse retrieval.

    Usage:
        retriever = BM25Retriever()
        retriever.index_documents(doc_ids, texts)
        results = retriever.search("query terms", top_k=10)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """
        Initialize BM25 retriever.

        Args:
            k1: Term frequency saturation parameter.
            b: Document length normalization parameter.
        """
        self._k1 = k1
        self._b = b
        self._index: BM25Index | None = None
        self._tracer = get_tracer("cdr.retrieval.bm25")

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into terms.

        Simple tokenization: lowercase, split on non-alphanumeric.
        """
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9]+\b", text)
        # Remove very short tokens
        return [t for t in tokens if len(t) > 1]

    def index_documents(self, doc_ids: list[str], texts: list[str]) -> None:
        """
        Build BM25 index from documents.

        Args:
            doc_ids: Document identifiers.
            texts: Document texts.
        """
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have same length")

        with self._tracer.span("index_documents", attributes={"count": len(doc_ids)}) as span:
            index = BM25Index(k1=self._k1, b=self._b)
            index.doc_ids = doc_ids
            index.doc_texts = texts
            index.total_docs = len(doc_ids)

            # Build inverted index
            inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
            doc_freqs: dict[str, set] = defaultdict(set)

            for doc_idx, text in enumerate(texts):
                tokens = self._tokenize(text)
                index.doc_lengths.append(len(tokens))

                # Count term frequencies in document
                term_freqs: dict[str, int] = defaultdict(int)
                for token in tokens:
                    term_freqs[token] += 1

                # Add to inverted index
                for term, freq in term_freqs.items():
                    inverted[term].append((doc_idx, freq))
                    doc_freqs[term].add(doc_idx)

            index.inverted_index = dict(inverted)
            index.doc_freqs = {term: len(docs) for term, docs in doc_freqs.items()}
            index.avg_doc_length = sum(index.doc_lengths) / max(len(index.doc_lengths), 1)

            self._index = index
            span.set_attribute("vocab_size", len(index.inverted_index))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Search for documents matching query.

        Args:
            query: Query text.
            top_k: Number of results.

        Returns:
            List of (doc_id, score) tuples.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call index_documents first.")

        with self._tracer.span("search", attributes={"query": query, "top_k": top_k}) as span:
            query_tokens = self._tokenize(query)
            scores: dict[int, float] = defaultdict(float)

            for term in query_tokens:
                if term not in self._index.inverted_index:
                    continue

                # IDF component
                df = self._index.doc_freqs.get(term, 0)
                idf = math.log((self._index.total_docs - df + 0.5) / (df + 0.5) + 1)

                # Score each document containing this term
                for doc_idx, tf in self._index.inverted_index[term]:
                    doc_len = self._index.doc_lengths[doc_idx]

                    # BM25 scoring
                    numerator = tf * (self._k1 + 1)
                    denominator = tf + self._k1 * (
                        1 - self._b + self._b * doc_len / self._index.avg_doc_length
                    )
                    scores[doc_idx] += idf * numerator / denominator

            # Sort by score
            sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_results = sorted_results[:top_k]

            result = [(self._index.doc_ids[doc_idx], score) for doc_idx, score in top_results]

            span.set_attribute("result_count", len(result))
            return result

    def search_with_text(self, query: str, top_k: int = 10) -> list[tuple[str, str, float]]:
        """
        Search and return document texts.

        Returns:
            List of (doc_id, text, score) tuples.
        """
        if self._index is None:
            raise RuntimeError("Index not built.")

        results = self.search(query, top_k)

        # Map doc_id to index
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self._index.doc_ids)}

        return [
            (doc_id, self._index.doc_texts[id_to_idx[doc_id]], score) for doc_id, score in results
        ]

    def get_document(self, doc_id: str) -> str | None:
        """Get document text by ID."""
        if self._index is None:
            return None

        try:
            idx = self._index.doc_ids.index(doc_id)
            return self._index.doc_texts[idx]
        except ValueError:
            return None

    @property
    def num_documents(self) -> int:
        """Number of indexed documents."""
        return self._index.total_docs if self._index else 0

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return len(self._index.inverted_index) if self._index else 0
