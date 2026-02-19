"""
CDR Retrieval Layer

Evidence retrieval from PubMed, ClinicalTrials.gov, and vector stores.
"""

from cdr.retrieval.bm25 import BM25Index, BM25Retriever
from cdr.retrieval.ct_client import ClinicalTrialsClient, CTSearchResult
from cdr.retrieval.embedder import Embedder, get_embedder, reset_embedder
from cdr.retrieval.fulltext_client import FullTextClient, FullTextResult
from cdr.retrieval.pubmed_client import PubMedClient, PubMedSearchResult
from cdr.retrieval.qdrant_store import QdrantStore, get_qdrant_store, reset_qdrant_store
from cdr.retrieval.reranker import Reranker, get_reranker, reset_reranker

__all__ = [
    # PubMed
    "PubMedClient",
    "PubMedSearchResult",
    # ClinicalTrials.gov
    "ClinicalTrialsClient",
    "CTSearchResult",
    # Full-text retrieval (PMC)
    "FullTextClient",
    "FullTextResult",
    # Embeddings
    "Embedder",
    "get_embedder",
    "reset_embedder",
    # BM25
    "BM25Retriever",
    "BM25Index",
    # Reranker
    "Reranker",
    "get_reranker",
    "reset_reranker",
    # Vector Store
    "QdrantStore",
    "get_qdrant_store",
    "reset_qdrant_store",
]
