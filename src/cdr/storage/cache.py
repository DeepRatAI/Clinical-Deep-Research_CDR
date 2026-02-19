"""
Cache Module

In-memory and disk-based caching for expensive operations.
Supports TTL and LRU eviction strategies.
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, TypeVar

from cdr.core.exceptions import CacheError


T = TypeVar("T")


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    value: Any
    created_at: float
    ttl: float | None = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = time.time()


class LRUCache:
    """
    Thread-safe LRU cache with optional TTL.

    Features:
        - LRU eviction when max size reached
        - Optional TTL per entry
        - Thread-safe operations
        - Hit/miss statistics
    """

    def __init__(self, max_size: int = 1000, default_ttl: float | None = None) -> None:
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries.
            default_ttl: Default TTL in seconds (None = no expiry).
        """
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Returns None if not found or expired.
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: TTL in seconds (None = use default).
        """
        with self._lock:
            # Remove if exists (to update position)
            if key in self._cache:
                del self._cache[key]

            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl if ttl is not None else self._default_ttl,
            )

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def contains(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None

    @property
    def size(self) -> int:
        """Current cache size."""
        with self._lock:
            return len(self._cache)

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }


class DiskCache:
    """
    Disk-based cache for large values.

    Uses filesystem storage with optional TTL.
    """

    def __init__(self, cache_dir: Path, default_ttl: float | None = None) -> None:
        """
        Initialize disk cache.

        Args:
            cache_dir: Directory for cache files.
            default_ttl: Default TTL in seconds.
        """
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._default_ttl = default_ttl
        self._meta_file = cache_dir / "_meta.json"
        self._meta = self._load_meta()

    def _load_meta(self) -> dict[str, Any]:
        """Load metadata file."""
        if self._meta_file.exists():
            return json.loads(self._meta_file.read_text())
        return {}

    def _save_meta(self) -> None:
        """Save metadata file."""
        self._meta_file.write_text(json.dumps(self._meta, indent=2))

    def _key_to_filename(self, key: str) -> str:
        """Convert key to safe filename."""
        return hashlib.sha256(key.encode()).hexdigest()

    def _file_path(self, key: str) -> Path:
        """Get file path for key."""
        return self._cache_dir / f"{self._key_to_filename(key)}.json"

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        meta = self._meta.get(key)
        if meta is None:
            return None

        # Check expiry
        if meta.get("ttl") is not None:
            if time.time() > meta["created_at"] + meta["ttl"]:
                self.delete(key)
                return None

        file_path = self._file_path(key)
        if not file_path.exists():
            return None

        try:
            data = json.loads(file_path.read_text())
            return data.get("value")
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set value in cache."""
        file_path = self._file_path(key)

        try:
            file_path.write_text(json.dumps({"value": value}, default=str))
            self._meta[key] = {
                "created_at": time.time(),
                "ttl": ttl if ttl is not None else self._default_ttl,
                "filename": file_path.name,
            }
            self._save_meta()
        except OSError as e:
            raise CacheError(f"Failed to write cache: {e}") from e

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        if key not in self._meta:
            return False

        file_path = self._file_path(key)
        if file_path.exists():
            file_path.unlink()

        del self._meta[key]
        self._save_meta()
        return True

    def clear(self) -> None:
        """Clear all cache entries."""
        for key in list(self._meta.keys()):
            self.delete(key)

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        removed = 0
        now = time.time()

        for key, meta in list(self._meta.items()):
            if meta.get("ttl") is not None:
                if now > meta["created_at"] + meta["ttl"]:
                    self.delete(key)
                    removed += 1

        return removed


def cached(
    cache: LRUCache, key_fn: Callable[..., str] | None = None, ttl: float | None = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for caching function results.

    Args:
        cache: LRUCache instance.
        key_fn: Function to generate cache key from args.
        ttl: TTL for cached entries.

    Usage:
        cache = LRUCache(max_size=100)

        @cached(cache, key_fn=lambda x: f"compute_{x}")
        def expensive_compute(x):
            return x ** 2
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Generate key
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{func.__name__}:{args}:{kwargs}"

            # Try cache
            result = cache.get(key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            return result

        return wrapper

    return decorator


# Global caches (initialized lazily)
_embedding_cache: LRUCache | None = None
_llm_cache: LRUCache | None = None


def get_embedding_cache() -> LRUCache:
    """Get the global embedding cache."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = LRUCache(max_size=10000, default_ttl=3600)  # 1 hour
    return _embedding_cache


def get_llm_cache() -> LRUCache:
    """Get the global LLM response cache."""
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = LRUCache(max_size=1000, default_ttl=None)  # No expiry
    return _llm_cache


def reset_caches() -> None:
    """Reset all global caches (for testing)."""
    global _embedding_cache, _llm_cache
    _embedding_cache = None
    _llm_cache = None
