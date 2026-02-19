"""
CDR Metrics

Metrics collection for monitoring CDR operations.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any


class MetricType(str, Enum):
    """Metric types."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """Single metric data point."""

    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


class Counter:
    """
    Monotonically increasing counter.

    Usage:
        counter = Counter("requests_total", "Total requests processed")
        counter.inc()
        counter.inc(5)
        counter.inc(labels={"status": "success"})
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._values: dict[str, float] = defaultdict(float)
        self._lock = Lock()

    def inc(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment counter."""
        key = self._labels_key(labels)
        with self._lock:
            self._values[key] += value

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Get current value."""
        key = self._labels_key(labels)
        with self._lock:
            return self._values[key]

    def _labels_key(self, labels: dict[str, str] | None) -> str:
        """Generate key from labels."""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def values(self) -> dict[str, float]:
        """Get all values."""
        with self._lock:
            return dict(self._values)


class Gauge:
    """
    Value that can go up or down.

    Usage:
        gauge = Gauge("active_connections", "Current active connections")
        gauge.set(10)
        gauge.inc()
        gauge.dec()
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._values: dict[str, float] = defaultdict(float)
        self._lock = Lock()

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Set gauge value."""
        key = self._labels_key(labels)
        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment gauge."""
        key = self._labels_key(labels)
        with self._lock:
            self._values[key] += value

    def dec(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Decrement gauge."""
        key = self._labels_key(labels)
        with self._lock:
            self._values[key] -= value

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Get current value."""
        key = self._labels_key(labels)
        with self._lock:
            return self._values[key]

    def _labels_key(self, labels: dict[str, str] | None) -> str:
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def values(self) -> dict[str, float]:
        """Get all values."""
        with self._lock:
            return dict(self._values)


class Histogram:
    """
    Distribution of values (latencies, sizes, etc.).

    Usage:
        hist = Histogram("request_duration_seconds", "Request duration")
        hist.observe(0.5)

        # Or use as timer
        with hist.time():
            process_request()
    """

    DEFAULT_BUCKETS = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
        float("inf"),
    )

    def __init__(
        self, name: str, description: str = "", buckets: tuple[float, ...] | None = None
    ) -> None:
        self.name = name
        self.description = description
        self._buckets = buckets or self.DEFAULT_BUCKETS
        self._bucket_counts: dict[str, dict[float, int]] = defaultdict(
            lambda: {b: 0 for b in self._buckets}
        )
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = Lock()

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Record an observation."""
        key = self._labels_key(labels)
        with self._lock:
            self._sums[key] += value
            self._counts[key] += 1
            for bucket in self._buckets:
                if value <= bucket:
                    self._bucket_counts[key][bucket] += 1

    def time(self, labels: dict[str, str] | None = None):
        """Context manager for timing operations."""
        return _HistogramTimer(self, labels)

    def get_count(self, labels: dict[str, str] | None = None) -> int:
        """Get observation count."""
        key = self._labels_key(labels)
        with self._lock:
            return self._counts[key]

    def get_sum(self, labels: dict[str, str] | None = None) -> float:
        """Get sum of observations."""
        key = self._labels_key(labels)
        with self._lock:
            return self._sums[key]

    def get_mean(self, labels: dict[str, str] | None = None) -> float:
        """Get mean of observations."""
        key = self._labels_key(labels)
        with self._lock:
            if self._counts[key] == 0:
                return 0.0
            return self._sums[key] / self._counts[key]

    def _labels_key(self, labels: dict[str, str] | None) -> str:
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


class _HistogramTimer:
    """Context manager for timing with histogram."""

    def __init__(self, histogram: Histogram, labels: dict[str, str] | None) -> None:
        self._histogram = histogram
        self._labels = labels
        self._start: float | None = None

    def __enter__(self) -> "_HistogramTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        if self._start is not None:
            duration = time.perf_counter() - self._start
            self._histogram.observe(duration, self._labels)


class MetricsRegistry:
    """
    Registry for all metrics.

    Usage:
        registry = MetricsRegistry()
        counter = registry.counter("requests_total", "Total requests")
        gauge = registry.gauge("active_jobs", "Active jobs")
        hist = registry.histogram("duration_seconds", "Duration")
    """

    def __init__(self) -> None:
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}
        self._lock = Lock()

    def counter(self, name: str, description: str = "") -> Counter:
        """Get or create counter."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Counter(name, description)
            return self._metrics[name]  # type: ignore

    def gauge(self, name: str, description: str = "") -> Gauge:
        """Get or create gauge."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(name, description)
            return self._metrics[name]  # type: ignore

    def histogram(
        self, name: str, description: str = "", buckets: tuple[float, ...] | None = None
    ) -> Histogram:
        """Get or create histogram."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Histogram(name, description, buckets)
            return self._metrics[name]  # type: ignore

    def get_all(self) -> dict[str, Any]:
        """Get all metric values."""
        result = {}
        with self._lock:
            for name, metric in self._metrics.items():
                if isinstance(metric, (Counter, Gauge)):
                    result[name] = metric.values()
                elif isinstance(metric, Histogram):
                    result[name] = {
                        "count": metric.get_count(),
                        "sum": metric.get_sum(),
                        "mean": metric.get_mean(),
                    }
        return result

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()


# Global metrics registry
_registry: MetricsRegistry | None = None


def get_registry() -> MetricsRegistry:
    """Get global metrics registry."""
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry


def reset_metrics() -> None:
    """Reset global metrics."""
    global _registry
    _registry = None


# Pre-defined CDR metrics
class CDRMetrics:
    """Pre-defined CDR metrics."""

    def __init__(self, registry: MetricsRegistry | None = None) -> None:
        self._registry = registry or get_registry()

    @property
    def runs_total(self) -> Counter:
        """Total CDR runs."""
        return self._registry.counter("cdr_runs_total", "Total CDR runs")

    @property
    def runs_active(self) -> Gauge:
        """Currently active runs."""
        return self._registry.gauge("cdr_runs_active", "Active CDR runs")

    @property
    def records_retrieved(self) -> Counter:
        """Total records retrieved."""
        return self._registry.counter("cdr_records_retrieved_total", "Records retrieved")

    @property
    def records_screened(self) -> Counter:
        """Total records screened."""
        return self._registry.counter("cdr_records_screened_total", "Records screened")

    @property
    def llm_requests(self) -> Counter:
        """LLM API requests."""
        return self._registry.counter("cdr_llm_requests_total", "LLM requests")

    @property
    def llm_tokens(self) -> Counter:
        """LLM tokens used."""
        return self._registry.counter("cdr_llm_tokens_total", "LLM tokens")

    @property
    def llm_latency(self) -> Histogram:
        """LLM request latency."""
        return self._registry.histogram("cdr_llm_latency_seconds", "LLM latency")

    @property
    def retrieval_latency(self) -> Histogram:
        """Retrieval latency."""
        return self._registry.histogram("cdr_retrieval_latency_seconds", "Retrieval latency")

    @property
    def verification_iterations(self) -> Counter:
        """Verification iterations."""
        return self._registry.counter(
            "cdr_verification_iterations_total", "Verification iterations"
        )

    @property
    def errors(self) -> Counter:
        """Errors by type."""
        return self._registry.counter("cdr_errors_total", "Errors by type")


def get_cdr_metrics() -> CDRMetrics:
    """Get CDR metrics instance."""
    return CDRMetrics()


class MetricsHelper:
    """
    Helper class for simple metrics operations.

    Provides a simpler interface for common metrics operations.
    Used as a global `metrics` instance for convenience.
    """

    def __init__(self) -> None:
        self._registry = get_registry()
        self._counters: dict[str, Counter] = {}

    def counter(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment a counter by name."""
        if name not in self._counters:
            self._counters[name] = self._registry.counter(name, f"Counter: {name}")
        self._counters[name].inc(value, labels)

    def get_all(self) -> dict[str, float]:
        """
        Get all metrics as a dictionary.

        Returns:
            Dict mapping metric names to their current values.
            For counters, returns the total count across all label combinations.

        Refs: CDR API /metrics endpoint (routes.py)
        """
        result: dict[str, float] = {}
        for name, counter in self._counters.items():
            # Get total value across all labels
            # Counter stores values in _values dict keyed by label combination
            result[name] = sum(counter._values.values())
        return result


# Global metrics instance for backward compatibility
# Modules can use `from cdr.observability.metrics import metrics`
metrics = MetricsHelper()
