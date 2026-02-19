"""
CDR Observability Layer

Tracing, metrics, and exporters.
"""

from cdr.observability.metrics import (
    CDRMetrics,
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    get_cdr_metrics,
    get_registry,
    reset_metrics,
)
from cdr.observability.tracer import (
    Span,
    SpanEvent,
    SpanKind,
    SpanStatus,
    Tracer,
    get_tracer,
    reset_tracers,
)

__all__ = [
    # Tracer
    "Tracer",
    "Span",
    "SpanEvent",
    "SpanKind",
    "SpanStatus",
    "get_tracer",
    "reset_tracers",
    # Metrics
    "MetricsRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "CDRMetrics",
    "get_registry",
    "get_cdr_metrics",
    "reset_metrics",
]
