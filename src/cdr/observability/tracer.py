"""
CDR Tracer

Structured tracing for CDR operations using OpenTelemetry-compatible format.
"""

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generator

from cdr.config import get_settings


class SpanKind(str, Enum):
    """Span types for categorization."""

    INTERNAL = "internal"
    CLIENT = "client"  # External API calls
    PRODUCER = "producer"  # Publishing events
    CONSUMER = "consumer"  # Processing events


class SpanStatus(str, Enum):
    """Span completion status."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanEvent:
    """Event within a span."""

    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """
    Trace span representing a unit of work.

    Attributes:
        trace_id: Unique trace identifier.
        span_id: Unique span identifier.
        parent_id: Parent span ID (None for root).
        name: Operation name.
        kind: Span type.
        start_time: Start timestamp.
        end_time: End timestamp (set on completion).
        status: Completion status.
        attributes: Key-value metadata.
        events: Events during span.
    """

    trace_id: str
    span_id: str
    name: str
    kind: SpanKind = SpanKind.INTERNAL
    parent_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: SpanStatus = SpanStatus.UNSET
    status_message: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        """Duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add event to span."""
        self.events.append(SpanEvent(name=name, attributes=attributes or {}))

    def set_status(self, status: SpanStatus, message: str | None = None) -> None:
        """Set span status."""
        self.status = status
        self.status_message = message

    def end(self) -> None:
        """End the span."""
        if self.end_time is None:
            self.end_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [
                {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
                for e in self.events
            ],
        }

    def __enter__(self) -> "Span":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager, ending the span."""
        if exc_type is not None:
            self.set_status(SpanStatus.ERROR, str(exc_val))
            self.add_event("exception", {"type": exc_type.__name__, "message": str(exc_val)})
        elif self.status == SpanStatus.UNSET:
            self.set_status(SpanStatus.OK)
        self.end()


class Tracer:
    """
    CDR Tracer for structured operation tracking.

    Usage:
        tracer = Tracer("cdr.retrieval")

        with tracer.span("fetch_pubmed") as span:
            span.set_attribute("query", query)
            results = fetch_results()
            span.set_attribute("result_count", len(results))
    """

    def __init__(
        self, name: str, trace_id: str | None = None, export_path: Path | None = None
    ) -> None:
        """
        Initialize tracer.

        Args:
            name: Tracer name (e.g., module name).
            trace_id: Existing trace ID (for continuation).
            export_path: Path to export traces (optional).
        """
        self._name = name
        self._trace_id = trace_id or self._generate_id()
        self._export_path = export_path
        self._spans: list[Span] = []
        self._current_span: Span | None = None

    @staticmethod
    def _generate_id() -> str:
        """Generate unique ID."""
        return uuid.uuid4().hex[:16]

    @property
    def trace_id(self) -> str:
        """Current trace ID."""
        return self._trace_id

    @property
    def current_span(self) -> Span | None:
        """Currently active span."""
        return self._current_span

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """
        Context manager for creating and managing spans.

        Args:
            name: Span name.
            kind: Span type.
            attributes: Initial attributes.

        Yields:
            Active span.
        """
        parent_id = self._current_span.span_id if self._current_span else None

        span = Span(
            trace_id=self._trace_id,
            span_id=self._generate_id(),
            parent_id=parent_id,
            name=f"{self._name}.{name}",
            kind=kind,
            attributes=attributes or {},
        )

        previous_span = self._current_span
        self._current_span = span

        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.add_event("exception", {"type": type(e).__name__, "message": str(e)})
            raise
        finally:
            span.end()
            self._spans.append(span)
            self._current_span = previous_span

            # Auto-export if path configured
            if self._export_path:
                self._export_span(span)

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """
        Start a span manually (must call end() explicitly).

        Prefer using the context manager version.
        """
        parent_id = self._current_span.span_id if self._current_span else None

        span = Span(
            trace_id=self._trace_id,
            span_id=self._generate_id(),
            parent_id=parent_id,
            name=f"{self._name}.{name}",
            kind=kind,
            attributes=attributes or {},
        )

        self._current_span = span
        return span

    def end_span(self, span: Span, status: SpanStatus = SpanStatus.OK) -> None:
        """End a manually started span."""
        if span.status == SpanStatus.UNSET:
            span.set_status(status)
        span.end()
        self._spans.append(span)
        self._current_span = None

        if self._export_path:
            self._export_span(span)

    def _export_span(self, span: Span) -> None:
        """Export span to file."""
        if self._export_path is None:
            return

        self._export_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"trace_{self._trace_id}_{timestamp}.jsonl"

        with open(self._export_path / filename, "a") as f:
            f.write(json.dumps(span.to_dict()) + "\n")

    def get_spans(self) -> list[Span]:
        """Get all recorded spans."""
        return self._spans.copy()

    def to_json(self) -> str:
        """Export all spans as JSON."""
        return json.dumps(
            {
                "trace_id": self._trace_id,
                "tracer": self._name,
                "spans": [s.to_dict() for s in self._spans],
            },
            indent=2,
        )


# Global tracer registry
_tracers: dict[str, Tracer] = {}


def get_tracer(name: str, trace_id: str | None = None) -> Tracer:
    """
    Get or create a tracer by name.

    Args:
        name: Tracer name (typically module name).
        trace_id: Optional trace ID for continuation.

    Returns:
        Tracer instance.
    """
    key = f"{name}:{trace_id or 'default'}"
    if key not in _tracers:
        settings = get_settings()
        export_path = None
        if settings.features.debug:
            export_path = settings.storage.artifact_path / "traces"
        _tracers[key] = Tracer(name, trace_id, export_path)
    return _tracers[key]


def reset_tracers() -> None:
    """Reset all tracers (for testing)."""
    global _tracers
    _tracers = {}


# Global default tracer for backward compatibility
# Modules can use `from cdr.observability.tracer import tracer`
tracer = get_tracer("cdr")
