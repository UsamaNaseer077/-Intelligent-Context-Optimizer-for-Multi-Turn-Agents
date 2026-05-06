from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class Span:
    name: str
    latency_ms: float


@dataclass
class RequestTrace:
    spans: List[Span] = field(default_factory=list)
    first_token_latency_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        yield
        self.spans.append(Span(name, (time.perf_counter() - start) * 1000))

    def as_dict(self) -> Dict[str, object]:
        return {
            "first_token_latency_ms": self.first_token_latency_ms,
            "end_to_end_latency_ms": self.end_to_end_latency_ms,
            "stage_latencies_ms": {span.name: span.latency_ms for span in self.spans},
        }


def percentile(values: Iterable[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def summarize_traces(traces: List[RequestTrace]) -> Dict[str, object]:
    e2e = [trace.end_to_end_latency_ms for trace in traces]
    ttft = [trace.first_token_latency_ms for trace in traces]
    by_stage: Dict[str, List[float]] = {}
    for trace in traces:
        for span in trace.spans:
            by_stage.setdefault(span.name, []).append(span.latency_ms)
    return {
        "requests": len(traces),
        "ttft_ms": {
            "avg": statistics.mean(ttft) if ttft else 0.0,
            "p50": percentile(ttft, 50),
            "p95": percentile(ttft, 95),
        },
        "end_to_end_ms": {
            "avg": statistics.mean(e2e) if e2e else 0.0,
            "p50": percentile(e2e, 50),
            "p95": percentile(e2e, 95),
        },
        "stages_ms": {
            name: {
                "avg": statistics.mean(values),
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
            }
            for name, values in sorted(by_stage.items())
        },
    }
