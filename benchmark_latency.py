from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from typing import Dict, List

from eval_harness import eval_cases, seed_memory
from memory_optimizer import ContextOptimizer, LongTermMemory, MemoryScope, ShortTermMemory, TTLCache, UserProfile
from production_pipeline import ProductionContextPipeline
from telemetry import RequestTrace, summarize_traces


def build_seeded_optimizer() -> ContextOptimizer:
    short = ShortTermMemory(max_messages=6)
    long = LongTermMemory()
    cache = TTLCache(max_items=64, ttl_seconds=600)
    seed_memory(short, long)
    return ContextOptimizer(short, long, cache)


def build_scoped_seeded_optimizer(scope: MemoryScope) -> ContextOptimizer:
    source_short = ShortTermMemory(max_messages=6)
    source_long = LongTermMemory()
    seed_memory(source_short, source_long)

    short = ShortTermMemory(max_messages=6)
    long = LongTermMemory()
    cache = TTLCache(max_items=64, ttl_seconds=600)
    for item in source_short.recent("s1"):
        short.add(scope.session_key, item.text, item.metadata.get("role", "user"), now=item.created_at)
    for item in source_long.items.values():
        long.upsert_scoped(
            scope,
            item.text,
            importance=item.importance,
            metadata={
                "memory_class": item.memory_class.value,
                "protected": "true" if item.protected else "false",
            },
        )
    return ContextOptimizer(short, long, cache)


def benchmark_local_context(iterations: int, token_budget: int) -> Dict[str, object]:
    optimizer = build_seeded_optimizer()
    profile = UserProfile()
    traces: List[RequestTrace] = []
    cases = eval_cases()
    for i in range(iterations):
        case = cases[i % len(cases)]
        trace = RequestTrace()
        start = time.perf_counter()
        with trace.stage("context_optimizer_only"):
            result = optimizer.build(session_id=case.session_id, query=case.query, profile=profile, token_budget=token_budget)
        trace.first_token_latency_ms = 0.0
        trace.end_to_end_latency_ms = (time.perf_counter() - start) * 1000
        for stage in result.stage_timings:
            trace.spans.append(stage)
        traces.append(trace)
    return summarize_traces(traces)


def benchmark_production_pipeline(iterations: int, token_budget: int) -> Dict[str, object]:
    profile = UserProfile()
    scope = MemoryScope("bench", "user", "project", "s1")
    optimizer = build_scoped_seeded_optimizer(scope)
    pipeline = ProductionContextPipeline(optimizer)
    traces: List[RequestTrace] = []
    cases = eval_cases()
    for i in range(iterations):
        case = cases[i % len(cases)]
        result = pipeline.run(scope=scope, query=case.query, profile=profile, token_budget=token_budget)
        traces.append(result.trace)
    return summarize_traces(traces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--token-budget", type=int, default=120)
    args = parser.parse_args()

    local = benchmark_local_context(args.iterations, args.token_budget)
    production = benchmark_production_pipeline(args.iterations, args.token_budget)
    print(
        json.dumps(
            {
                "iterations": args.iterations,
                "token_budget": args.token_budget,
                "local_context_only": local,
                "production_pipeline": production,
                "comparison": {
                    "e2e_overhead_ms_avg": production["end_to_end_ms"]["avg"] - local["end_to_end_ms"]["avg"],
                    "ttft_added_ms_avg": production["ttft_ms"]["avg"] - local["ttft_ms"]["avg"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
