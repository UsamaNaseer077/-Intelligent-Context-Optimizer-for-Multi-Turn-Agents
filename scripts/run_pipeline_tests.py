from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_latency import benchmark_local_context, benchmark_production_pipeline


def run_command(args: list[str], timeout: int = 120) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            args,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": args,
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    except Exception as exc:
        return {
            "command": args,
            "ok": False,
            "returncode": None,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": repr(exc),
        }


def run_unittest_suite() -> dict[str, Any]:
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    failures = [case.id() for case, _ in result.failures]
    errors = [case.id() for case, _ in result.errors]
    return {
        "ok": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": failures,
        "errors": errors,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 5.0) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "body": json.loads(body) if body else {},
            }
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        return {
            "ok": False,
            "status": exc.code,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "body": body,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": repr(exc),
        }


def wait_for_health(base_url: str, attempts: int = 30) -> dict[str, Any]:
    checks = []
    for _ in range(attempts):
        check = http_json("GET", f"{base_url}/healthz", timeout=2.0)
        checks.append(check)
        if check["ok"]:
            return {"ok": True, "attempts": len(checks), "last": check}
        time.sleep(1)
    return {"ok": False, "attempts": len(checks), "last": checks[-1] if checks else None}


def run_docker_http_smoke(base_url: str) -> dict[str, Any]:
    health = wait_for_health(base_url)
    if not health["ok"]:
        return {"ok": False, "health": health}

    scope = {
        "tenant_id": "tenant-smoke",
        "user_id": "user-smoke",
        "project_id": "project-smoke",
        "session_id": "session-smoke",
    }
    memory = http_json(
        "POST",
        f"{base_url}/v1/memory",
        {
            **scope,
            "text": "Final cutoff is Friday 8 May 2026 end of day UK time.",
            "importance": 1.0,
            "memory_class": "exact_fact",
            "protected": True,
        },
    )
    session = http_json(
        "POST",
        f"{base_url}/v1/session/message",
        {
            **scope,
            "text": "User wants direct tone and concise bullets.",
            "role": "user",
        },
    )
    first_response = http_json(
        "POST",
        f"{base_url}/v1/context/respond",
        {
            **scope,
            "query": "What is the final cutoff and how should you answer?",
            "token_budget": 120,
            "tone": "direct",
            "detail_level": "medium",
            "preferred_format": "concise bullets",
        },
    )
    second_response = http_json(
        "POST",
        f"{base_url}/v1/context/respond",
        {
            **scope,
            "query": "What is the final cutoff and how should you answer?",
            "token_budget": 120,
            "tone": "direct",
            "detail_level": "medium",
            "preferred_format": "concise bullets",
        },
    )
    item_id = memory.get("body", {}).get("item_id")
    feedback = http_json(
        "POST",
        f"{base_url}/v1/feedback",
        {"item_ids": [item_id] if item_id else [], "value": 1, "reason": "smoke test useful answer"},
    )

    response_body = first_response.get("body", {}) if isinstance(first_response.get("body"), dict) else {}
    second_body = second_response.get("body", {}) if isinstance(second_response.get("body"), dict) else {}
    trace = response_body.get("trace", {})
    checks = {
        "health_ok": health["ok"],
        "memory_write_ok": memory["ok"] and bool(item_id),
        "session_write_ok": session["ok"],
        "first_response_ok": first_response["ok"] and "Friday 8 May 2026" in response_body.get("answer", ""),
        "cache_reuse_ok": second_response["ok"] and second_body.get("cache_hit") is True,
        "feedback_ok": feedback["ok"],
        "trace_has_ttft": trace.get("first_token_latency_ms", -1) >= 0,
        "trace_has_e2e": trace.get("end_to_end_latency_ms", 0) > 0,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "health": health,
        "memory_write": memory,
        "session_write": session,
        "first_response": first_response,
        "second_response": second_response,
        "feedback": feedback,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--token-budget", type=int, default=120)
    parser.add_argument("--output", default=str(ROOT / "build" / "pipeline_test_results.json"))
    args = parser.parse_args()

    started = time.perf_counter()
    local_tests = run_unittest_suite()
    local_context = benchmark_local_context(args.iterations, args.token_budget)
    production_pipeline = benchmark_production_pipeline(args.iterations, args.token_budget)
    docker_ps = run_command(["docker", "compose", "ps", "--format", "json"], timeout=30)
    docker_smoke = run_docker_http_smoke(args.base_url)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_ok": local_tests["ok"] and docker_smoke["ok"],
        "environment": {
            "local": {"ok": local_tests["ok"]},
            "docker": {"ok": docker_smoke["ok"], "base_url": args.base_url},
        },
        "local": {
            "unit_and_integration_tests": local_tests,
            "benchmark": {
                "iterations": args.iterations,
                "token_budget": args.token_budget,
                "local_context_only": local_context,
                "production_pipeline": production_pipeline,
                "comparison": {
                    "e2e_overhead_ms_avg": production_pipeline["end_to_end_ms"]["avg"]
                    - local_context["end_to_end_ms"]["avg"],
                    "ttft_added_ms_avg": production_pipeline["ttft_ms"]["avg"]
                    - local_context["ttft_ms"]["avg"],
                },
            },
        },
        "docker": {
            "compose_ps": docker_ps,
            "http_smoke": docker_smoke,
        },
        "notes": [
            "Docker API image uses requirements-api.txt; full LlamaIndex/Ollama/HuggingFace eval dependencies remain in requirements.txt.",
            "The Docker HTTP smoke test exercises Nginx -> API -> guardrails -> prompt decomposition -> multi-agent planner -> context optimizer -> deterministic LLM -> feedback.",
        ],
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"ok": result["overall_ok"], "output": str(output), "duration_ms": result["duration_ms"]}, indent=2))


if __name__ == "__main__":
    main()
