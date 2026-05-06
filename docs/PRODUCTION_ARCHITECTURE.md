# Production Architecture

## Deployment

```text
Client
  -> Load balancer
  -> Nginx least-connection routing
  -> FastAPI replicas
  -> Guardrails
  -> Prompt decomposition
  -> LangChain-compatible multi-agent planner
  -> Context optimizer
  -> Redis short-term memory / CAG cache / feedback stream
  -> Qdrant long-term vector memory
  -> Cross-encoder reranker
  -> LlamaIndex / Ollama or model gateway
```

## Scaling

- Scale FastAPI replicas horizontally.
- Keep Redis and Qdrant stateful and shared.
- Scale model serving separately from API replicas.
- Use HPA signals from CPU, request rate, P95 end-to-end latency, P95 Qdrant latency, cache hit rate, and TTFT.
- Add queueing or backpressure when model TTFT crosses the SLO.

## Guardrails

- Input guardrail detects prompt injection, secret-like values, and PII-like values.
- Output guardrail redacts PII/secret-like values.
- Prompt injection does not block by default; it marks the request and continues with memory policy.

## Prompt Decomposition

The prompt decomposer extracts subqueries and routes exact-fact, preference, and procedural needs. This prevents one broad user query from forcing the same retrieval policy for every part.

## Multi-Agent Design

The orchestrator is LangChain-compatible through `as_langchain_runnable()`.

Agents:

- Privacy/Policy Agent
- Retrieval Planner Agent
- Context Optimizer Agent
- Feedback Analyst Agent
- Evaluation Agent

## Latency

Run:

```bash
python3 benchmark_latency.py --iterations 50 --token-budget 120
```

Latest deterministic comparison:

| Metric | Local context only | Production pipeline |
| --- | ---: | ---: |
| Avg TTFT | 0.000 ms | 0.002 ms |
| Avg end-to-end | 0.199 ms | 0.284 ms |
| P95 end-to-end | 0.397 ms | 0.490 ms |
| Added overhead | n/a | 0.085 ms |

The real LLM path should be measured separately because generation dominates latency.
