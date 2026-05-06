# Memory Context Optimizer

Production-style scaffold for Assignment 3: compress multi-turn agent context while preserving relevant memory.

## What It Tests

- Short-term memory window, modeled as a Redis-like session store
- Long-term classified memory with Qdrant production adapter shape
- Cache-Augmented Generation behavior through a TTL/LRU cache
- Feedback-weighted memory confidence
- Separate feedback event storage
- Tenant/user/project/session memory scoping
- Cache invalidation through memory index versions
- Local reranking of top long-term memories
- Optional real cross-encoder reranking through `sentence-transformers`
- Protected exact-fact blocks for dates, prices, IDs, constraints, and policies
- P50/P95 stage latency tracking
- Token-budget-aware context selection and compression
- Local evaluation for quality, latency, token count, cost, cache hit rate, and win/tie/loss labels
- Open-source LLM adapter through LlamaIndex + Ollama

## Runtime Architecture

```text
Client
  -> Load balancer / Nginx
  -> FastAPI app replica
  -> Guardrails
  -> Prompt decomposition
  -> LangChain-style multi-agent planner
User message
  -> Session router
  -> Tenant/user/project/session scoped key
  -> Short-term memory read from Redis
  -> Long-term memory search from vector store
  -> Rerank top long-term candidates
  -> CAG cache lookup
  -> Context scorer
  -> Token-budget allocator
  -> Compression / fallback
  -> Prompt builder
  -> LLM
  -> Feedback capture
  -> Memory update + memory index version bump
  -> Cache invalidation
```

`ContextOptimizer.build()` implements the online context path through `llm_ready`.
The feedback path is represented by `LongTermMemory.apply_feedback()` and
`LongTermMemory.bump_index_version()`.

## Run Tests

```bash
cd memory_context_optimizer
python3 -m unittest discover -s tests
```

## Run API Locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
curl -s http://localhost:8000/healthz
```

## Run Production Stack

```bash
docker compose up --build
curl -s http://localhost:8080/healthz
```

Scale app replicas behind Nginx:

```bash
docker compose up --build --scale app=4
```

Kubernetes autoscaling manifest:

```bash
kubectl apply -f k8s/memoryos-api.yaml
```

## Latency Benchmark

Compare the original in-process context optimizer with the production pipeline:

```bash
python3 benchmark_latency.py --iterations 50 --token-budget 120
```

The benchmark reports first-token latency, end-to-end latency, and per-stage P50/P95 latency.

## Run Evaluation

```bash
cd memory_context_optimizer
python3 eval_harness.py --token-budget 160
```

By default, the harness uses a deterministic local LLM simulator so tests run without model downloads.

To use an open-source model through LlamaIndex + Ollama:

```bash
ollama pull llama3.2:1b
pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface
python3 eval_harness.py --token-budget 160 --llm-provider ollama --llm-model llama3.2:1b
```

Fast local model run used in the report:

```bash
ollama pull llama3.2:1b
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python eval_harness.py --token-budget 120 --llm-provider ollama --llm-model llama3.2:1b
```

Recommended open-source setup:

- LLM: `llama3.2:1b` via Ollama for local generation.
- Embeddings: `BAAI/bge-small-en-v1.5` via LlamaIndex `HuggingFaceEmbedding`.
- Long-term memory: Qdrant.
- Short-term memory and cache: Redis.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Optional stronger local models: `mistral`, `qwen2.5`, or larger Llama variants if hardware allows.

The adapter is in `llamaindex_llm.py`.

See `REPORT.md` for measured deterministic and LlamaIndex/Ollama results, win/tie/loss by query type, fallback tests, and break-even analysis.

For a complete diagrammed explanation of the system architecture, AI architecture, memory architecture, full end-to-end flow, evals, latency, and tests, see `docs/ARCHITECTURE_AND_EVALUATION.md`.

## Production Adapters (Implemented)

All production storage and reranking adapters are implemented in `storage_adapters.py` and `rerankers.py`. They are drop-in replacements for the local in-process defaults and require only a live service connection.

| Component | Local default | Production adapter | File |
| --- | --- | --- | --- |
| Short-term memory | `ShortTermMemory` | `RedisShortTermMemory` | `storage_adapters.py` |
| CAG cache | `TTLCache` | `RedisCache` (uses `SETEX`) | `storage_adapters.py` |
| Long-term memory | `LongTermMemory` | `QdrantLongTermMemory` | `storage_adapters.py` |
| Feedback store | `FeedbackStore` | `RedisFeedbackStore` | `storage_adapters.py` |
| Embeddings | deterministic token vectors | `HashEmbedding` (CI) / `LlamaIndexHuggingFaceEmbeddingConfig` | `storage_adapters.py` / `llamaindex_llm.py` |
| Reranker | `CrossEncoderReranker` | `SentenceTransformersCrossEncoderReranker` | `rerankers.py` |
| Memory index version | in-process dict | persisted in Redis via `incr` / `get` | `storage_adapters.py` |

### Swap to Redis + Qdrant

```python
import redis
from qdrant_client import QdrantClient
from storage_adapters import HashEmbedding, QdrantLongTermMemory, RedisCache, RedisShortTermMemory

r = redis.Redis(host="localhost", port=6379)
short = RedisShortTermMemory(r, max_messages=12)
cache = RedisCache(r, ttl_seconds=300)
long = QdrantLongTermMemory(QdrantClient(host="localhost", port=6333), HashEmbedding(), redis_client=r)
```

### Swap to real cross-encoder reranker

```python
from rerankers import SentenceTransformersCrossEncoderReranker
reranker = SentenceTransformersCrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
optimizer = ContextOptimizer(short, long, cache, reranker=reranker)
```

Memory index versions are persisted in Redis automatically when a `redis_client` is passed to `QdrantLongTermMemory`, so cache invalidation survives restarts.
