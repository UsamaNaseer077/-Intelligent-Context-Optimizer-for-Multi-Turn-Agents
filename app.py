from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from eval_harness import seed_memory
from llamaindex_llm import DeterministicLLM, LlamaIndexOllamaLLM
from memory_optimizer import ContextOptimizer, LongTermMemory, MemoryScope, ShortTermMemory, TTLCache, UserProfile
from production_pipeline import ProductionContextPipeline
from storage_adapters import HashEmbedding, QdrantLongTermMemory, RedisCache, RedisShortTermMemory


class ContextRequest(BaseModel):
    tenant_id: str = "default"
    user_id: str = "user"
    project_id: str = "project"
    session_id: str = "session"
    query: str
    token_budget: int = Field(default=160, ge=40, le=4000)
    tone: str = "direct"
    detail_level: str = "medium"
    preferred_format: str = "concise bullets"


class FeedbackRequest(BaseModel):
    item_ids: list[str]
    value: int = Field(ge=-1, le=1)
    reason: Optional[str] = ""


class MemoryWriteRequest(BaseModel):
    tenant_id: str = "default"
    user_id: str = "user"
    project_id: str = "project"
    session_id: str = "session"
    text: str
    importance: float = Field(default=0.7, ge=0.0, le=1.0)
    memory_class: Optional[str] = None
    protected: bool = False


class SessionMessageRequest(BaseModel):
    tenant_id: str = "default"
    user_id: str = "user"
    project_id: str = "project"
    session_id: str = "session"
    text: str
    role: str = "user"


def build_pipeline() -> ProductionContextPipeline:
    redis_url = os.getenv("REDIS_URL")
    qdrant_url = os.getenv("QDRANT_URL")
    redis_client = None
    if redis_url:
        import redis

        redis_client = redis.from_url(redis_url, decode_responses=True)

    if redis_client is not None:
        short = RedisShortTermMemory(redis_client, max_messages=int(os.getenv("SHORT_TERM_MAX_MESSAGES", "12")))
        cache = RedisCache(redis_client, ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "300")))
    else:
        short = ShortTermMemory(max_messages=int(os.getenv("SHORT_TERM_MAX_MESSAGES", "12")))
        cache = TTLCache(max_items=int(os.getenv("CACHE_MAX_ITEMS", "256")), ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "300")))

    if qdrant_url:
        from qdrant_client import QdrantClient

        long = QdrantLongTermMemory(
            QdrantClient(url=qdrant_url, check_compatibility=False),
            HashEmbedding(dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "64"))),
            redis_client=redis_client,
        )
    else:
        long = LongTermMemory()
    source_short = ShortTermMemory(max_messages=6)
    source_long = LongTermMemory()
    seed_memory(source_short, source_long)
    demo_scope = MemoryScope("default", "user", "project", "session")
    for item in source_short.recent("s1"):
        if hasattr(short, "add_scoped"):
            short.add_scoped(demo_scope, item.text, item.metadata.get("role", "user"), now=item.created_at)
        else:
            short.add(demo_scope.session_key, item.text, item.metadata.get("role", "user"), now=item.created_at)
    for item in source_long.items.values():
        long.upsert_scoped(
            demo_scope,
            item.text,
            importance=item.importance,
            metadata={
                "memory_class": item.memory_class.value,
                "protected": "true" if item.protected else "false",
            },
        )
    optimizer = ContextOptimizer(short, long, cache)

    provider = os.getenv("LLM_PROVIDER", "deterministic")
    if provider == "ollama":
        llm = LlamaIndexOllamaLLM(model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"))
    else:
        llm = DeterministicLLM()
    return ProductionContextPipeline(optimizer, llm=llm)


app = FastAPI(title="MemoryOS Context Optimizer", version="1.0.0")
pipeline = build_pipeline()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/v1/context/respond")
def respond(request: ContextRequest):
    scope = MemoryScope(request.tenant_id, request.user_id, request.project_id, request.session_id)
    profile = UserProfile(request.tone, request.detail_level, request.preferred_format)
    result = pipeline.run(scope=scope, query=request.query, profile=profile, token_budget=request.token_budget)
    return result.as_dict()


@app.post("/v1/feedback")
def feedback(request: FeedbackRequest):
    pipeline.optimizer.long_term.apply_feedback(request.item_ids, request.value)
    return {"status": "recorded", "items": len(request.item_ids)}


@app.post("/v1/memory")
def write_memory(request: MemoryWriteRequest):
    scope = MemoryScope(request.tenant_id, request.user_id, request.project_id, request.session_id)
    metadata = {}
    if request.memory_class:
        metadata["memory_class"] = request.memory_class
    if request.protected:
        metadata["protected"] = "true"
    item = pipeline.optimizer.long_term.upsert_scoped(
        scope,
        request.text,
        importance=request.importance,
        metadata=metadata,
    )
    return {"status": "stored", "item_id": item.id, "memory_class": item.memory_class.value, "protected": item.protected}


@app.post("/v1/session/message")
def write_session_message(request: SessionMessageRequest):
    scope = MemoryScope(request.tenant_id, request.user_id, request.project_id, request.session_id)
    if hasattr(pipeline.optimizer.short_term, "add_scoped"):
        item = pipeline.optimizer.short_term.add_scoped(scope, request.text, request.role)
    else:
        item = pipeline.optimizer.short_term.add(scope.session_key, request.text, request.role)
    return {"status": "stored", "item_id": item.id}
