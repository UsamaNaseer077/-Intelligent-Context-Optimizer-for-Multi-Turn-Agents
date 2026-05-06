from __future__ import annotations

import json
import math
import time
import uuid
from typing import Iterable, Optional

from memory_optimizer import (
    ContextBlock,
    FeedbackEvent,
    MemoryItem,
    MemoryClass,
    MemoryScope,
    OptimizedContext,
    ShortTermMemory,
    TTLCache,
    stable_hash,
    tokenize,
)


class HashEmbedding:
    """
    Deterministic dense embedder for local/container smoke tests.

    Real production deployments should replace this with
    LlamaIndexHuggingFaceEmbeddingConfig or another embedding provider. Keeping
    this tiny embedder available lets Redis/Qdrant integration tests run without
    downloading model weights in CI.
    """

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def get_text_embedding(self, text: str):
        return self._embed(text)

    def get_query_embedding(self, query: str):
        return self._embed(query)

    def _embed(self, text: str):
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            idx = int(stable_hash(token)[:8], 16) % self.dimensions
            vector[idx] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class RedisShortTermMemory(ShortTermMemory):
    """
    Production adapter for Redis-backed short-term memory.

    The project remains runnable without Redis. In production, pass a redis-py
    client. Keys are scoped by tenant/user/project/session to avoid leakage.
    """

    def __init__(self, redis_client, max_messages: int = 12) -> None:
        super().__init__(max_messages=max_messages)
        self.redis = redis_client

    def add_scoped(self, scope: MemoryScope, text: str, role: str, now: Optional[float] = None) -> MemoryItem:
        item = MemoryItem.create(text, "short_term", created_at=now, metadata={"role": role})
        key = scope.session_key
        self.redis.rpush(key, json.dumps(self._to_json(item)))
        self.redis.ltrim(key, -self.max_messages, -1)
        return item

    def recent_scoped(self, scope: MemoryScope):
        rows = self.redis.lrange(scope.session_key, 0, -1)
        return [self._from_json(json.loads(row)) for row in rows]

    @staticmethod
    def _to_json(item: MemoryItem) -> dict:
        return {
            "id": item.id,
            "text": item.text,
            "kind": item.kind,
            "created_at": item.created_at,
            "importance": item.importance,
            "feedback_score": item.feedback_score,
            "metadata": item.metadata,
            "vector": item.vector,
            "memory_class": item.memory_class.value,
            "protected": item.protected,
        }

    @staticmethod
    def _from_json(row: dict) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            text=row["text"],
            kind=row["kind"],
            created_at=row["created_at"],
            importance=row.get("importance", 0.5),
            feedback_score=row.get("feedback_score", 0.0),
            metadata=row.get("metadata", {}),
            vector=row.get("vector", {}),
            memory_class=row.get("memory_class", "semantic"),
            protected=row.get("protected", False),
        )


class RedisCache(TTLCache):
    """
    Production cache adapter for Redis.

    Values are JSON-serialized by the caller in a real deployment. This adapter
    is included as the production boundary; the local TTLCache is used in tests.
    """

    def __init__(self, redis_client, ttl_seconds: int = 300) -> None:
        super().__init__(ttl_seconds=ttl_seconds)
        self.redis = redis_client

    def get(self, key: str) -> Optional[object]:
        raw = self.redis.get(key)
        if raw is None:
            return None
        row = json.loads(raw)
        return OptimizedContext(
            prompt=row["prompt"],
            blocks=[ContextBlock(**block) for block in row["blocks"]],
            cache_hit=row.get("cache_hit", False),
            input_tokens=row["input_tokens"],
            estimated_cost_usd=row["estimated_cost_usd"],
            latency_ms=row.get("latency_ms", 0.0),
            pipeline_steps=row.get("pipeline_steps", []),
        )

    def set(self, key: str, value: object) -> None:
        if not isinstance(value, OptimizedContext):
            raise TypeError("RedisCache only stores OptimizedContext values")
        row = {
            "prompt": value.prompt,
            "blocks": [block.__dict__ for block in value.blocks],
            "cache_hit": value.cache_hit,
            "input_tokens": value.input_tokens,
            "estimated_cost_usd": value.estimated_cost_usd,
            "latency_ms": value.latency_ms,
            "pipeline_steps": value.pipeline_steps,
            "stage_timings": [timing.__dict__ for timing in value.stage_timings],
        }
        self.redis.setex(key, self.ttl_seconds, json.dumps(row))


class RedisFeedbackStore:
    """Stores feedback separately from memory facts for diagnosis and audit."""

    def __init__(self, redis_client) -> None:
        self.redis = redis_client

    def record(
        self,
        item_ids: Iterable[str],
        value: int,
        *,
        reason: str = "",
        scope: Optional[MemoryScope] = None,
    ) -> FeedbackEvent:
        ids = list(item_ids)
        event = FeedbackEvent(
            id=f"feedback:{stable_hash('|'.join(ids) + str(value) + reason + str(time.time()))}",
            item_ids=ids,
            value=value,
            reason=reason,
            created_at=time.time(),
            scope_key=scope.memory_key_prefix if scope else None,
        )
        key = f"{event.scope_key or 'global'}:feedback"
        self.redis.rpush(key, json.dumps(event.__dict__))
        return event


class QdrantLongTermMemory:
    """
    Production adapter shape for Qdrant-backed long-term memory.

    Pass a qdrant-client instance and an embedder exposing `get_text_embedding`.
    This stays optional so local tests do not need Qdrant.
    """

    def __init__(
        self,
        qdrant_client,
        embed_model,
        collection_name: str = "memoryos_long_term",
        redis_client=None,
    ) -> None:
        self.client = qdrant_client
        self.embed_model = embed_model
        self.collection_name = collection_name
        self.redis = redis_client
        self.index_versions = {}
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client import models

        vector_size = getattr(self.embed_model, "dimensions", 64)
        collections = self.client.get_collections().collections
        if any(collection.name == self.collection_name for collection in collections):
            return
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
        except Exception as exc:
            if "already exists" not in str(exc).lower() and "409" not in str(exc):
                raise

    @staticmethod
    def _point_id(item_id: str) -> str:
        raw = stable_hash(item_id)[:32]
        return str(uuid.UUID(hex=raw))

    def upsert_scoped(
        self,
        scope: MemoryScope,
        text: str,
        *,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> MemoryItem:
        item = MemoryItem.create(
            text,
            "long_term",
            importance=importance,
            metadata={
                **(metadata or {}),
                "tenant_id": scope.tenant_id,
                "user_id": scope.user_id,
                "project_id": scope.project_id,
            },
        )
        vector = self.embed_model.get_text_embedding(text)
        payload = {
            "text": item.text,
            "kind": item.kind,
            "created_at": item.created_at,
            "importance": item.importance,
            "feedback_score": item.feedback_score,
            "metadata": item.metadata,
            "memory_class": item.memory_class.value,
            "protected": item.protected,
        }
        payload["item_id"] = item.id
        self.client.upsert(
            collection_name=self.collection_name,
            points=[{"id": self._point_id(item.id), "vector": vector, "payload": payload}],
        )
        self.bump_index_version(scope)
        return item

    def search_scoped(self, scope: MemoryScope, query: str, limit: int = 20):
        from qdrant_client import models

        vector = self.embed_model.get_query_embedding(query)
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="metadata.tenant_id", match=models.MatchValue(value=scope.tenant_id)),
                    models.FieldCondition(key="metadata.user_id", match=models.MatchValue(value=scope.user_id)),
                    models.FieldCondition(key="metadata.project_id", match=models.MatchValue(value=scope.project_id)),
                ]
            ),
            limit=limit,
        ).points
        items = []
        for point in result:
            payload = point.payload
            item = MemoryItem.create(
                payload["text"],
                payload.get("kind", "long_term"),
                created_at=payload.get("created_at"),
                importance=payload.get("importance", 0.5),
                feedback_score=payload.get("feedback_score", 0.0),
                metadata=payload.get("metadata", {}),
            )
            item.id = payload.get("item_id", str(point.id))
            item.protected = payload.get("protected", False)
            item.memory_class = MemoryClass(payload.get("memory_class", item.memory_class.value))
            items.append(item)
        return items

    def protected_exact_facts(self, scope: Optional[MemoryScope] = None, limit: int = 8):
        from qdrant_client import models

        if scope is None:
            return []
        result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="metadata.tenant_id", match=models.MatchValue(value=scope.tenant_id)),
                    models.FieldCondition(key="metadata.user_id", match=models.MatchValue(value=scope.user_id)),
                    models.FieldCondition(key="metadata.project_id", match=models.MatchValue(value=scope.project_id)),
                    models.FieldCondition(key="protected", match=models.MatchValue(value=True)),
                ]
            ),
            limit=limit,
        )[0]
        items = []
        for point in result:
            payload = point.payload
            item = MemoryItem.create(
                payload["text"],
                payload.get("kind", "long_term"),
                created_at=payload.get("created_at"),
                importance=payload.get("importance", 0.5),
                feedback_score=payload.get("feedback_score", 0.0),
                metadata=payload.get("metadata", {}),
            )
            item.id = payload.get("item_id", str(point.id))
            item.protected = payload.get("protected", False)
            item.memory_class = MemoryClass(payload.get("memory_class", item.memory_class.value))
            items.append(item)
        return items

    def apply_feedback(self, item_ids, value: int) -> None:
        # In production, update payload feedback_score with a point update.
        # Feedback events are still stored separately in RedisFeedbackStore.
        return None

    def index_version(self, scope: Optional[MemoryScope] = None) -> int:
        key = "__global__" if scope is None else scope.memory_key_prefix
        if self.redis is not None:
            raw = self.redis.get(f"{key}:memory_index_version")
            return int(raw or 0)
        return self.index_versions.get(key, 0)

    def bump_index_version(self, scope: Optional[MemoryScope] = None) -> int:
        key = "__global__" if scope is None else scope.memory_key_prefix
        if self.redis is not None:
            return int(self.redis.incr(f"{key}:memory_index_version"))
        self.index_versions[key] = self.index_versions.get(key, 0) + 1
        return self.index_versions[key]


class VectorStoreLongTermMemoryDesign:
    """
    Design placeholder for Qdrant / Redis Stack / pgvector.

    Required production methods:
    - upsert_scoped(scope, text, importance, metadata)
    - search_scoped(scope, query, limit)
    - apply_feedback(item_ids, value)
    - index_version(scope)
    - bump_index_version(scope)

    Storage schema:
    - tenant_id, user_id, project_id
    - memory_id/content_hash
    - text
    - embedding
    - importance
    - feedback_score
    - created_at/updated_at
    - source/confidence
    """
