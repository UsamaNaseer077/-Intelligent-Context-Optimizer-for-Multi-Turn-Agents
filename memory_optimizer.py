from __future__ import annotations

import hashlib
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def token_count(text: str) -> int:
    return len(tokenize(text))


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryScope:
    tenant_id: str
    user_id: str
    project_id: str
    session_id: str

    @property
    def session_key(self) -> str:
        return f"tenant:{self.tenant_id}:user:{self.user_id}:project:{self.project_id}:session:{self.session_id}"

    @property
    def memory_key_prefix(self) -> str:
        return f"tenant:{self.tenant_id}:user:{self.user_id}:project:{self.project_id}"


class MemoryClass(str, Enum):
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    SEMANTIC = "semantic"
    PREFERENCE = "preference"
    EXACT_FACT = "exact_fact"
    FEEDBACK = "feedback"


EXACT_FACT_RE = re.compile(
    r"(\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b|\b\d{1,2}:\d{2}\b|\b\d+(?:\.\d+)?%\b|\$[\d,.]+|\b[A-Z]{2,}-?\d{2,}\b)"
)


def classify_memory(text: str, metadata: Optional[Dict[str, str]] = None) -> MemoryClass:
    metadata = metadata or {}
    explicit = metadata.get("memory_class")
    if explicit:
        return MemoryClass(explicit)
    lowered = text.lower()
    if EXACT_FACT_RE.search(text) or any(word in lowered for word in ["deadline", "cutoff", "price", "id", "policy"]):
        return MemoryClass.EXACT_FACT
    if any(word in lowered for word in ["prefers", "style", "tone", "format"]):
        return MemoryClass.PREFERENCE
    if any(word in lowered for word in ["how to", "procedure", "workflow", "steps", "runbook"]):
        return MemoryClass.PROCEDURAL
    if any(word in lowered for word in ["user asked", "assistant recommended", "approved", "meeting", "conversation"]):
        return MemoryClass.EPISODIC
    return MemoryClass.SEMANTIC


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    overlap = set(a).intersection(b)
    dot = sum(a[t] * b[t] for t in overlap)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def tf_vector(text: str) -> Dict[str, float]:
    vec: Dict[str, float] = {}
    for tok in tokenize(text):
        vec[tok] = vec.get(tok, 0.0) + 1.0
    return vec


@dataclass
class MemoryItem:
    id: str
    text: str
    kind: str
    created_at: float
    importance: float = 0.5
    feedback_score: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)
    vector: Dict[str, float] = field(default_factory=dict)
    memory_class: MemoryClass = MemoryClass.SEMANTIC
    protected: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.memory_class, str):
            self.memory_class = MemoryClass(self.memory_class)

    @classmethod
    def create(
        cls,
        text: str,
        kind: str,
        *,
        created_at: Optional[float] = None,
        importance: float = 0.5,
        feedback_score: float = 0.0,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "MemoryItem":
        ts = time.time() if created_at is None else created_at
        raw_id = f"{kind}:{stable_hash(text)}"
        memory_class = classify_memory(text, metadata)
        protected = memory_class == MemoryClass.EXACT_FACT or (metadata or {}).get("protected") == "true"
        return cls(
            id=raw_id,
            text=text,
            kind=kind,
            created_at=ts,
            importance=max(0.0, min(1.0, importance)),
            feedback_score=max(-1.0, min(1.0, feedback_score)),
            metadata=metadata or {},
            vector=tf_vector(text),
            memory_class=memory_class,
            protected=protected,
        )


class TTLCache:
    def __init__(self, max_items: int = 256, ttl_seconds: int = 300) -> None:
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, Tuple[float, object]] = OrderedDict()

    def get(self, key: str) -> Optional[object]:
        item = self._items.get(key)
        if item is None:
            return None
        created_at, value = item
        if time.time() - created_at > self.ttl_seconds:
            self._items.pop(key, None)
            return None
        self._items.move_to_end(key)
        return value

    def set(self, key: str, value: object) -> None:
        self._items[key] = (time.time(), value)
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)


class CacheStore(Protocol):
    def get(self, key: str) -> Optional[object]:
        ...

    def set(self, key: str, value: object) -> None:
        ...


class ShortTermMemory:
    """Redis-like recent message store with fixed session window."""

    def __init__(self, max_messages: int = 12) -> None:
        self.max_messages = max_messages
        self.sessions: Dict[str, List[MemoryItem]] = {}

    def add(self, session_id: str, text: str, role: str, now: Optional[float] = None) -> MemoryItem:
        item = MemoryItem.create(text, "short_term", created_at=now, metadata={"role": role})
        bucket = self.sessions.setdefault(session_id, [])
        bucket.append(item)
        if len(bucket) > self.max_messages:
            del bucket[: len(bucket) - self.max_messages]
        return item

    def recent(self, session_id: str) -> List[MemoryItem]:
        return list(self.sessions.get(session_id, []))


class ScopedShortTermMemory(ShortTermMemory):
    def add_scoped(self, scope: MemoryScope, text: str, role: str, now: Optional[float] = None) -> MemoryItem:
        return self.add(scope.session_key, text, role, now)

    def recent_scoped(self, scope: MemoryScope) -> List[MemoryItem]:
        return self.recent(scope.session_key)


class LongTermMemory:
    def __init__(self) -> None:
        self.items: Dict[str, MemoryItem] = {}
        self.index_versions: Dict[str, int] = {}

    def upsert(self, text: str, *, importance: float = 0.5, metadata: Optional[Dict[str, str]] = None) -> MemoryItem:
        item = MemoryItem.create(text, "long_term", importance=importance, metadata=metadata)
        existing = self.items.get(item.id)
        if existing:
            existing.importance = max(existing.importance, item.importance)
            existing.metadata.update(item.metadata)
            existing.memory_class = item.memory_class
            existing.protected = existing.protected or item.protected
            self.bump_index_version()
            return existing
        self.items[item.id] = item
        self.bump_index_version()
        return item

    def upsert_scoped(
        self,
        scope: MemoryScope,
        text: str,
        *,
        importance: float = 0.5,
        metadata: Optional[Dict[str, str]] = None,
    ) -> MemoryItem:
        scoped_metadata = dict(metadata or {})
        scoped_metadata.update(
            {
                "tenant_id": scope.tenant_id,
                "user_id": scope.user_id,
                "project_id": scope.project_id,
            }
        )
        item = self.upsert(text, importance=importance, metadata=scoped_metadata)
        self.bump_index_version(scope)
        return item

    def apply_feedback(self, item_ids: Iterable[str], value: int) -> None:
        delta = 0.2 if value > 0 else -0.25
        for item_id in item_ids:
            if item_id in self.items:
                item = self.items[item_id]
                item.feedback_score = max(-1.0, min(1.0, item.feedback_score + delta))

    def protected_exact_facts(self, scope: Optional[MemoryScope] = None, limit: int = 8) -> List[MemoryItem]:
        items = [item for item in self.items.values() if item.protected]
        if scope is not None:
            items = [
                item
                for item in items
                if item.metadata.get("tenant_id") == scope.tenant_id
                and item.metadata.get("user_id") == scope.user_id
                and item.metadata.get("project_id") == scope.project_id
            ]
        return sorted(items, key=lambda item: (item.importance, item.created_at), reverse=True)[:limit]

    def search(self, query: str, limit: int = 20) -> List[MemoryItem]:
        qv = tf_vector(query)
        ranked = sorted(self.items.values(), key=lambda item: cosine(qv, item.vector), reverse=True)
        return ranked[:limit]

    def search_scoped(self, scope: MemoryScope, query: str, limit: int = 20) -> List[MemoryItem]:
        qv = tf_vector(query)
        scoped = [
            item
            for item in self.items.values()
            if item.metadata.get("tenant_id") == scope.tenant_id
            and item.metadata.get("user_id") == scope.user_id
            and item.metadata.get("project_id") == scope.project_id
        ]
        ranked = sorted(scoped, key=lambda item: cosine(qv, item.vector), reverse=True)
        return ranked[:limit]

    def index_version(self, scope: Optional[MemoryScope] = None) -> int:
        if scope is None:
            return self.index_versions.get("__global__", 0)
        return self.index_versions.get(scope.memory_key_prefix, 0)

    def bump_index_version(self, scope: Optional[MemoryScope] = None) -> int:
        key = "__global__" if scope is None else scope.memory_key_prefix
        self.index_versions[key] = self.index_versions.get(key, 0) + 1
        return self.index_versions[key]


class CrossEncoderReranker:
    """Local reranker stand-in. Swap this for Cohere/Jina/bge-reranker in production."""

    def rerank(self, query: str, items: Sequence[MemoryItem], limit: int) -> List[MemoryItem]:
        q_tokens = set(tokenize(query))

        def score(item: MemoryItem) -> float:
            item_tokens = set(tokenize(item.text))
            exact_overlap = len(q_tokens.intersection(item_tokens)) / max(1, len(q_tokens))
            phrase_bonus = 0.15 if query.lower() in item.text.lower() else 0.0
            return exact_overlap + phrase_bonus + (0.10 * item.importance) + (0.10 * item.feedback_score)

        return sorted(items, key=score, reverse=True)[:limit]


@dataclass
class FeedbackEvent:
    id: str
    item_ids: List[str]
    value: int
    reason: str
    created_at: float
    scope_key: Optional[str] = None


class FeedbackStore:
    def __init__(self) -> None:
        self.events: List[FeedbackEvent] = []

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
        self.events.append(event)
        return event

    def by_item(self, item_id: str) -> List[FeedbackEvent]:
        return [event for event in self.events if item_id in event.item_ids]


@dataclass
class StageTiming:
    name: str
    latency_ms: float


class StageTimer:
    def __init__(self) -> None:
        self.timings: List[StageTiming] = []

    def measure(self, name: str, fn):
        start = time.perf_counter()
        result = fn()
        self.timings.append(StageTiming(name, (time.perf_counter() - start) * 1000))
        return result


@dataclass
class UserProfile:
    tone: str = "direct"
    detail_level: str = "medium"
    preferred_format: str = "concise bullets"


@dataclass
class ContextBlock:
    item_id: str
    text: str
    score: float
    reason: str
    tokens: int
    protected: bool = False
    memory_class: str = MemoryClass.SEMANTIC.value


@dataclass
class OptimizedContext:
    prompt: str
    blocks: List[ContextBlock]
    cache_hit: bool
    input_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    pipeline_steps: List[str] = field(default_factory=list)
    stage_timings: List[StageTiming] = field(default_factory=list)


class ContextOptimizer:
    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        cache: CacheStore,
        *,
        reranker: Optional[CrossEncoderReranker] = None,
        input_cost_per_million_tokens: float = 2.50,
    ) -> None:
        self.short_term = short_term
        self.long_term = long_term
        self.cache = cache
        self.reranker = reranker or CrossEncoderReranker()
        self.input_cost_per_million_tokens = input_cost_per_million_tokens

    def build(
        self,
        *,
        session_id: Optional[str] = None,
        scope: Optional[MemoryScope] = None,
        query: str,
        profile: UserProfile,
        token_budget: int = 900,
        now: Optional[float] = None,
    ) -> OptimizedContext:
        start = time.perf_counter()
        now = time.time() if now is None else now
        if scope is None and session_id is None:
            raise ValueError("Either session_id or scope is required")
        pipeline_steps = [
            "user_message",
            "session_router",
            "scoped_key",
            "short_term_memory_read",
            "long_term_vector_search",
            "rerank_long_term_candidates",
            "cag_cache_lookup",
        ]
        timer = StageTimer()

        session_key = timer.measure("session_router", lambda: self._session_router(session_id=session_id, scope=scope))
        short_term_items = timer.measure(
            "short_term_memory_read",
            lambda: self._read_short_term_memory(session_key=session_key, scope=scope),
        )
        long_term_items = timer.measure(
            "long_term_vector_search",
            lambda: self._search_long_term_memory(scope=scope, query=query),
        )
        protected_items = timer.measure(
            "protected_exact_fact_lookup",
            lambda: self.long_term.protected_exact_facts(scope=scope),
        )
        reranked_long_term = timer.measure(
            "rerank_long_term_candidates",
            lambda: self._rerank_long_term_candidates(query=query, items=long_term_items),
        )

        cache_key = self._cache_key(session_id, query, profile, token_budget, scope=scope)
        cached = timer.measure("cag_cache_lookup", lambda: self.cache.get(cache_key))
        if isinstance(cached, OptimizedContext):
            return OptimizedContext(
                prompt=cached.prompt,
                blocks=cached.blocks,
                cache_hit=True,
                input_tokens=cached.input_tokens,
                estimated_cost_usd=cached.estimated_cost_usd,
                latency_ms=(time.perf_counter() - start) * 1000,
                pipeline_steps=pipeline_steps,
                stage_timings=timer.timings,
            )

        candidates = timer.measure(
            "context_scorer",
            lambda: self._score_context_blocks(query, now, short_term_items, reranked_long_term, protected_items),
        )
        pipeline_steps.append("context_scorer")
        selected = timer.measure("token_budget_allocator", lambda: self._select_under_budget(candidates, token_budget))
        pipeline_steps.append("token_budget_allocator")
        prompt = timer.measure("prompt_builder", lambda: self._render_prompt(query, profile, selected))
        pipeline_steps.extend(["compression_fallback", "prompt_builder", "llm_ready"])
        input_tokens = token_count(prompt)
        result = OptimizedContext(
            prompt=prompt,
            blocks=selected,
            cache_hit=False,
            input_tokens=input_tokens,
            estimated_cost_usd=(input_tokens / 1_000_000) * self.input_cost_per_million_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            pipeline_steps=pipeline_steps,
            stage_timings=timer.timings,
        )
        timer.measure("cache_write", lambda: self.cache.set(cache_key, result))
        result.stage_timings = timer.timings
        return result

    def _session_router(self, *, session_id: Optional[str], scope: Optional[MemoryScope]) -> str:
        return scope.session_key if scope is not None else str(session_id)

    def _read_short_term_memory(self, *, session_key: str, scope: Optional[MemoryScope]) -> List[MemoryItem]:
        if scope is not None and hasattr(self.short_term, "recent_scoped"):
            return self.short_term.recent_scoped(scope)  # type: ignore[attr-defined]
        return self.short_term.recent(session_key)

    def _search_long_term_memory(self, *, scope: Optional[MemoryScope], query: str) -> List[MemoryItem]:
        if scope is not None:
            return self.long_term.search_scoped(scope, query, limit=50)
        return self.long_term.search(query, limit=50)

    def _rerank_long_term_candidates(self, *, query: str, items: Sequence[MemoryItem]) -> List[MemoryItem]:
        return self.reranker.rerank(query, items, limit=30)

    def _score_context_blocks(
        self,
        query: str,
        now: float,
        recent_items: Sequence[MemoryItem],
        reranked_long_term: Sequence[MemoryItem],
        protected_items: Sequence[MemoryItem],
    ) -> List[ContextBlock]:
        qv = tf_vector(query)
        blocks: List[ContextBlock] = []

        for idx, item in enumerate(reversed(recent_items)):
            recency = max(0.0, 1.0 - (idx / max(1, self.short_term.max_messages)))
            semantic = cosine(qv, item.vector)
            score = 0.35 * semantic + 0.45 * recency + 0.10 * item.importance + 0.10 * item.feedback_score
            blocks.append(
                ContextBlock(
                    item.id,
                    item.text,
                    score,
                    "recent session context",
                    token_count(item.text),
                    item.protected,
                    item.memory_class.value,
                )
            )

        protected_ids = {item.id for item in protected_items}
        for item in protected_items:
            semantic = cosine(qv, item.vector)
            if semantic > 0 or any(tok in tokenize(item.text) for tok in tokenize(query)):
                blocks.append(
                    ContextBlock(
                        item.id,
                        item.text,
                        0.80 + 0.10 * item.importance,
                        "protected exact fact",
                        token_count(item.text),
                        True,
                        item.memory_class.value,
                    )
                )

        for item in reranked_long_term:
            if item.id in protected_ids:
                continue
            age_days = max(0.0, (now - item.created_at) / 86_400)
            recency = 1.0 / (1.0 + age_days / 30.0)
            semantic = cosine(qv, item.vector)
            score = 0.50 * semantic + 0.15 * recency + 0.20 * item.importance + 0.15 * item.feedback_score
            if item.memory_class == MemoryClass.PREFERENCE:
                score += 0.05
            if item.memory_class == MemoryClass.PROCEDURAL:
                score += 0.03
            if score > 0.08:
                blocks.append(
                    ContextBlock(
                        item.id,
                        item.text,
                        score,
                        f"long-term {item.memory_class.value} memory",
                        token_count(item.text),
                        item.protected,
                        item.memory_class.value,
                    )
                )

        deduped: Dict[str, ContextBlock] = {}
        for block in sorted(blocks, key=lambda b: b.score, reverse=True):
            fingerprint = stable_hash(" ".join(tokenize(block.text)[:40]))
            if fingerprint not in deduped:
                deduped[fingerprint] = block
        return sorted(deduped.values(), key=lambda b: b.score, reverse=True)

    def _select_under_budget(self, blocks: Sequence[ContextBlock], token_budget: int) -> List[ContextBlock]:
        selected: List[ContextBlock] = []
        used = 0
        for block in blocks:
            if block.protected:
                if used + block.tokens <= token_budget:
                    selected.append(block)
                    used += block.tokens
                continue
            if block.tokens > token_budget:
                compressed = self._compress(block, max_tokens=max(30, token_budget // 3))
                if compressed.tokens + used <= token_budget:
                    selected.append(compressed)
                    used += compressed.tokens
                continue
            if used + block.tokens <= token_budget:
                selected.append(block)
                used += block.tokens
        return selected

    def _compress(self, block: ContextBlock, max_tokens: int) -> ContextBlock:
        if block.protected:
            return block
        words = tokenize(block.text)
        text = " ".join(words[:max_tokens])
        return ContextBlock(
            block.item_id,
            text,
            block.score * 0.92,
            f"{block.reason}; compressed",
            token_count(text),
            block.protected,
            block.memory_class,
        )

    def _render_prompt(self, query: str, profile: UserProfile, blocks: Sequence[ContextBlock]) -> str:
        context = "\n".join(f"- [{b.reason}; score={b.score:.3f}] {b.text}" for b in blocks)
        return (
            "System: Answer using only relevant memory. If memory is weak, say what is missing.\n"
            f"User style: tone={profile.tone}; detail={profile.detail_level}; format={profile.preferred_format}.\n"
            f"Relevant memory:\n{context if context else '- none'}\n"
            f"Current user query: {query}\n"
            "Assistant:"
        )

    def _cache_key(
        self,
        session_id: Optional[str],
        query: str,
        profile: UserProfile,
        token_budget: int,
        *,
        scope: Optional[MemoryScope] = None,
    ) -> str:
        memory_version = self.long_term.index_version(scope)
        session_key = scope.session_key if scope else str(session_id)
        raw = f"{session_key}|v={memory_version}|{query.lower().strip()}|{profile}|{token_budget}"
        return stable_hash(raw)


def score_answer(candidate: str, expected_facts: Sequence[str]) -> float:
    candidate_tokens = set(tokenize(candidate))
    if not expected_facts:
        return 1.0
    hits = 0
    for fact in expected_facts:
        fact_tokens = set(tokenize(fact))
        if fact_tokens and len(candidate_tokens.intersection(fact_tokens)) / len(fact_tokens) >= 0.6:
            hits += 1
    return hits / len(expected_facts)
