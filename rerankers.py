from __future__ import annotations

from typing import Sequence

from memory_optimizer import CrossEncoderReranker, MemoryItem


class SentenceTransformersCrossEncoderReranker(CrossEncoderReranker):
    """
    Real local cross-encoder reranker.

    Suggested model:
      cross-encoder/ms-marco-MiniLM-L-6-v2

    The import is lazy so deterministic CI does not require sentence-transformers.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, items: Sequence[MemoryItem], limit: int):
        if not items:
            return []
        pairs = [(query, item.text) for item in items]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(items, scores), key=lambda row: float(row[1]), reverse=True)
        return [item for item, _score in ranked[:limit]]
