from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class PromptPlan:
    original_query: str
    subqueries: List[str]
    requires_exact_facts: bool
    requires_preferences: bool
    requires_procedures: bool


class PromptDecomposer:
    def decompose(self, query: str) -> PromptPlan:
        parts = [part.strip() for part in re.split(r"\band\b|[?;]", query, flags=re.IGNORECASE) if part.strip()]
        if not parts:
            parts = [query.strip()]
        lowered = query.lower()
        return PromptPlan(
            original_query=query,
            subqueries=parts[:4],
            requires_exact_facts=any(word in lowered for word in ["date", "deadline", "cutoff", "price", "id", "policy"]),
            requires_preferences=any(word in lowered for word in ["style", "tone", "prefer", "adapt"]),
            requires_procedures=any(word in lowered for word in ["how", "steps", "workflow", "run", "deploy"]),
        )
