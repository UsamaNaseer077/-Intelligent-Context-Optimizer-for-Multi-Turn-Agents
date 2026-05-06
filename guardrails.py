from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


PROMPT_INJECTION_RE = re.compile(
    r"(ignore (all )?(previous|system) instructions|reveal.*system prompt|developer message|exfiltrate|jailbreak)",
    re.IGNORECASE,
)
SECRET_RE = re.compile(r"(api[_-]?key|password|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{8,}", re.IGNORECASE)
PII_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|\b\d{3}-\d{2}-\d{4}\b")


@dataclass
class GuardrailDecision:
    allowed: bool
    sanitized_query: str
    flags: List[str]
    action: str


class Guardrails:
    def check(self, query: str) -> GuardrailDecision:
        flags: List[str] = []
        sanitized = query
        if PROMPT_INJECTION_RE.search(query):
            flags.append("prompt_injection")
        if SECRET_RE.search(query):
            flags.append("secret_like_input")
            sanitized = SECRET_RE.sub("[REDACTED_SECRET]", sanitized)
        if PII_RE.search(query):
            flags.append("pii_like_input")
            sanitized = PII_RE.sub("[REDACTED_PII]", sanitized)
        if "prompt_injection" in flags:
            return GuardrailDecision(True, sanitized, flags, "continue_with_memory_policy")
        return GuardrailDecision(True, sanitized, flags, "continue")

    def protect_output(self, text: str) -> str:
        text = SECRET_RE.sub("[REDACTED_SECRET]", text)
        return PII_RE.sub("[REDACTED_PII]", text)
