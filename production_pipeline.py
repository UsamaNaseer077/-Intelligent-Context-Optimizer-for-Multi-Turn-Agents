from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from guardrails import Guardrails
from llamaindex_llm import DeterministicLLM
from memory_optimizer import ContextOptimizer, MemoryScope, UserProfile
from multiagent import AgentDecision, LangChainStyleMultiAgentOrchestrator
from prompt_decomposer import PromptPlan, PromptDecomposer
from telemetry import RequestTrace


@dataclass
class PipelineResponse:
    answer: str
    context_tokens: int
    cache_hit: bool
    selected_blocks: int
    trace: RequestTrace
    prompt_plan: PromptPlan
    agent_decisions: List[AgentDecision]
    guardrail_flags: List[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "answer": self.answer,
            "context_tokens": self.context_tokens,
            "cache_hit": self.cache_hit,
            "selected_blocks": self.selected_blocks,
            "trace": self.trace.as_dict(),
            "prompt_plan": self.prompt_plan.__dict__,
            "agent_decisions": [decision.__dict__ for decision in self.agent_decisions],
            "guardrail_flags": self.guardrail_flags,
        }


class ProductionContextPipeline:
    def __init__(
        self,
        optimizer: ContextOptimizer,
        *,
        llm=None,
        guardrails: Optional[Guardrails] = None,
        decomposer: Optional[PromptDecomposer] = None,
        orchestrator: Optional[LangChainStyleMultiAgentOrchestrator] = None,
    ) -> None:
        self.optimizer = optimizer
        self.llm = llm or DeterministicLLM()
        self.guardrails = guardrails or Guardrails()
        self.decomposer = decomposer or PromptDecomposer()
        self.orchestrator = orchestrator or LangChainStyleMultiAgentOrchestrator()

    def run(
        self,
        *,
        scope: MemoryScope,
        query: str,
        profile: UserProfile,
        token_budget: int = 900,
    ) -> PipelineResponse:
        trace = RequestTrace()
        e2e_start = time.perf_counter()

        with trace.stage("guardrails_input"):
            guardrail = self.guardrails.check(query)
        with trace.stage("prompt_decomposition"):
            prompt_plan = self.decomposer.decompose(guardrail.sanitized_query)
        with trace.stage("multiagent_planning"):
            agent_decisions = self.orchestrator.plan(guardrail, prompt_plan)
        with trace.stage("context_optimizer"):
            context = self.optimizer.build(
                scope=scope,
                query=guardrail.sanitized_query,
                profile=profile,
                token_budget=token_budget,
            )

        llm_start = time.perf_counter()
        first_token_seen = False
        chunks: List[str] = []
        with trace.stage("llm_generation"):
            if hasattr(self.llm, "stream_complete"):
                for chunk in self.llm.stream_complete(context.prompt):
                    if not first_token_seen:
                        trace.first_token_latency_ms = (time.perf_counter() - llm_start) * 1000
                        first_token_seen = True
                    chunks.append(str(chunk))
                answer = "".join(chunks)
            else:
                answer = self.llm.complete(context.prompt)
                trace.first_token_latency_ms = (time.perf_counter() - llm_start) * 1000

        with trace.stage("guardrails_output"):
            answer = self.guardrails.protect_output(answer)

        trace.end_to_end_latency_ms = (time.perf_counter() - e2e_start) * 1000
        return PipelineResponse(
            answer=answer,
            context_tokens=context.input_tokens,
            cache_hit=context.cache_hit,
            selected_blocks=len(context.blocks),
            trace=trace,
            prompt_plan=prompt_plan,
            agent_decisions=agent_decisions,
            guardrail_flags=guardrail.flags,
        )
