from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from guardrails import GuardrailDecision
from prompt_decomposer import PromptPlan


@dataclass
class AgentDecision:
    agent: str
    decision: str
    metadata: Dict[str, object]


class LangChainStyleMultiAgentOrchestrator:
    """
    Production-shaped multi-agent controller.

    The implementation is intentionally deterministic for CI, but mirrors the
    LangChain pattern of narrow runnable units with typed inputs/outputs. In a
    full deployment, each method can become a LangChain Runnable or LangGraph
    node with retries, tracing, and human review gates.
    """

    def plan(self, guardrail: GuardrailDecision, prompt_plan: PromptPlan) -> List[AgentDecision]:
        decisions = [
            AgentDecision(
                "privacy_policy_agent",
                guardrail.action,
                {"flags": guardrail.flags},
            ),
            AgentDecision(
                "retrieval_planner_agent",
                "route_query",
                {
                    "subqueries": prompt_plan.subqueries,
                    "exact_facts": prompt_plan.requires_exact_facts,
                    "preferences": prompt_plan.requires_preferences,
                    "procedures": prompt_plan.requires_procedures,
                },
            ),
            AgentDecision(
                "context_optimizer_agent",
                "build_budgeted_context",
                {"max_subqueries": len(prompt_plan.subqueries)},
            ),
        ]
        if guardrail.flags:
            decisions.append(
                AgentDecision(
                    "evaluation_agent",
                    "mark_for_review",
                    {"reason": "guardrail_flags_present"},
                )
            )
        return decisions

    def as_langchain_runnable(self):
        """
        Return a LangChain Runnable when langchain-core is installed.

        This keeps CI dependency-light while making the production graph directly
        pluggable into LangChain/LangGraph deployments.
        """
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(lambda inputs: self.plan(inputs["guardrail"], inputs["prompt_plan"]))
