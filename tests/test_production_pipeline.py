import unittest

from eval_harness import seed_memory
from guardrails import Guardrails
from memory_optimizer import ContextOptimizer, LongTermMemory, MemoryScope, ShortTermMemory, TTLCache, UserProfile
from production_pipeline import ProductionContextPipeline


class ProductionPipelineTests(unittest.TestCase):
    def setUp(self):
        short = ShortTermMemory(max_messages=6)
        long = LongTermMemory()
        seed_memory(short, long)
        optimizer = ContextOptimizer(short, long, TTLCache())
        self.pipeline = ProductionContextPipeline(optimizer)
        self.scope = MemoryScope("tenant", "user", "project", "s1")
        self.profile = UserProfile()

    def test_guardrails_redact_pii(self):
        decision = Guardrails().check("My email is test@example.com; what is the deadline?")

        self.assertIn("pii_like_input", decision.flags)
        self.assertIn("[REDACTED_PII]", decision.sanitized_query)

    def test_pipeline_records_ttft_and_e2e_latency(self):
        result = self.pipeline.run(
            scope=self.scope,
            query="When should I submit the LEC assignment?",
            profile=self.profile,
            token_budget=120,
        )

        self.assertGreaterEqual(result.trace.first_token_latency_ms, 0)
        self.assertGreater(result.trace.end_to_end_latency_ms, 0)
        stage_names = [span.name for span in result.trace.spans]
        self.assertIn("guardrails_input", stage_names)
        self.assertIn("prompt_decomposition", stage_names)
        self.assertIn("multiagent_planning", stage_names)
        self.assertIn("context_optimizer", stage_names)


if __name__ == "__main__":
    unittest.main()
