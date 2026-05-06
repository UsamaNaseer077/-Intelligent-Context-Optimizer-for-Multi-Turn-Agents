import unittest

from memory_optimizer import (
    ContextOptimizer,
    FeedbackStore,
    LongTermMemory,
    MemoryClass,
    MemoryScope,
    ShortTermMemory,
    TTLCache,
    UserProfile,
)


class MemoryOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.short = ShortTermMemory(max_messages=3)
        self.long = LongTermMemory()
        self.cache = TTLCache(max_items=2, ttl_seconds=60)
        self.optimizer = ContextOptimizer(self.short, self.long, self.cache)
        self.profile = UserProfile(tone="direct", detail_level="medium", preferred_format="bullets")

    def test_short_term_memory_keeps_recent_window(self):
        for i in range(5):
            self.short.add("s", f"message {i}", "user", now=i)

        recent = self.short.recent("s")

        self.assertEqual([m.text for m in recent], ["message 2", "message 3", "message 4"])

    def test_long_term_upsert_deduplicates_unchanged_memory(self):
        first = self.long.upsert("Redis stores short-term session memory.", importance=0.4)
        second = self.long.upsert("Redis stores short-term session memory.", importance=0.9)

        self.assertEqual(first.id, second.id)
        self.assertEqual(len(self.long.items), 1)
        self.assertEqual(second.importance, 0.9)

    def test_relevant_long_term_memory_is_selected(self):
        self.long.upsert("The absolute cutoff is Friday 8 May 2026 end of day UK time.", importance=0.9)
        self.long.upsert("The UI should use thumbs up and thumbs down feedback.", importance=0.5)

        result = self.optimizer.build(
            session_id="s",
            query="What is the final cutoff date?",
            profile=self.profile,
            token_budget=80,
        )

        self.assertIn("Friday 8 May 2026", result.prompt)
        self.assertFalse(result.cache_hit)

    def test_cag_cache_hits_on_repeated_query(self):
        self.long.upsert("CAG cache reuses stable context for repeated queries.", importance=0.9)

        first = self.optimizer.build(session_id="s", query="How does CAG help?", profile=self.profile, token_budget=80)
        second = self.optimizer.build(session_id="s", query="How does CAG help?", profile=self.profile, token_budget=80)

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.prompt, second.prompt)

    def test_feedback_changes_memory_rank(self):
        good = self.long.upsert("Thumbs up increases memory confidence.", importance=0.5)
        bad = self.long.upsert("Thumbs down should be ignored forever.", importance=0.5)

        self.long.apply_feedback([good.id], 1)
        self.long.apply_feedback([bad.id], -1)

        self.assertGreater(self.long.items[good.id].feedback_score, 0)
        self.assertLess(self.long.items[bad.id].feedback_score, 0)

    def test_large_memory_is_compressed_under_budget(self):
        long_text = " ".join(["critical"] * 200)
        self.long.upsert(long_text, importance=1.0)

        result = self.optimizer.build(
            session_id="s",
            query="critical",
            profile=self.profile,
            token_budget=50,
        )

        self.assertLessEqual(sum(block.tokens for block in result.blocks), 50)
        self.assertIn("compressed", result.blocks[0].reason)

    def test_empty_memory_degrades_gracefully(self):
        result = self.optimizer.build(session_id="missing", query="unknown thing", profile=self.profile, token_budget=50)

        self.assertIn("- none", result.prompt)
        self.assertEqual(result.blocks, [])

    def test_cache_eviction_respects_max_items(self):
        self.cache.set("a", 1)
        self.cache.set("b", 2)
        self.cache.set("c", 3)

        self.assertIsNone(self.cache.get("a"))
        self.assertEqual(self.cache.get("b"), 2)
        self.assertEqual(self.cache.get("c"), 3)

    def test_tenant_sessions_do_not_share_short_term_memory(self):
        self.short.add("tenant-a", "secret project alpha", "user")
        self.short.add("tenant-b", "public project beta", "user")

        result = self.optimizer.build(
            session_id="tenant-b",
            query="What project is this?",
            profile=self.profile,
            token_budget=80,
        )

        self.assertIn("public project beta", result.prompt)
        self.assertNotIn("secret project alpha", result.prompt)

    def test_prompt_contains_style_profile(self):
        result = self.optimizer.build(
            session_id="s",
            query="How should you answer?",
            profile=UserProfile(tone="warm", detail_level="short", preferred_format="table"),
            token_budget=50,
        )

        self.assertIn("tone=warm", result.prompt)
        self.assertIn("detail=short", result.prompt)
        self.assertIn("format=table", result.prompt)

    def test_scoped_long_term_memory_does_not_cross_tenants(self):
        scope_a = MemoryScope("tenant-a", "user-1", "project-1", "session-1")
        scope_b = MemoryScope("tenant-b", "user-1", "project-1", "session-1")
        self.long.upsert_scoped(scope_a, "Private tenant A deadline is Friday.", importance=1.0)
        self.long.upsert_scoped(scope_b, "Tenant B uses Qdrant for long-term memory.", importance=1.0)

        result = self.optimizer.build(
            scope=scope_b,
            query="What memory backend is used?",
            profile=self.profile,
            token_budget=80,
        )

        self.assertIn("Qdrant", result.prompt)
        self.assertNotIn("tenant A deadline", result.prompt)

    def test_cache_invalidates_when_memory_index_version_changes(self):
        scope = MemoryScope("tenant", "user", "project", "session")
        self.long.upsert_scoped(scope, "Initial memory says use Redis.", importance=0.8)

        first = self.optimizer.build(scope=scope, query="What storage should I use?", profile=self.profile)
        second = self.optimizer.build(scope=scope, query="What storage should I use?", profile=self.profile)
        self.long.upsert_scoped(scope, "New memory says use pgvector for long-term memory.", importance=1.0)
        third = self.optimizer.build(scope=scope, query="What storage should I use?", profile=self.profile)

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertFalse(third.cache_hit)
        self.assertIn("pgvector", third.prompt)

    def test_reranker_promotes_exact_query_overlap(self):
        self.long.upsert("Generic memory backend information.", importance=1.0)
        self.long.upsert("Qdrant is the vector database for memory retrieval.", importance=0.3)

        result = self.optimizer.build(
            session_id="s",
            query="Qdrant vector database",
            profile=self.profile,
            token_budget=40,
        )

        self.assertIn("Qdrant", result.blocks[0].text)

    def test_pipeline_trace_matches_architecture_order(self):
        result = self.optimizer.build(
            session_id="s",
            query="What should happen?",
            profile=self.profile,
            token_budget=80,
        )

        self.assertEqual(
            result.pipeline_steps,
            [
                "user_message",
                "session_router",
                "scoped_key",
                "short_term_memory_read",
                "long_term_vector_search",
                "rerank_long_term_candidates",
                "cag_cache_lookup",
                "context_scorer",
                "token_budget_allocator",
                "compression_fallback",
                "prompt_builder",
                "llm_ready",
            ],
        )

    def test_exact_fact_memory_is_classified_and_protected(self):
        item = self.long.upsert("Final cutoff is Friday 8 May 2026 end of day UK time.", importance=0.9)

        self.assertEqual(item.memory_class, MemoryClass.EXACT_FACT)
        self.assertTrue(item.protected)

    def test_protected_exact_facts_are_not_compressed_away(self):
        self.long.upsert("Final cutoff is Friday 8 May 2026 end of day UK time.", importance=1.0)
        self.long.upsert(" ".join(["noise"] * 200), importance=0.9)

        result = self.optimizer.build(
            session_id="s",
            query="What is the final cutoff?",
            profile=self.profile,
            token_budget=40,
        )

        self.assertIn("Friday 8 May 2026", result.prompt)
        self.assertTrue(any(block.protected for block in result.blocks))

    def test_feedback_events_are_separate_from_memory_facts(self):
        item = self.long.upsert("Use direct style.", importance=0.8)
        store = FeedbackStore()

        event = store.record([item.id], -1, reason="too verbose")
        self.long.apply_feedback([item.id], -1)

        self.assertEqual(event.reason, "too verbose")
        self.assertEqual(len(store.by_item(item.id)), 1)
        self.assertEqual(self.long.items[item.id].text, "Use direct style.")

    def test_stage_timings_are_recorded(self):
        self.long.upsert("Qdrant stores long-term semantic memory.", importance=0.9)

        result = self.optimizer.build(
            session_id="s",
            query="Where is semantic memory stored?",
            profile=self.profile,
            token_budget=80,
        )

        names = [timing.name for timing in result.stage_timings]
        self.assertIn("long_term_vector_search", names)
        self.assertIn("context_scorer", names)


if __name__ == "__main__":
    unittest.main()
