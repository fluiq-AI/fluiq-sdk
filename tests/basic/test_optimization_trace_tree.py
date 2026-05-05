"""
Contract: ``auto_optimize()`` caching and reranking must be visible in the
trace tree. Each cache lookup and each reranker invocation must emit a
``LogTrace`` event with ``parent_id`` linked to the surrounding
``@trace``-decorated function (or LLM context) so the dashboard renders
them as proper children rather than orphaned root rows.

Run:  python -m unittest tests.basic.test_optimization_trace_tree
"""
import unittest
from unittest.mock import patch

from fluiq import trace
from fluiq.optimization import auto_optimize
from fluiq.optimization.rerankers import (
    BM25Reranker,
    HybridReranker,
    MMRReranker,
)
from fluiq.optimization.rerankers._traced import TracedReranker, apply_tracing


def _capture_events():
    """Patch ``send_event`` and return the list events get appended to."""
    events: list = []
    return events, patch(
        "fluiq.tracer.send_event", side_effect=lambda data: events.append(data)
    )


class RerankerTraceEmission(unittest.TestCase):
    def test_bm25_emits_rerank_span(self):
        events, p = _capture_events()
        reranker = TracedReranker(BM25Reranker())
        with p:
            res = reranker.rerank(
                "machine learning",
                ["the quick brown fox", "machine learning is fun", "hello"],
                top_k=2,
            )
        self.assertEqual(len(res.documents), 2)
        rerank_events = [e for e in events if e.get("type") == "rerank"]
        self.assertEqual(len(rerank_events), 1)
        ev = rerank_events[0]
        self.assertEqual(ev["reranker"], "bm25")
        self.assertEqual(ev["input_count"], 3)
        self.assertEqual(ev["output_count"], 2)
        self.assertEqual(ev["top_k"], 2)
        self.assertTrue(ev["success"])
        self.assertIn("trace_id", ev)
        self.assertIn("latency", ev)

    def test_hybrid_emits_parent_with_inner_children(self):
        # apply_tracing wraps Hybrid + its keyword/semantic legs; the inner
        # BM25 leg's span must point at the Hybrid span as parent.
        events, p = _capture_events()
        # Use a Hybrid with two BM25 legs to avoid sentence-transformers dep.
        hybrid = HybridReranker(
            keyword=BM25Reranker(),
            semantic=BM25Reranker(),
            fusion="rrf",
        )
        traced = apply_tracing(hybrid)
        with p:
            traced.rerank("ml", ["a", "ml is fun", "b"], top_k=2)
        rerank_events = [e for e in events if e.get("type") == "rerank"]
        # 1 outer hybrid + 2 inner bm25
        self.assertEqual(len(rerank_events), 3)
        outer = next(e for e in rerank_events if e["reranker"] == "hybrid")
        inner = [e for e in rerank_events if e["reranker"] == "bm25"]
        self.assertEqual(len(inner), 2)
        for child in inner:
            self.assertEqual(child["parent_id"], outer["trace_id"])

    def test_mmr_emits_parent_with_relevance_child(self):
        events, p = _capture_events()
        mmr = MMRReranker(relevance=BM25Reranker(), lambda_mult=0.5)
        traced = apply_tracing(mmr)
        with p:
            traced.rerank("ml", ["ml is great", "cats", "machine learning"], top_k=2)
        rerank_events = [e for e in events if e.get("type") == "rerank"]
        self.assertEqual(len(rerank_events), 2)
        outer = next(e for e in rerank_events if e["reranker"] == "mmr")
        inner = next(e for e in rerank_events if e["reranker"] == "bm25")
        self.assertEqual(inner["parent_id"], outer["trace_id"])

    def test_reranker_failure_still_emits_span_and_propagates(self):
        events, p = _capture_events()

        class Boom(BM25Reranker):
            def rerank(self, *a, **kw):
                raise RuntimeError("rerank exploded")

        traced = TracedReranker(Boom())
        with p:
            with self.assertRaises(RuntimeError):
                traced.rerank("q", ["a", "b"], top_k=1)
        rerank_events = [e for e in events if e.get("type") == "rerank"]
        self.assertEqual(len(rerank_events), 1)
        self.assertFalse(rerank_events[0]["success"])
        self.assertIn("rerank exploded", rerank_events[0]["error"])


class TraceTreeIntegration(unittest.TestCase):
    def test_rerank_inside_trace_decorator_links_to_parent(self):
        opt = auto_optimize(rerank="bm25", trace=True)
        events, p = _capture_events()

        @trace
        def answer(q):
            return opt.rerank(q, ["foo", "bar baz", "ml"], top_k=2).texts

        with p:
            answer("ml")
        rerank_events = [e for e in events if e.get("type") == "rerank"]
        fn_events = [e for e in events if e.get("type") == "function"]
        self.assertGreaterEqual(len(rerank_events), 1)
        self.assertGreaterEqual(len(fn_events), 1)
        # The @trace function pushes its trace_id; the rerank span emits with
        # parent_id = that pushed id, producing a real parent->child link.
        fn_trace_id = fn_events[0]["trace_id"]
        self.assertTrue(
            any(e.get("parent_id") == fn_trace_id for e in rerank_events),
            f"no rerank event linked to @trace function {fn_trace_id!r}",
        )

    def test_embedding_cache_inside_trace_decorator_links_to_parent(self):
        calls = []

        def embed_fn(texts):
            calls.append(list(texts))
            return [[0.1, 0.2, 0.3] for _ in texts]

        opt = auto_optimize(embed_fn=embed_fn, trace=True)
        events, p = _capture_events()

        @trace
        def answer(q):
            return opt.embed([q, q + "!"])

        with p:
            answer("hello")
        cache_events = [e for e in events if e.get("type") == "cache"]
        fn_events = [e for e in events if e.get("type") == "function"]
        self.assertGreaterEqual(len(cache_events), 1)
        self.assertGreaterEqual(len(fn_events), 1)
        fn_trace_id = fn_events[0]["trace_id"]
        self.assertTrue(
            any(e.get("parent_id") == fn_trace_id for e in cache_events),
            f"no cache event linked to @trace function {fn_trace_id!r}",
        )


if __name__ == "__main__":
    unittest.main()
