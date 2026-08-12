# Changelog

All notable changes to the Fluiq Python SDK.

This project follows [Semantic Versioning](https://semver.org/).

## 0.3.0 — 2026-08-12

### Removed — breaking

The optimization pillar is gone. Fluiq now stands on three pillars: secure,
observe, evaluate, with dataset and prompt management inside evaluation.

- **`fluiq.optimize()` is removed.** There is no replacement. Delete the call;
  everything else in your integration keeps working unchanged.
- **`fluiq.lookup_tool_result()` is removed.** It read the tool cache, which no
  longer exists.
- The `fluiq.optimization` package is removed, along with the pre-call optimize
  gate, the tool cache, and the Anthropic `cache_control` injection helper.
- MCP `list_tools()` / `call_tool()` are still patched and still traced; only the
  caching layer around them is gone.
- Vector-store integrations (Chroma, Pinecone, Qdrant, Weaviate, FAISS) keep
  full tracing; their cached and cache-invalidating wrappers collapse into the
  plain traced wrappers.

### Migration

```diff
  import fluiq

  fluiq.instrument(api_key="fl_...")
  fluiq.secure(mode="block")
- fluiq.optimize()
  fluiq.eval(thresholds={"hallucination": 0.8})
```

If you called `fluiq.lookup_tool_result(name, args)`, call your tool directly.

### Kept deliberately

Provider prompt-cache **token capture** stays: Anthropic
`prompt_cache_read_tokens` / `prompt_cache_creation_tokens`, and OpenAI and
Gemini `prompt_cached_tokens`. That is cost accuracy — it makes reported spend
match the provider bill — not a caching product. The SDK injects nothing and
serves nothing from a cache.

### Changed

- The CI eval gate endpoint moved from `GET /api/v1/optimize/evals` to
  `GET /api/v1/evaluate/recent-evals`. `python -m fluiq.ci` is unaffected; it
  uses `/ci/eval-runs` and always did.

## 0.2.1

Prior releases are not catalogued here.
