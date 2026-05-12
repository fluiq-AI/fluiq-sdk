# Fluiq Python SDK

Instrument any LLM application in two lines. Auto-tracing for OpenAI, Anthropic, Gemini, LangChain, and MCP — plus a `@trace` decorator for everything else. Every run is cost-tracked, security-scanned, and scored on retrieval quality automatically.

---

## Installation

```bash
pip install fluiq
```

After installing, download the spaCy language model used by the PII scanner:

```bash
python -m spacy download en_core_web_lg
```

For semantic reranking (CrossEncoderReranker, HybridReranker) install the optional extra:

```bash
pip install "fluiq[rerank]"
```

**Requires Python 3.9+**

---

## Quickstart

```python
from fluiq import instrument

instrument(api_key="fl_...")

# Every OpenAI / Anthropic / Gemini / LangChain / MCP call is now traced.
```

Set `FLUIQ_API_KEY` in your environment and call `instrument()` with no arguments:

```python
import os
from fluiq import instrument

instrument(api_key=os.getenv("FLUIQ_API_KEY"))
```

---

## Auto-instrumentation

`instrument()` patches every supported provider it finds on import. If a provider isn't installed the corresponding patch is skipped silently — no feature flags required.

### OpenAI

```python
import openai
from fluiq import instrument

instrument(api_key="fl_...")

client = openai.OpenAI()
client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Covers chat completions, responses, parse, streaming, embeddings, images, and audio — sync and async.

### Anthropic

```python
import anthropic
from fluiq import instrument

instrument(api_key="fl_...")

client = anthropic.Anthropic()
client.messages.create(
    model="claude-opus-4-7",
    max_tokens=512,
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Gemini / Vertex AI

```python
from google import genai
from fluiq import instrument

instrument(api_key="fl_...")

client = genai.Client()
client.models.generate_content(model="gemini-2.5-pro", contents="Hello")
```

### LangChain

```python
from langchain_openai import ChatOpenAI
from fluiq import instrument

instrument(api_key="fl_...")

llm = ChatOpenAI(model="gpt-4o")
llm.invoke("Hello")
```

Chains, agents, and retrievers all emit traces automatically.

### MCP

```python
from fluiq import instrument

instrument(api_key="fl_...")

# Any MCP client.initialize() call is now traced
# alongside the LLM that invokes the tool.
```

---

## Custom tracing

Wrap any function with `@trace` to record its inputs, outputs, latency, and errors. Async functions are detected and awaited automatically. Nested calls preserve parent/child relationships.

```python
from fluiq import instrument, trace

instrument(api_key="fl_...")

@trace
def retrieve(question: str) -> list[str]:
    return vector_store.similarity_search(question, k=4)

@trace
async def answer(question: str) -> str:
    docs = retrieve(question)          # nested span
    return await llm.ainvoke(prompt(question, docs))
```

Override the span name shown in the dashboard:

```python
@trace(name="research_agent")
def run(question: str) -> str:
    ...
```

> **Fail-open by design.** Every span emission is wrapped in a safety guard so a Fluiq SDK error never crashes your application. Network failures, malformed payloads, and missing optional dependencies are absorbed silently — your decorated function still returns its real result.

---

## Tracing agents

An *agent* is any function or chain you want to monitor as a single unit of work. Wrap the entrypoint with `@trace` so every nested LLM call, tool invocation, and retrieval step is grouped under one root.

### Plain Python agents

```python
from fluiq import instrument, trace

instrument(api_key="fl_...")

@trace
def run_research_agent(question: str) -> str:
    plan = planner(question)         # nested @trace
    docs = retrieve(plan)            # nested @trace
    return synthesize(question, docs)
```

### LangGraph

```python
from langgraph.graph import StateGraph
from fluiq import instrument

instrument(api_key="fl_...")

graph = StateGraph(AgentState)
graph.add_node("planner", planner_node)
graph.add_node("tool_executor", tool_node)
app = graph.compile()
app.invoke({"messages": [...]})
```

Each node emits its own span tagged with `langgraph_node`. The dashboard shows one row per node so you can see which step drives cost.

### LangChain agents

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from fluiq import instrument

instrument(api_key="fl_...")

executor = AgentExecutor(agent=create_openai_tools_agent(...), tools=[...])
executor.invoke({"input": "What's the weather in Paris?"})
```

No decorator needed. The LangChain integration emits a root span for the runnable and child spans for every internal step.

---

## Security scanning

Every traced prompt and response is scanned automatically for PII, prompt injection, and leaked secrets. Scanning runs entirely in your process — no data leaves your environment.

Three scanners run on each trace:

| Scanner | Dependencies | Detects |
|---|---|---|
| **PII** | `presidio-analyzer`, `presidio-anonymizer`, spaCy model | Credit cards, SSNs, IBANs, email, phone, IP, names, API keys |
| **Injection** | none | "ignore previous instructions", jailbreak phrases, persona overrides, DAN |
| **Secrets** | none | OpenAI / Anthropic / AWS / GitHub / Stripe key patterns, high-entropy tokens |

Risk levels:

| Level | Score | Behaviour |
|---|---|---|
| `clean` | < 0.3 | No action |
| `low` | 0.3 – 0.49 | Flagged in dashboard |
| `medium` | 0.5 – 0.89 | Flagged in dashboard |
| `high` | ≥ 0.9 | Prompt and response are **auto-redacted** before storage |

Security findings are stored with every trace and visible on the **Security** tab in the Traces and Agents drawers.

Disable scanning globally:

```python
instrument(api_key="fl_...", security_scan=False)
```

Use the scanners standalone:

```python
from fluiq.security import FluiqPIIScanner, FluiqInjectionScanner, FluiqSecretScanner

pii    = FluiqPIIScanner()
inject = FluiqInjectionScanner()
secret = FluiqSecretScanner()

pii_result    = pii.scan("My SSN is 123-45-6789")
inject_result = inject.scan("Ignore previous instructions and...")
secret_result = secret.scan("key=sk-abc123...")

print(pii_result.risk_level)      # RiskLevel.HIGH
print(inject_result.detected)     # True
print(secret_result.secret_types) # ['openai_key']
```

---

## Optimization

`fluiq.optimization` ships rerankers, caches, context shapers, and query transforms that drop into any RAG pipeline. All run in-process with no extra service.

### Auto-optimize (one call)

```python
from openai import OpenAI
from fluiq.optimization import auto_optimize

client = OpenAI()
opt = auto_optimize(
    embed_fn=lambda texts: [
        d.embedding for d in client.embeddings.create(
            model="text-embedding-3-small", input=texts
        ).data
    ],
    llm_fn=lambda prompt, **kw: client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        **kw,
    ).choices[0].message.content,
    embed_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    cache_dir=".fluiq-cache",
    rerank="hybrid",
)

vectors = opt.embed(["chunk one", "chunk two"])
top_5   = opt.rerank(question, candidates, top_k=5)
answer  = opt.ask("Summarize: ...", temperature=0)
```

`auto_optimize()` returns an `OptimizedRAG` bundle with a shared cache backend, embedding cache, prompt cache, document cache, and reranker — all wired with sensible defaults.

- `cache_dir=".fluiq-cache"` uses `DiskCache` so hits survive process restarts; omit for an in-memory LRU.
- `rerank="hybrid"` (default) falls back to BM25 with a warning when `sentence-transformers` is absent.
- `trace="auto"` (default) emits cache and rerank spans only after `instrument()` has been called.

### Rerankers

```python
from fluiq.optimization import BM25Reranker, CrossEncoderReranker, HybridReranker, MMRReranker

# Keyword (no extra deps)
result = BM25Reranker().rerank(query, candidates, top_k=5)

# Semantic (requires fluiq[rerank])
result = CrossEncoderReranker().rerank(query, candidates, top_k=5)

# Hybrid — fuses keyword + semantic via RRF
result = HybridReranker(fusion="rrf", alpha=0.5).rerank(query, candidates, top_k=5)

# Diversity-aware — re-selects with Maximal Marginal Relevance
result = MMRReranker(base=HybridReranker()).rerank(query, candidates, top_k=5)

context = "\n\n".join(result.texts)
```

`result.documents` exposes per-item `index`, `document`, and `score` when you need the original payloads.

### Caching

```python
from openai import OpenAI
from fluiq.optimization import DiskCache, EmbeddingCache, PromptCache

client = OpenAI()
shared = DiskCache(".fluiq-cache")

embed = EmbeddingCache(
    embed_fn=lambda texts: [
        d.embedding for d in client.embeddings.create(
            model="text-embedding-3-small", input=texts
        ).data
    ],
    model="text-embedding-3-small",
    backend=shared,
)

ask = PromptCache(
    llm_fn=lambda prompt, **kw: client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        **kw,
    ).choices[0].message.content,
    model="gpt-4o-mini",
    backend=shared,
)

vectors = embed(["chunk one", "chunk two"])   # forwards to OpenAI
vectors = embed(["chunk one", "chunk three"]) # only "chunk three" forwards
answer  = ask("Summarize: ...", temperature=0) # cached on second call
```

Four cache types ship in the box:

| Class | Caches by |
|---|---|
| `EmbeddingCache` | `(model, text)` — batches only miss entries |
| `PromptCache` | `(model, prompt, params)` |
| `DocumentCache` | source id — skips re-chunking |
| `ToolCache` | `(name, kwargs)` — dedupes deterministic tool calls |

Two backends are available: `InMemoryCache` (LRU + TTL) and `DiskCache` (file-backed, survives restarts). Implement `BaseCache` to plug in Redis or Memcached.

Pass `trace=True` to any cache constructor to emit hit/miss spans to the dashboard:

```python
embed = EmbeddingCache(..., trace=True)
```

### Context shaping

```python
from fluiq.optimization import auto_optimize

opt = auto_optimize(llm_fn=..., embed_fn=...)

shorter = opt.compress(top_5.texts, query=question)  # drop low-relevance sentences
context = opt.pack(shorter, max_tokens=4000)          # fit into a token budget
answer  = opt.ask(f"{context}\n\nQuestion: {question}", temperature=0)
```

`pack_context` defaults to `reorder="lost-in-middle"` to mitigate attention dropoff at the centre of long contexts. Both helpers are pure Python with no extra dependencies.

### Query transforms

```python
fakes    = opt.hyde("What killed the dinosaurs?")    # hypothetical document embeddings
variants = opt.multi_query("How did the dinosaurs die?", n=4)  # fan-out paraphrases

candidates = [c for q in variants for c in vector_store.similarity_search(q, k=10)]
top_5      = opt.rerank(question, [c.page_content for c in candidates], top_k=5)
```

Both transforms route through `opt.prompts` so identical questions reuse cached rewrites.

### Tool caching for agents

```python
import requests
from fluiq.optimization import auto_optimize

opt = auto_optimize(cache_dir=".fluiq-cache", ...)
opt.register_tool("web_fetch", lambda url: requests.get(url).text)
opt.register_tool("search_db", lambda query, k=5: db.search(query, k=k))

page = opt.tool("web_fetch", url="https://example.com")  # miss → network
page = opt.tool("web_fetch", url="https://example.com")  # hit  → instant
```

---

## Configuration

`instrument()` accepts four parameters. Only `api_key` is required.

```python
def instrument(
    api_key: str,
    *,
    version: str = "v1",
    endpoint: str = "https://api.getfluiq.com/api",
    security_scan: bool = True,
) -> None: ...
```

| Parameter | Default | Description |
|---|---|---|
| `api_key` | — | Your workspace API key. |
| `version` | `"v1"` | Trace schema version. Pin in production to opt in to schema bumps. |
| `endpoint` | `"https://api.getfluiq.com/api"` | Ingest URL. Override for self-hosted deployments. |
| `security_scan` | `True` | Set `False` to disable PII, injection, and secret scanning globally. |

The SDK reads no environment variables on its own — wire `os.getenv("FLUIQ_API_KEY")` yourself if you prefer one.

---

## License

MIT. See [LICENSE](LICENSE).
