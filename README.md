# Fluiq Python SDK
[![PyPI version](https://badge.fury.io/py/fluiq.svg)](https://pypi.org/project/fluiq/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

Instrument any LLM application in two lines. Auto-tracing for OpenAI, Anthropic, Gemini, LangChain, LangGraph, CrewAI, Google ADK, and all major vector stores — plus a `@trace` decorator for everything else. Every run is cost-tracked in your dashboard. Add one more line to enable security scanning, evaluation, or Redis caching.

---

## Installation

```bash
pip install fluiq
```

**Requires Python 3.9+**

---

## Quickstart

```python
import fluiq

fluiq.instrument(api_key="fl_...")

# Every OpenAI / Anthropic / Gemini / LangChain / MCP call is now traced.
```

Use the `FLUIQ_API_KEY` environment variable to avoid hardcoding the key:

```python
import fluiq

fluiq.instrument()  # reads FLUIQ_API_KEY automatically
```

---

## Auto-instrumentation

`instrument()` patches every supported provider it finds at import time. If a provider isn't installed the corresponding patch is skipped silently — no feature flags required.

### OpenAI

```python
import openai
import fluiq

fluiq.instrument(api_key="fl_...")

client = openai.OpenAI()
client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Covers chat completions, responses API, structured outputs (`.parse`), streaming, embeddings, images, and audio — sync and async.

### Anthropic

```python
import anthropic
import fluiq

fluiq.instrument(api_key="fl_...")

client = anthropic.Anthropic()
client.messages.create(
    model="claude-opus-4-7",
    max_tokens=512,
    messages=[{"role": "user", "content": "Hello"}],
)
```

Covers sync and async messages, streaming, `count_tokens`, and beta messages.

### Gemini / Vertex AI

```python
from google import genai
import fluiq

fluiq.instrument(api_key="fl_...")

client = genai.Client()
client.models.generate_content(model="gemini-2.5-pro", contents="Hello")
```

Covers sync and async `generate_content`, streaming, `count_tokens`, and Vertex AI.

### LangChain

```python
from langchain_openai import ChatOpenAI
import fluiq

fluiq.instrument(api_key="fl_...")

llm = ChatOpenAI(model="gpt-4o")
llm.invoke("Hello")
```

Chains, agents, and retrievers all emit traces automatically.

### LangGraph

```python
from langgraph.graph import StateGraph
import fluiq

fluiq.instrument(api_key="fl_...")

graph = StateGraph(AgentState)
graph.add_node("planner", planner_node)
graph.add_node("tool_executor", tool_node)
app = graph.compile()
app.invoke({"messages": [...]})
```

Each node emits its own span. The dashboard shows one row per node so you can see which step drives cost.

### CrewAI

```python
from crewai import Agent, Task, Crew
import fluiq

fluiq.instrument(api_key="fl_...")

crew = Crew(agents=[...], tasks=[...])
crew.kickoff()
```

### Google ADK

```python
import fluiq

fluiq.instrument(api_key="fl_...")

# google.adk agent calls are traced automatically.
```

### Vector stores

Chromadb, Pinecone, Qdrant, Weaviate, and FAISS queries are traced automatically after `instrument()` is called. No extra setup required.

### MCP

```python
import fluiq

fluiq.instrument(api_key="fl_...")

# Any MCP client.initialize() call is traced alongside the LLM that invokes the tool.
```

---

## Custom tracing with `@trace`

Wrap any function with `@trace` to record its inputs, outputs, latency, and errors. Async functions are detected and awaited automatically. Nested calls preserve parent/child relationships.

```python
import fluiq

fluiq.instrument(api_key="fl_...")

@fluiq.trace
def retrieve(question: str) -> list[str]:
    return vector_store.similarity_search(question, k=4)

@fluiq.trace
async def answer(question: str) -> str:
    docs = retrieve(question)          # nested span
    return await llm.ainvoke(prompt(question, docs))
```

Override the span name shown in the dashboard:

```python
@fluiq.trace(name="research_agent")
def run(question: str) -> str:
    ...
```

> **Fail-open by design.** Every span emission is wrapped in a safety guard so a Fluiq error never crashes your application. Network failures, malformed payloads, and missing optional dependencies are absorbed silently.

---

## Security scanning — `fluiq.secure()`

Activate server-side security scanning. Every prompt and response is checked for PII, prompt injection, jailbreaks, skeleton-key attacks, leaked secrets, and indirect injection in tool outputs. Detection runs on the Fluiq backend — patterns are never shipped in the public SDK.

**Requires Team plan or above.**

```python
import fluiq

fluiq.instrument(api_key="fl_...")
fluiq.secure()                   # warn mode (default)
```

In `warn` mode (default) scanning runs after each LLM call. Security fields are written into the stored trace. HIGH-risk content is redacted before persistence. Your application is never interrupted.

In `block` mode every prompt is checked *before* the LLM API call. If an attack is detected a `FluiqSecurityError` is raised and the LLM call is never made.

```python
fluiq.secure(mode="block")
```

```python
from fluiq.exceptions import FluiqSecurityError

try:
    response = client.chat.completions.create(...)
except FluiqSecurityError as e:
    print(e.risk_level)    # "high"
    print(e.attack_types)  # ["jailbreak", "prompt_injection"]
```

**Parameters**

| Parameter | Values | Default | Description |
|---|---|---|---|
| `mode` | `"warn"` \| `"block"` | `"warn"` | `warn`: post-call scan only. `block`: pre-call guard + post-call scan. |

**What gets scanned**

| Category | Detects |
|---|---|
| PII | Credit cards, SSNs, IBANs, email, phone, IP address, names, API keys |
| Prompt injection | Instruction-override patterns, system-prompt leaking, template injection |
| Jailbreak | Role-play escapes, persona hijacks, fictional-framing bypasses, DAN and variants |
| Skeleton key | "Add a mode / unlock capabilities" style attacks |
| Secrets | OpenAI / Anthropic / AWS / GitHub / Stripe key patterns, high-entropy tokens |
| Indirect injection | Injection patterns in tool outputs and retrieved context documents |
| Semantic | Cosine similarity against attack centroids (when sentence-transformers is available) |

Security findings are visible on the **Security** tab in the Traces drawer.

---

## Evaluation — `fluiq.eval()`

Activate server-side LLM-as-judge evaluation. After each LLM call Fluiq scores the response on the requested metrics (0 = worst, 1 = best) and stores the results in your dashboard.

```python
import fluiq

fluiq.instrument(api_key="fl_...")
fluiq.eval()                     # warn mode, default metrics
```

In `warn` mode (default) evaluation runs in a background thread and logs a warning when a score falls below its threshold. Your application is never interrupted.

In `block` mode evaluation runs synchronously after each LLM call. If any metric falls below its threshold a `FluiqEvalError` is raised.

```python
fluiq.eval(
    mode="block",
    thresholds={"hallucination": 0.8, "relevance": 0.7},
    metrics=["hallucination", "relevance", "toxicity"],
    judge_model="gpt-4o-mini",
)
```

```python
from fluiq.exceptions import FluiqEvalError

try:
    response = client.chat.completions.create(...)
except FluiqEvalError as e:
    print(e.failures)  # {"hallucination": 0.61}
    print(e.scores)    # {"hallucination": 0.61, "relevance": 0.94}
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mode` | `"warn"` \| `"block"` | `"warn"` | `warn`: background eval + log. `block`: synchronous eval + raise on failure. |
| `metrics` | `list[str]` | `["hallucination", "relevance"]` | Which metrics to score. |
| `thresholds` | `dict[str, float]` | `{}` | Per-metric pass/fail thresholds (0–1). Warnings / errors only fire when a threshold is set. |
| `judge_model` | `str` | `"gpt-4o-mini"` | The model Fluiq uses as judge. |

**Supported metrics**

`hallucination`, `faithfulness`, `relevance`, `toxicity`, `coherence`, `completeness`

---

## Trace-driven caching — `fluiq.optimize()`

Activate trace-driven Redis caching. Fluiq's backend analyses your historical traces to find which LLM calls repeat most often, provisions a dedicated Redis instance for your account, and serves repeated prompts from cache — saving both latency and LLM cost.

**Requires Team plan or above. Must be called after `fluiq.instrument()`.**

```python
import fluiq

fluiq.instrument(api_key="fl_...")
fluiq.optimize()                 # cache mode (default)
```

On the first LLM call after `optimize()` the SDK fetches the cache profile (which models to cache, TTL, Redis URL) and connects. Repeated identical prompts are then served from Redis without hitting the LLM API.

Use `mode="observe"` to record what *would* have been a cache hit without actually intercepting calls — useful to review potential savings before enabling full caching:

```python
fluiq.optimize(mode="observe")
```

**Parameters**

| Parameter | Values | Default | Description |
|---|---|---|---|
| `mode` | `"cache"` \| `"observe"` | `"cache"` | `cache`: intercept repeated calls and serve from Redis. `observe`: record potential hits only, no interception. |

Cache hits and misses are visible on the **Optimize** tab in your dashboard.

---

## Combining features

All four features compose freely:

```python
import fluiq

fluiq.instrument(api_key="fl_...")
fluiq.secure(mode="block")
fluiq.eval(thresholds={"hallucination": 0.8}, mode="warn")
fluiq.optimize()
```

Call order per LLM request:
1. **secure (block)** — pre-call prompt check; raises `FluiqSecurityError` if blocked
2. **optimize** — cache lookup; returns immediately on hit, no LLM call made
3. LLM API call
4. **secure (warn)** — post-call scan; enriches trace with security fields
5. **eval** — evaluation in background thread (warn) or synchronously (block)

---

## Configuration reference

```python
fluiq.instrument(
    api_key: str = os.getenv("FLUIQ_API_KEY"),
    *,
    endpoint: str = "https://api.getfluiq.com/api",
    version:  str = "v1",
)
```

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `FLUIQ_API_KEY` env var | Your workspace API key. |
| `endpoint` | `https://api.getfluiq.com/api` | Ingest URL. Override for self-hosted deployments. Set via `FLUIQ_API_ENDPOINT`. |
| `version` | `"v1"` | Trace schema version. Pin in production to opt in to schema bumps explicitly. |

**Environment variables**

| Variable | Description |
|---|---|
| `FLUIQ_API_KEY` | Default API key used by `instrument()`. |
| `FLUIQ_API_ENDPOINT` | Default endpoint URL. |

---

## Exceptions

| Exception | Raised when |
|---|---|
| `FluiqSecurityError` | `fluiq.secure(mode="block")` is active and the pre-call check returns a HIGH-risk result. |
| `FluiqEvalError` | `fluiq.eval(mode="block")` is active and one or more metrics fall below their threshold. |

Both exceptions are importable from `fluiq.exceptions` or directly from `fluiq`:

```python
from fluiq.exceptions import FluiqSecurityError, FluiqEvalError
```

---

## Plan requirements

| Feature | Minimum plan |
|---|---|
| `fluiq.instrument()` | Free |
| `@fluiq.trace` | Free |
| `fluiq.eval()` | Free |
| `fluiq.secure()` | Team |
| `fluiq.optimize()` | Team |

Free-tier keys that call `fluiq.secure()` or `fluiq.optimize()` receive a 402 response and fall back to no-op behaviour automatically — your application continues to run unaffected.

---

## License

MIT. See [LICENSE](LICENSE).
