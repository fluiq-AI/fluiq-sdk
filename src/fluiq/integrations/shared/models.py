from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict
from enum import Enum

class TraceType(str, Enum):
    General_Function: str = "OTHERFUNCTION"
    Gemini: str = "GEMINI"
    OpenAI: str = "OPENAI"
    Anthropic: str = "ANTHROPIC"
    LangChain: str = "LANGCHAIN"
    LangGraph: str = "LANGGRAPH"
    LlamaIndex: str = "LLAMAINDEX"
    CrewAI: str = "CREWAI"
    GoogleADK: str = "GOOGLEADK"
    AutoGen: str = "AUTOGEN"
    AgentToAgent: str = "AGENTTOAGENT"
    ChromaDB: str = "CHROMADB"
    Pinecone: str = "PINECONE"
    Qdrant: str = "QDRANT"
    Weaviate: str = "WEAVIATE"
    FAISS: str = "FAISS"

class Tokens(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: Optional[int] = None
    completion: Optional[int] = None
    total: Optional[int] = None


class LogTrace(BaseModel):
    integration: Optional[TraceType] = ""
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    type: Optional[str] = None
    timestamp: Optional[float] = None
    latency: Optional[float] = None
    tokens: Optional[Tokens] = None

    model: Optional[str] = None
    api: Optional[str] = None
    messages: Optional[Any] = None
    contents: Optional[Any] = None
    input: Optional[Any] = None
    system: Optional[Any] = None
    system_instruction: Optional[Any] = None
    tools: Optional[Any] = None
    tool_choice: Optional[Any] = None
    tool_config: Optional[Any] = None

    response: Optional[Any] = None
    thinking: Optional[List[Any]] = None
    tool_calls: Optional[List[Any]] = None
    tool_uses: Optional[List[Any]] = None
    function_calls: Optional[List[Any]] = None
    tool_call_latencies: Optional[Any] = None
    finish_reasons: Optional[List[Any]] = None
    stop_reason: Optional[str] = None

    mcp_servers: Optional[List[Any]] = None
    mcp_calls: Optional[List[Any]] = None
    mcp_results: Optional[List[Any]] = None

    trace_id: Optional[str] = None
    parent_id: Optional[str] = None
    # Multiple parents for a DAG join / fan-in node (e.g. a LangGraph node
    # triggered by several upstream nodes). When set, the evaluator uses these
    # as the step's parents instead of the single ``parent_id``.
    parent_ids: Optional[List[Any]] = None
    # ALL declared predecessor run_ids (>= 1), including single-dependency
    # fan-out edges that parent_ids (fan-in joins only) omits. Purely for the
    # dashboard to draw the full DAG (CrewAI/GoogleADK); the evaluator uses
    # parent_ids. (LangGraph stamps node-name predecessors under `langgraph`.)
    predecessors: Optional[List[Any]] = None
    function: Optional[str] = None
    output: Optional[Any] = None
    success: Optional[bool] = None
