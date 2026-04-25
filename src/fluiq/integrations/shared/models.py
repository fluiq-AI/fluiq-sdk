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
    CrewAI: str = "CREWAI"
    GoogleADK: str = "GOOGLEADK"
    AgentToAgent: str = "AGENTTOAGENT"

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
    function: Optional[str] = None
    output: Optional[Any] = None
    success: Optional[bool] = None
