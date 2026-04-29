from fluiq.integrations.Langchain.trace import patch_langchain


def patch_langgraph():
    # LangGraph executes nodes through LangChain Core's callback infrastructure.
    # The same FluiqCallbackHandler captures node/tool/llm events; the handler
    # detects LangGraph metadata (langgraph_node, langgraph_step, ...) and
    # tags those traces with TraceType.LangGraph. Registration is idempotent.
    patch_langchain()
