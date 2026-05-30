
def _safe(import_path, attr):
    try:
        mod = __import__(import_path, fromlist=[attr])
        getattr(mod, attr)()
    except Exception:
        pass


def init():
    _safe("fluiq.integrations.OpenAI.trace", "patch_openai")
    _safe("fluiq.integrations.OpenAI.trace", "patch_openai_async")
    _safe("fluiq.integrations.OpenAI.trace", "patch_openai_responses")
    _safe("fluiq.integrations.OpenAI.trace", "patch_openai_responses_async")
    _safe("fluiq.integrations.OpenAI.trace", "patch_openai_parse")
    _safe("fluiq.integrations.OpenAI.trace", "patch_openai_parse_async")
    _safe("fluiq.integrations.OpenAI.trace", "patch_openai_stream_helper")
    _safe("fluiq.integrations.OpenAI.trace", "patch_openai_stream_helper_async")
    _safe("fluiq.integrations.OpenAI.endpoints", "patch_openai_embeddings")
    _safe("fluiq.integrations.OpenAI.endpoints", "patch_openai_embeddings_async")
    _safe("fluiq.integrations.OpenAI.endpoints", "patch_openai_images")
    _safe("fluiq.integrations.OpenAI.endpoints", "patch_openai_images_async")
    _safe("fluiq.integrations.OpenAI.endpoints", "patch_openai_audio")
    _safe("fluiq.integrations.OpenAI.endpoints", "patch_openai_audio_async")

    _safe("fluiq.integrations.Anthropic.trace", "patch_anthropic")
    _safe("fluiq.integrations.Anthropic.trace", "patch_anthropic_async")
    _safe("fluiq.integrations.Anthropic.trace", "patch_anthropic_beta")
    _safe("fluiq.integrations.Anthropic.trace", "patch_anthropic_beta_async")

    _safe("fluiq.integrations.Gemini.trace", "patch_genai")
    _safe("fluiq.integrations.Gemini.trace", "patch_genai_async")
    _safe("fluiq.integrations.Gemini.trace", "patch_genai_stream")
    _safe("fluiq.integrations.Gemini.trace", "patch_genai_stream_async")
    _safe("fluiq.integrations.Gemini.trace", "patch_genai_count_tokens")
    _safe("fluiq.integrations.Gemini.trace", "patch_genai_count_tokens_async")
    _safe("fluiq.integrations.Gemini.trace", "patch_vertexai")
    _safe("fluiq.integrations.Gemini.trace", "patch_vertexai_async")
    _safe("fluiq.integrations.Gemini.trace", "patch_vertexai_count_tokens")
    _safe("fluiq.integrations.Gemini.trace", "patch_vertexai_count_tokens_async")
    _safe("fluiq.integrations.Gemini.trace", "patch_genai_embeddings")
    _safe("fluiq.integrations.Gemini.trace", "patch_genai_embeddings_async")

    _safe("fluiq.integrations.Langchain.trace", "patch_langchain")

    _safe("fluiq.integrations.LangGraph.trace", "patch_langgraph")

    _safe("fluiq.integrations.CrewAI.trace", "patch_crewai")

    _safe("fluiq.integrations.GoogleADK.trace", "patch_google_adk")

    _safe("fluiq.integrations.Vectorstores.Chromadb.trace", "patch_chromadb")
    _safe("fluiq.integrations.Vectorstores.Chromadb.trace", "patch_chromadb_async")
    _safe("fluiq.integrations.Vectorstores.Pinecone.trace", "patch_pinecone")
    _safe("fluiq.integrations.Vectorstores.Pinecone.trace", "patch_pinecone_async")
    _safe("fluiq.integrations.Vectorstores.Qdrant.trace", "patch_qdrant")
    _safe("fluiq.integrations.Vectorstores.Qdrant.trace", "patch_qdrant_async")
    _safe("fluiq.integrations.Vectorstores.Weaviate.trace", "patch_weaviate")
    _safe("fluiq.integrations.Vectorstores.Weaviate.trace", "patch_weaviate_async")
    _safe("fluiq.integrations.Vectorstores.FAISS.trace", "patch_faiss")

    _safe("fluiq.integrations.Voyage.trace", "patch_voyage_embeddings")
    _safe("fluiq.integrations.Voyage.trace", "patch_voyage_embeddings_async")

    _safe("fluiq.integrations.shared.mcp_patch", "patch_mcp_initialize")
    _safe("fluiq.integrations.shared.mcp_patch", "patch_mcp_list_tools")
    _safe("fluiq.integrations.shared.mcp_patch", "patch_mcp_call_tool")