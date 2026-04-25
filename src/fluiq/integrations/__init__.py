
def init():
    try:
        from fluiq.integrations.OpenAI.trace import patch_openai
        patch_openai()
    except ImportError:
        pass

    try:
        from fluiq.integrations.OpenAI.trace import patch_openai_responses
        patch_openai_responses()
    except ImportError:
        pass

    try:
        from fluiq.integrations.Anthropic.trace import patch_anthropic
        patch_anthropic()
    except ImportError:
        pass

    try:
        from fluiq.integrations.Anthropic.trace import patch_anthropic_beta
        patch_anthropic_beta()
    except ImportError:
        pass

    try:
        from fluiq.integrations.Gemini.trace import patch_genai
        patch_genai()
    except ImportError:
        pass

    try:
        from fluiq.integrations.Gemini.trace import patch_genai_async
        patch_genai_async()
    except ImportError:
        pass

    try:
        from fluiq.integrations.Gemini.trace import patch_vertexai
        patch_vertexai()
    except ImportError:
        pass