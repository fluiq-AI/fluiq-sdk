_registered = False


def patch_google_adk():
    # Inject FluiqADKPlugin into every google.adk Runner by wrapping the
    # PluginManager constructor. ADK's PluginManager accepts a `plugins` list
    # (passed by Runner.__init__) and we simply prepend our plugin so it
    # captures agent/tool spans alongside any user-defined plugins. LLM calls
    # are not duplicated here — the existing Gemini patch already traces
    # google.genai.Client.models.generate_content, and our plugin sets the
    # parent_id ContextVar so those LLM traces nest under the active agent.
    global _registered
    if _registered:
        return

    from google.adk.plugins import plugin_manager as _pm
    from fluiq.integrations.GoogleADK.plugin import FluiqADKPlugin

    original_init = _pm.PluginManager.__init__

    def patched_init(self, *args, plugins=None, **kwargs):
        plugins = list(plugins or [])
        if not any(isinstance(p, FluiqADKPlugin) for p in plugins):
            plugins.insert(0, FluiqADKPlugin())
        return original_init(self, *args, plugins=plugins, **kwargs)

    _pm.PluginManager.__init__ = patched_init
    _registered = True
