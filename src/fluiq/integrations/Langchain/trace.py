from contextvars import ContextVar

from fluiq.integrations.Langchain.handler import FluiqCallbackHandler


_fluiq_lc_var: ContextVar = ContextVar("fluiq_langchain_handler", default=None)
_registered = False


def patch_langchain():
    global _registered
    if _registered:
        return

    from langchain_core.tracers.context import register_configure_hook

    register_configure_hook(_fluiq_lc_var, True, FluiqCallbackHandler)
    _fluiq_lc_var.set(FluiqCallbackHandler())
    _registered = True
