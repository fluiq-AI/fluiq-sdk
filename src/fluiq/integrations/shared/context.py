from contextvars import ContextVar

_in_langchain_llm: ContextVar = ContextVar("fluiq_in_langchain_llm", default=False)


def is_in_langchain_llm() -> bool:
    return bool(_in_langchain_llm.get())


def enter_langchain_llm():
    return _in_langchain_llm.set(True)


def exit_langchain_llm(token):
    if token is None:
        return
    try:
        _in_langchain_llm.reset(token)
    except (ValueError, LookupError):
        pass
