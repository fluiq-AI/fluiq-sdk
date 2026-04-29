def _to_jsonable(obj):
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, bytes):
        import base64
        return base64.b64encode(obj).decode("ascii")
    return obj


def _content_to_text(content):
    # google.genai.types.Content has a `parts` list of Part objects with `text`.
    if content is None:
        return None
    parts = getattr(content, "parts", None)
    if not parts:
        return None
    chunks = []
    for p in parts:
        text = getattr(p, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks) if chunks else None


def _invocation_id(callback_context):
    inv = getattr(callback_context, "invocation_id", None)
    if inv:
        return str(inv)
    inner = getattr(callback_context, "_invocation_context", None)
    if inner is not None:
        inv = getattr(inner, "invocation_id", None)
        if inv:
            return str(inv)
    return None


def _user_message(callback_context):
    inner = getattr(callback_context, "_invocation_context", None)
    if inner is None:
        return None
    msg = getattr(inner, "user_content", None) or getattr(inner, "new_message", None)
    return _to_jsonable(msg)


def _agent_model(agent):
    model = getattr(agent, "model", None)
    if model is None:
        return None
    if isinstance(model, str):
        return model
    return getattr(model, "model", None) or getattr(model, "name", None)


def _collect_output_keys(agent):
    # LlmAgent has output_key; wrapper agents (LoopAgent/SequentialAgent/
    # ParallelAgent) don't, but their sub_agents do. Walk the sub_agent tree
    # so a wrapper's after_agent trace can surface the keys its descendants
    # wrote to session state.
    keys = []
    seen = set()
    stack = [agent]
    while stack:
        a = stack.pop()
        ok = getattr(a, "output_key", None)
        if isinstance(ok, str) and ok and ok not in seen:
            seen.add(ok)
            keys.append(ok)
        for sub in getattr(a, "sub_agents", None) or []:
            stack.append(sub)
    return keys


def _state_snapshot(callback_context, keys):
    state = getattr(callback_context, "state", None)
    if state is None or not keys:
        return {}
    snap = {}
    for k in keys:
        try:
            if k in state:
                snap[k] = state[k]
        except Exception:
            pass
    return snap


def _agent_output(callback_context, output_keys, before_snapshot):
    state = getattr(callback_context, "state", None)
    if state is None or not output_keys:
        return None
    changed = {}
    for k in output_keys:
        try:
            if k not in state:
                continue
            v = state[k]
        except Exception:
            continue
        if k in before_snapshot and before_snapshot[k] == v:
            continue
        changed[k] = v
    if not changed:
        for k in output_keys:
            try:
                if k in state:
                    changed[k] = state[k]
            except Exception:
                pass
    if not changed:
        return None
    if len(changed) == 1:
        return _to_jsonable(next(iter(changed.values())))
    return _to_jsonable(changed)


def _tool_input_schema(tool):
    for attr in ("get_declaration", "_get_declaration"):
        fn = getattr(tool, attr, None)
        if not callable(fn):
            continue
        try:
            decl = fn()
        except Exception:
            continue
        if decl is None:
            continue
        params = getattr(decl, "parameters", None)
        return _to_jsonable(params)
    return None
