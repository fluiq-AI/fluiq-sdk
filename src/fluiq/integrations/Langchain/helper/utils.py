def _to_jsonable(obj):
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, bytes):
        import base64
        return base64.b64encode(obj).decode("ascii")
    return obj


def _model_name(serialized, invocation_params=None, metadata=None):
    sources = (
        invocation_params,
        (serialized or {}).get("kwargs"),
        serialized,
        metadata,
    )
    keys = (
        "model",
        "model_name",
        "deployment_name",
        "azure_deployment",
        "ls_model_name",
    )
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            val = source.get(key)
            if val:
                return val
    return None


def _component_name(serialized):
    if not serialized:
        return None
    name = serialized.get("name")
    if name:
        return name
    ids = serialized.get("id")
    if isinstance(ids, list) and ids:
        return ids[-1]
    return None


def _extract_tokens(response):
    sources = []
    llm_output = getattr(response, "llm_output", None) if response is not None else None
    if isinstance(llm_output, dict):
        sources.extend([
            llm_output.get("token_usage"),
            llm_output.get("usage"),
            llm_output.get("usage_metadata"),
        ])
    try:
        msg = response.generations[0][0].message
        sources.append(getattr(msg, "usage_metadata", None))
        rmd = getattr(msg, "response_metadata", None)
        if isinstance(rmd, dict):
            sources.extend([
                rmd.get("token_usage"),
                rmd.get("usage"),
                rmd.get("usage_metadata"),
            ])
    except Exception:
        pass
    for usage in sources:
        if not isinstance(usage, dict):
            continue
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion = usage.get("completion_tokens") or usage.get("output_tokens")
        total = usage.get("total_tokens")
        if prompt or completion or total:
            return {"prompt": prompt, "completion": completion, "total": total}
    return None


def _extract_response_text(response):
    try:
        gens = response.generations
        if gens and gens[0]:
            return gens[0][0].text
    except Exception:
        pass
    return None


def _extract_response_model(response):
    try:
        msg = response.generations[0][0].message
        rmd = getattr(msg, "response_metadata", None)
        if isinstance(rmd, dict):
            return rmd.get("model_name") or rmd.get("model")
    except Exception:
        pass
    return None


def _extract_finish_reason(response):
    try:
        for gen_list in (response.generations or []):
            for gen in gen_list:
                info = getattr(gen, "generation_info", None) or {}
                reason = info.get("finish_reason") or info.get("done_reason")
                if reason:
                    return reason
                msg = getattr(gen, "message", None)
                rmd = getattr(msg, "response_metadata", None) if msg else None
                if isinstance(rmd, dict):
                    reason = rmd.get("finish_reason") or rmd.get("done_reason")
                    if reason:
                        return reason
    except Exception:
        pass
    return None


def _extract_tool_calls(response):
    calls = []
    try:
        for gen_list in (response.generations or []):
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                tcs = getattr(msg, "tool_calls", None) or []
                for tc in tcs:
                    calls.append(_to_jsonable(tc))
    except Exception:
        pass
    return calls or None
