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
    return obj


def _crew_name(crew):
    """Return a human-readable name for a Crew instance."""
    try:
        agents = getattr(crew, "agents", None) or []
        roles = [getattr(a, "role", None) for a in agents if getattr(a, "role", None)]
        if roles:
            return f"Crew({', '.join(roles)})"
    except Exception:
        pass
    return "Crew"


def _agent_name(agent):
    """Return the agent's role as its name."""
    try:
        role = getattr(agent, "role", None)
        if role:
            return str(role)
    except Exception:
        pass
    return "Agent"


def _task_description(task):
    """Return a short description of the task."""
    try:
        desc = getattr(task, "description", None)
        if desc:
            return str(desc)[:200]
    except Exception:
        pass
    return "Task"


def _tool_name(tool):
    """Return the tool's name."""
    try:
        name = getattr(tool, "name", None)
        if name:
            return str(name)
    except Exception:
        pass
    return "Tool"


def _extract_crew_output(result):
    """Extract a serialisable representation of a CrewOutput."""
    try:
        raw = getattr(result, "raw", None)
        if raw:
            return str(raw)
    except Exception:
        pass
    return _to_jsonable(result)


def _extract_task_output(result):
    """Extract a serialisable representation of a TaskOutput."""
    try:
        raw = getattr(result, "raw", None)
        if raw:
            return str(raw)
    except Exception:
        pass
    return _to_jsonable(result)


def _extract_token_usage(result):
    """Extract token usage from a CrewOutput if available."""
    try:
        usage = getattr(result, "token_usage", None)
        if usage is None:
            return None
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        if not isinstance(usage, dict):
            return None
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion = usage.get("completion_tokens") or usage.get("output_tokens")
        total = usage.get("total_tokens")
        if any([prompt, completion, total]):
            return {"prompt": prompt, "completion": completion, "total": total}
    except Exception:
        pass
    return None
