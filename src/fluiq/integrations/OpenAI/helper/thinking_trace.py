def _message_reasoning(message):
    if message is None:
        return None
    if isinstance(message, dict):
        return message.get("reasoning_content") or message.get("reasoning")
    return (
        getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
    )


def _reasoning_tokens(usage):
    if usage is None:
        return None
    details = (
        usage.get("completion_tokens_details")
        if isinstance(usage, dict)
        else getattr(usage, "completion_tokens_details", None)
    )
    if details is None:
        return None
    if isinstance(details, dict):
        return details.get("reasoning_tokens")
    return getattr(details, "reasoning_tokens", None)


def _extract_thinking(choices, usage=None):
    items = []
    for choice in choices or []:
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        reasoning = _message_reasoning(message)
        if reasoning:
            items.append({"reasoning": reasoning})

    tokens = _reasoning_tokens(usage)
    if tokens:
        items.append({"reasoning_tokens": tokens})

    return items or None
