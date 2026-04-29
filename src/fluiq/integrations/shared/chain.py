import hashlib
import json
import uuid
from contextvars import ContextVar
from typing import Any, Optional, Tuple


_CONVERSATION_KEYS = ("contents", "messages", "input")
_chain_state: ContextVar = ContextVar("fluiq_chain_state", default=None)


def _extract_turns(data: dict) -> Optional[list]:
    for key in _CONVERSATION_KEYS:
        value = data.get(key)
        if isinstance(value, list) and len(value) > 0:
            # LangChain wraps `messages` as a list of generations
            # ([[msg, msg, ...], ...]). Flatten one level so prefix-extension
            # comparison works across calls in an agent loop.
            if all(isinstance(item, list) for item in value):
                flat: list = []
                for inner in value:
                    flat.extend(inner)
                return flat if flat else None
            return value
    return None


def _hash_turn(turn: Any) -> str:
    try:
        canonical = json.dumps(
            turn, sort_keys=True, separators=(",", ":"), default=str
        )
    except (TypeError, ValueError):
        canonical = repr(turn)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def compute_chain_id(data: dict) -> Optional[str]:
    turns = _extract_turns(data)
    state = _chain_state.get()
    if turns is None:
        # Tool / function / non-LLM trace. Inherit the currently active chain
        # (if any) so it visually groups with the surrounding LLM calls.
        # Don't mutate state — preserves the original prefix for the next
        # LLM call's prefix-extension comparison.
        if state is not None:
            return state[0]
        return None
    hashes: Tuple[str, ...] = tuple(_hash_turn(t) for t in turns)
    if state is not None:
        prev_id, prev_hashes = state
        if (
            len(hashes) >= len(prev_hashes)
            and hashes[: len(prev_hashes)] == prev_hashes
        ):
            _chain_state.set((prev_id, hashes))
            return prev_id
    chain_id = str(uuid.uuid4())
    _chain_state.set((chain_id, hashes))
    return chain_id


def reset_chain_state() -> None:
    _chain_state.set(None)
