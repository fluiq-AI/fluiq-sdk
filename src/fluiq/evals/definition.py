"""Evals defined in code, where the task is *your function*.

This is the difference that matters. A task defined in the dashboard is a copy
of your application's behaviour, and a copy goes stale: your app grows a step,
the eval doesn't, and it keeps passing while measuring something you no longer
ship. A task that is a Python callable calls the real thing, so it cannot drift
— when the app changes, the next run tests the change.

    # support_eval.py
    from fluiq import Eval
    from myapp import answer_customer

    Eval(
        name="support quality",
        task=answer_customer,          # the live code, not a copy of it
        dataset="checkout-agent",      # or data=[...]
        scores=["relevance", "completeness", "mentions-refund"],
    )

Then::

    python -m fluiq.cli eval support_eval.py --fail-below 0.7

The task runs **locally**, in your process, with your dependencies and your
secrets. Only the inputs and outputs go to Fluiq, to be scored and recorded — so
this works against a service that is not reachable from the internet, which is
most of them during development.

Declaring registers; it does not run. Importing an eval file must be free.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

#: Every Eval declared since the process started, in declaration order.
_REGISTRY: List["Eval"] = []

#: A scorer is a built-in metric name, a saved scorer's slug, or a local
#: function ``(output, expected, input) -> float | bool``.
Scorer = Union[str, Callable[..., Any]]


@dataclass
class Eval:
    """One named evaluation: a task, some data, and how to score it."""

    name: str
    task: Callable[..., Any]
    #: Inline rows, each ``{"input": ..., "expected": ...}``. Mutually exclusive
    #: with ``dataset``.
    data: Optional[Sequence[Dict[str, Any]]] = None
    #: A dataset by name, pulled from the platform at run time. Preferred once
    #: you have real production failures in one — inline data is for getting
    #: started, not for staying there.
    dataset: Optional[str] = None
    scores: Sequence[Scorer] = field(default_factory=list)
    #: "provider:model" for the judge, else the server default.
    judge: Optional[str] = None
    #: Repeat every example N times and average, to blunt judge variance.
    trials: int = 1
    #: Concurrent task invocations. Your code, your rate limits — hence low.
    concurrency: int = 4
    description: Optional[str] = None

    def __post_init__(self) -> None:
        self.name = (self.name or "").strip()
        if not self.name:
            raise ValueError("Eval needs a name")
        if not callable(self.task):
            raise ValueError(
                f"Eval {self.name!r}: task must be callable. Pass the function "
                f"itself, not a call to it."
            )
        if self.data is None and not self.dataset:
            raise ValueError(
                f"Eval {self.name!r}: needs either data=[...] or dataset='name'"
            )
        if self.data is not None and self.dataset:
            raise ValueError(
                f"Eval {self.name!r}: pass data or dataset, not both — which one "
                f"won would be invisible in the results."
            )
        if not self.scores:
            raise ValueError(
                f"Eval {self.name!r}: needs at least one scorer, or it measures nothing"
            )
        if self.trials < 1:
            raise ValueError(f"Eval {self.name!r}: trials must be at least 1")
        _REGISTRY.append(self)

    # ── Scorer classification ──
    # Local callables run here; names are resolved server-side. Splitting them
    # up front means one round trip carries every remote scorer rather than one
    # per scorer per example.

    @property
    def local_scorers(self) -> List[Callable[..., Any]]:
        return [s for s in self.scores if callable(s)]

    @property
    def remote_scorers(self) -> List[str]:
        return [str(s) for s in self.scores if not callable(s)]

    def run_task(self, example: Dict[str, Any]) -> str:
        """Invoke the task for one example.

        Accepts the three shapes people actually write — ``task(input)``,
        ``task(**example)``, and ``task(example)`` — because insisting on one
        would mean rewriting the function you are trying to test, which defeats
        the point of pointing at the live one.
        """
        value = example.get("input")
        try:
            signature = inspect.signature(self.task)
        except (TypeError, ValueError):
            # Builtins and C callables have no introspectable signature.
            return _stringify(self.task(value))

        params = [
            p for p in signature.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        names = {p.name for p in params}

        if "expected" in names or "metadata" in names:
            kwargs = {"input": value}
            if "expected" in names:
                kwargs["expected"] = example.get("expected")
            if "metadata" in names:
                kwargs["metadata"] = example.get("metadata") or {}
            return _stringify(self.task(**kwargs))

        if len(params) == 1 and params[0].name in ("example", "row", "case"):
            return _stringify(self.task(example))

        return _stringify(self.task(value))

    def score_locally(
        self, *, output: str, expected: str, input: str,
    ) -> Dict[str, float]:
        """Run the callable scorers. Never raises: one broken scorer must not
        cost the run every other score on the example."""
        results: Dict[str, float] = {}
        for scorer in self.local_scorers:
            label = getattr(scorer, "__name__", None) or "scorer"
            try:
                value = _call_scorer(scorer, output=output, expected=expected, input=input)
            except Exception as exc:  # noqa: BLE001
                print(f"[fluiq.eval] scorer {label!r} failed: {exc!r}")
                continue
            if isinstance(value, bool):
                results[label] = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                results[label] = max(0.0, min(1.0, float(value)))
            else:
                print(
                    f"[fluiq.eval] scorer {label!r} returned {type(value).__name__}; "
                    f"expected a number 0-1 or a bool"
                )
        return results


def _call_scorer(scorer: Callable[..., Any], **available: Any) -> Any:
    """Call a scorer with whatever subset of (output, expected, input) it takes."""
    try:
        signature = inspect.signature(scorer)
    except (TypeError, ValueError):
        return scorer(available["output"])
    names = set(signature.parameters)
    if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        return scorer(**available)
    kwargs = {k: v for k, v in available.items() if k in names}
    return scorer(**kwargs) if kwargs else scorer(available["output"])


def _stringify(value: Any) -> str:
    """Task outputs land in a text field, so normalise the shapes people return."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # The common wrapper shapes, so returning a response object works.
        for key in ("output", "content", "text", "response", "answer"):
            inner = value.get(key)
            if isinstance(inner, str):
                return inner
    return str(value)


def declared() -> List[Eval]:
    return list(_REGISTRY)


def clear() -> None:
    _REGISTRY.clear()


__all__ = ["Eval", "Scorer", "clear", "declared"]
