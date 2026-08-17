"""Declare prompts, scorers, and datasets in code, then push them to Fluiq.

The dashboard is a fine place to *write* a prompt and a bad place to keep one.
A prompt is a piece of your application's behaviour: it belongs next to the code
it steers, in the same review, on the same branch, deployed by the same pipeline.
Editing it in a web form means the version that shipped and the version in the
repo are two different things, and nobody finds out until they disagree.

So declare them in a file::

    # fluiq_resources.py
    from fluiq.resources import Prompt, Scorer, Dataset

    Prompt(
        slug="support-reply",
        name="Support reply",
        template="You are a support agent. Reply to: {{input}}",
        model="gpt-5-mini",
    )

    Scorer(
        slug="mentions-refund",
        name="Mentions refund",
        kind="code",
        body="contains(output, 'refund')",
    )

    Dataset(
        name="checkout-agent",
        examples=[{"input": "where is my order?", "expected": "..."}],
    )

and push them::

    python -m fluiq.cli push fluiq_resources.py

Declaring is not pushing. Constructing one of these registers it in-process and
does nothing else, so importing the file is free and safe — a module that made
network calls on import would be unusable in tests and would fire on every
`python -c` that happened to touch it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Everything declared since the process started, in declaration order.
#: Order matters on push: a dataset that references a scorer must not be sent
#: before it exists.
_REGISTRY: Dict[str, List[Any]] = {"prompts": [], "scorers": [], "datasets": []}

VALID_SCORER_KINDS = ("judge", "code")


def _register(bucket: str, item: Any) -> None:
    existing = _REGISTRY[bucket]
    key = getattr(item, "slug", None) or getattr(item, "name", None)
    for index, other in enumerate(existing):
        other_key = getattr(other, "slug", None) or getattr(other, "name", None)
        if other_key == key:
            # A redeclaration replaces rather than duplicates: re-importing a
            # module (pytest collection, a REPL reload) must not push the same
            # prompt twice under two slightly different definitions.
            existing[index] = item
            return
    existing.append(item)


@dataclass
class Prompt:
    """A prompt template, versioned with your code.

    ``kind='completion'`` is an ordinary prompt; ``'judge'`` is an LLM-as-judge
    template referenced by slug from ``fluiq.eval(custom_judges=...)``.
    """
    slug: str
    name: str
    template: str
    model: Optional[str] = None
    kind: str = "completion"
    variables: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.slug = (self.slug or "").strip().lower()
        if not self.slug:
            raise ValueError("Prompt needs a slug")
        if not (self.template or "").strip():
            raise ValueError(f"Prompt {self.slug!r} has an empty template")
        _register("prompts", self)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "slug": self.slug,
            "name": self.name or self.slug,
            "template": self.template,
            "model": self.model,
            "kind": self.kind,
            "variables": self.variables,
        }


@dataclass
class Scorer:
    """A judge prompt or a deterministic code scorer.

    ``choices`` turns a judge into a forced-choice one: it picks a label and the
    score comes from your table rather than from the model's arithmetic.
    """
    slug: str
    name: str
    body: str
    kind: str = "judge"
    threshold: float = 0.5
    choices: Optional[List[Dict[str, Any]]] = None
    #: Datasets that should remember this scorer for their runs.
    datasets: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.slug = (self.slug or "").strip().lower()
        if not self.slug:
            raise ValueError("Scorer needs a slug")
        if self.kind not in VALID_SCORER_KINDS:
            raise ValueError(
                f"Scorer {self.slug!r}: kind must be one of {VALID_SCORER_KINDS}"
            )
        if not (self.body or "").strip():
            raise ValueError(f"Scorer {self.slug!r} has an empty body")
        if self.kind == "code" and self.choices:
            raise ValueError(
                f"Scorer {self.slug!r}: choices apply to judges. A code scorer "
                f"returns its own number, so there is nothing to map."
            )
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError(f"Scorer {self.slug!r}: threshold must be between 0 and 1")
        _register("scorers", self)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "slug": self.slug,
            "name": self.name or self.slug,
            "template": self.body,
            "kind": self.kind,
            "threshold": self.threshold,
            "choices": self.choices,
            "datasets": self.datasets,
        }


@dataclass
class Dataset:
    """Test cases, versioned with your code.

    Examples pushed here are *added*, never used to delete rows the platform
    already has. A dataset accumulates real production failures over time, and a
    push that pruned it to whatever the file happened to list would throw those
    away on the first deploy after someone trimmed the fixture.
    """
    name: str
    examples: List[Dict[str, Any]] = field(default_factory=list)
    description: Optional[str] = None
    kind: str = "text"

    def __post_init__(self) -> None:
        self.name = (self.name or "").strip()
        if not self.name:
            raise ValueError("Dataset needs a name")
        _register("datasets", self)

    def to_payload(self) -> Dict[str, Any]:
        rows = []
        for example in self.examples:
            if not isinstance(example, dict):
                raise ValueError(
                    f"Dataset {self.name!r}: each example must be a dict with "
                    f"'input' and optionally 'expected'"
                )
            rows.append({
                "input": str(example.get("input") or ""),
                "expected_output": example.get("expected") or example.get("expected_output"),
                "metadata": example.get("metadata") or {},
            })
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "examples": rows,
        }


def declared() -> Dict[str, List[Any]]:
    """Everything declared so far, for the push command."""
    return {bucket: list(items) for bucket, items in _REGISTRY.items()}


def clear() -> None:
    """Empty the registry. For tests; the CLI runs one file per process."""
    for items in _REGISTRY.values():
        items.clear()


def payload() -> Dict[str, Any]:
    """The wire shape ``fluiq push`` sends."""
    return {
        "prompts":  [p.to_payload() for p in _REGISTRY["prompts"]],
        "scorers":  [s.to_payload() for s in _REGISTRY["scorers"]],
        "datasets": [d.to_payload() for d in _REGISTRY["datasets"]],
    }


__all__ = [
    "Dataset",
    "Prompt",
    "Scorer",
    "VALID_SCORER_KINDS",
    "clear",
    "declared",
    "payload",
]
