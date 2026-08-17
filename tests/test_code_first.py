"""Code-declared resources and code-defined evals.

The load-bearing property here is that declaring is *free*: importing a file
that declares a prompt or an eval must make no network call and run no task.
Anything else makes these files unusable in tests and hazardous to import.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from fluiq import resources  # noqa: E402
from fluiq.evals import definition  # noqa: E402
from fluiq.evals.definition import Eval, _stringify  # noqa: E402
from fluiq.resources import Dataset, Prompt, Scorer  # noqa: E402


@pytest.fixture(autouse=True)
def clean():
    resources.clear()
    definition.clear()
    yield
    resources.clear()
    definition.clear()


# ══ Resources ════════════════════════════════════════════════════════════════

def test_declaring_registers_without_pushing():
    """A module that made network calls on import would be unusable in tests and
    would fire on any `python -c` that touched it."""
    Prompt(slug="a", name="A", template="hi {{input}}")
    assert [p.slug for p in resources.declared()["prompts"]] == ["a"]


def test_redeclaring_replaces_rather_than_duplicates():
    """Re-importing a module — pytest collection, a REPL reload — must not push
    the same prompt twice under two slightly different definitions."""
    Prompt(slug="a", name="First", template="one {{input}}")
    Prompt(slug="a", name="Second", template="two {{input}}")
    prompts = resources.declared()["prompts"]
    assert len(prompts) == 1
    assert prompts[0].name == "Second"


def test_slugs_are_lowercased():
    assert Prompt(slug="  MyPrompt ", name="X", template="t").slug == "myprompt"


def test_an_empty_template_is_rejected_at_declaration():
    with pytest.raises(ValueError, match="empty template"):
        Prompt(slug="a", name="A", template="   ")


def test_a_prompt_needs_a_slug():
    with pytest.raises(ValueError, match="needs a slug"):
        Prompt(slug="", name="A", template="t")


def test_a_code_scorer_cannot_carry_choices():
    """It returns its own number; there is nothing for a choice table to map."""
    with pytest.raises(ValueError, match="choices apply to judges"):
        Scorer(
            slug="s", name="S", body="len(output) < 5", kind="code",
            choices=[{"label": "Y", "score": 1}, {"label": "N", "score": 0}],
        )


def test_an_unknown_scorer_kind_is_rejected():
    with pytest.raises(ValueError, match="kind must be one of"):
        Scorer(slug="s", name="S", body="x", kind="wasm")


def test_a_threshold_outside_the_unit_range_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        Scorer(slug="s", name="S", body="Grade {{answer}}", threshold=5)


def test_dataset_examples_accept_either_expected_spelling():
    Dataset(name="d", examples=[
        {"input": "a", "expected": "x"},
        {"input": "b", "expected_output": "y"},
    ])
    rows = resources.payload()["datasets"][0]["examples"]
    assert [r["expected_output"] for r in rows] == ["x", "y"]


def test_a_non_dict_example_is_rejected_with_a_useful_message():
    Dataset(name="d", examples=["just a string"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="must be a dict"):
        resources.payload()


def test_the_push_payload_carries_all_three_buckets():
    Prompt(slug="p", name="P", template="{{input}}")
    Scorer(slug="s", name="S", body="Grade {{answer}}")
    Dataset(name="d", examples=[{"input": "x"}])
    payload = resources.payload()
    assert len(payload["prompts"]) == 1
    assert len(payload["scorers"]) == 1
    assert len(payload["datasets"]) == 1


# ══ Eval definition ══════════════════════════════════════════════════════════

def task(input):  # noqa: A002
    return f"answered: {input}"


def test_declaring_an_eval_runs_nothing():
    calls = []

    def spy(input):  # noqa: A002
        calls.append(input)
        return "x"

    Eval(name="e", task=spy, data=[{"input": "a"}], scores=["relevance"])
    assert calls == [], "declaring an eval invoked the task"


def test_an_eval_needs_data_or_a_dataset():
    with pytest.raises(ValueError, match="data=|dataset="):
        Eval(name="e", task=task, scores=["relevance"])


def test_data_and_dataset_together_are_rejected():
    """Which one won would be invisible in the results."""
    with pytest.raises(ValueError, match="not both"):
        Eval(name="e", task=task, data=[{"input": "a"}], dataset="d", scores=["x"])


def test_an_eval_needs_at_least_one_scorer():
    with pytest.raises(ValueError, match="at least one scorer"):
        Eval(name="e", task=task, data=[{"input": "a"}], scores=[])


def test_passing_a_call_instead_of_a_function_is_caught():
    """`task=my_func()` is the mistake everyone makes once."""
    with pytest.raises(ValueError, match="must be callable"):
        Eval(name="e", task="already a string", data=[{"input": "a"}], scores=["x"])  # type: ignore[arg-type]


def test_scorers_split_into_local_and_remote():
    """Names resolve server-side; callables run here. Splitting up front means
    one round trip carries every remote scorer."""
    def my_check(output):
        return True

    spec = Eval(
        name="e", task=task, data=[{"input": "a"}],
        scores=["relevance", "my-saved-scorer", my_check],
    )
    assert spec.remote_scorers == ["relevance", "my-saved-scorer"]
    assert spec.local_scorers == [my_check]


# ══ Calling the task ═════════════════════════════════════════════════════════
#
# Three shapes people actually write. Insisting on one would mean rewriting the
# function you are trying to test, which defeats pointing at the live one.

def test_a_single_argument_task_gets_the_input():
    spec = Eval(name="e", task=lambda x: f"got {x}", data=[{"input": "a"}], scores=["s"])
    assert spec.run_task({"input": "a"}) == "got a"


def test_a_task_that_wants_expected_receives_it():
    def with_expected(input, expected):  # noqa: A002
        return f"{input}|{expected}"

    spec = Eval(name="e", task=with_expected, data=[{"input": "a"}], scores=["s"])
    assert spec.run_task({"input": "a", "expected": "b"}) == "a|b"


def test_a_task_that_wants_metadata_receives_it():
    def with_metadata(input, metadata):  # noqa: A002
        return str(metadata.get("tier"))

    spec = Eval(name="e", task=with_metadata, data=[{"input": "a"}], scores=["s"])
    assert spec.run_task({"input": "a", "metadata": {"tier": "gold"}}) == "gold"


def test_a_task_taking_the_whole_row_gets_it():
    def whole(example):
        return example["input"] + str(example.get("expected"))

    spec = Eval(name="e", task=whole, data=[{"input": "a"}], scores=["s"])
    assert spec.run_task({"input": "a", "expected": "b"}) == "ab"


def test_a_missing_metadata_key_defaults_to_empty():
    def with_metadata(input, metadata):  # noqa: A002
        return str(len(metadata))

    spec = Eval(name="e", task=with_metadata, data=[{"input": "a"}], scores=["s"])
    assert spec.run_task({"input": "a"}) == "0"


# ══ Normalising what a task returns ══════════════════════════════════════════

def test_a_string_passes_through():
    assert _stringify("hello") == "hello"


def test_none_becomes_empty_rather_than_the_word_none():
    """"None" scored as a response would look like a real, terrible answer."""
    assert _stringify(None) == ""


@pytest.mark.parametrize("key", ["output", "content", "text", "response", "answer"])
def test_common_wrapper_shapes_are_unwrapped(key):
    """Returning a response object should just work."""
    assert _stringify({key: "the answer"}) == "the answer"


def test_an_unrecognised_shape_falls_back_to_str():
    assert _stringify(42) == "42"


# ══ Local scoring ════════════════════════════════════════════════════════════

def test_a_boolean_scorer_becomes_one_or_zero():
    spec = Eval(
        name="e", task=task, data=[{"input": "a"}],
        scores=[lambda output: "yes" in output],
    )
    assert spec.score_locally(output="yes", expected="", input="")["<lambda>"] == 1.0
    assert spec.score_locally(output="no", expected="", input="")["<lambda>"] == 0.0


def test_a_numeric_scorer_is_clamped_to_the_unit_range():
    def over(output):
        return 5.0

    spec = Eval(name="e", task=task, data=[{"input": "a"}], scores=[over])
    assert spec.score_locally(output="x", expected="", input="")["over"] == 1.0


def test_a_scorer_receives_only_the_arguments_it_declares():
    def compares(output, expected):
        return output == expected

    spec = Eval(name="e", task=task, data=[{"input": "a"}], scores=[compares])
    assert spec.score_locally(output="same", expected="same", input="ignored")["compares"] == 1.0


def test_a_broken_scorer_does_not_cost_the_others():
    """One bad scorer must not lose every other score on the example."""
    def broken(output):
        raise RuntimeError("boom")

    def fine(output):
        return 1.0

    spec = Eval(name="e", task=task, data=[{"input": "a"}], scores=[broken, fine])
    scores = spec.score_locally(output="x", expected="", input="")
    assert "broken" not in scores
    assert scores["fine"] == 1.0


def test_a_scorer_returning_the_wrong_type_is_skipped_not_coerced():
    def stringy(output):
        return "great"

    spec = Eval(name="e", task=task, data=[{"input": "a"}], scores=[stringy])
    assert spec.score_locally(output="x", expected="", input="") == {}


def test_trials_must_be_at_least_one():
    with pytest.raises(ValueError, match="at least 1"):
        Eval(name="e", task=task, data=[{"input": "a"}], scores=["s"], trials=0)
