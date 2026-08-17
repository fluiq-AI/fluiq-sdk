"""The PR comment (TODO-29).

The body is built by a pure function precisely so these can be asserted without
a network, and the properties worth asserting are about honesty: a missing
baseline must not render as "no change", and a customer prompt must not be able
to break the table it is displayed in.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fluiq.pr_comment import (
    MARKER, _delta, _escape_cell, build_comment, pr_number_from_env,
)


RUN = {"task": {"prompt_name": "support", "prompt_version": 3, "model": "gpt-4o"}}


def comment(**over):
    kwargs = {
        "run": RUN,
        "summary": {"avg_score": 0.82, "metrics": {"quality": 0.9, "safety": 0.74}},
        "items": [],
        "baseline_summary": None,
        "gate": 0.8,
        "passed": True,
        "dashboard_url": None,
    }
    kwargs.update(over)
    return build_comment(**kwargs)


def test_the_marker_leads_so_the_next_run_can_find_this_comment():
    """Without it every push appends a new table and the PR fills with stale
    verdicts, which is how a useful signal becomes something people mute."""
    assert comment().startswith(MARKER)


def test_pass_and_fail_are_distinguishable_at_a_glance():
    assert "✅" in comment(passed=True)
    assert "❌" in comment(passed=False)


def test_a_missing_baseline_shows_a_dash_not_a_zero():
    """"±0.0" against no baseline would tell a reviewer this change is safe on
    the strength of no evidence at all."""
    assert _delta(0.9, None) == "—"
    assert _delta(0.9, 0.9) == "±0.0"
    assert _delta(0.95, 0.90) == "+5.0"
    assert _delta(0.85, 0.90) == "-5.0"

    # With no baseline the comparison column is dropped entirely rather than
    # filled with dashes — a column that can never say anything is noise.
    body = comment()
    assert "vs base" not in body
    assert "| quality | 90.0% |" in body


def test_a_regression_is_marked_as_one():
    body = comment(baseline_summary={"avg_score": 0.9,
                                     "metrics": {"quality": 0.95, "safety": 0.74}})
    assert "🔻" in body, "quality dropped 5 points and must be flagged"
    assert "-5.0" in body


def test_every_metric_gets_a_row():
    body = comment()
    assert "| quality |" in body
    assert "| safety |" in body


def test_a_pipe_in_a_customer_prompt_cannot_break_the_table():
    """The inputs most likely to be interesting are the ones most likely to
    contain markdown, and a split row would corrupt the report silently."""
    assert _escape_cell("a | b") == "a \\| b"
    assert "\n" not in _escape_cell("line one\nline two")

    body = comment(items=[{"input": "sum | of | cols", "result": [{"score": 0.1}]}])
    assert "sum \\| of \\| cols" in body


def test_a_very_long_input_is_truncated():
    cell = _escape_cell("x" * 500)
    assert len(cell) <= 121 and cell.endswith("…")


def test_the_weakest_examples_lead():
    body = comment(items=[
        {"input": "good one", "result": [{"score": 0.99}]},
        {"input": "the bad one", "result": [{"score": 0.05}]},
    ])
    assert body.index("the bad one") < body.index("good one")


def test_ungraded_examples_are_disclosed_not_silently_excluded():
    """Six of thirty examples failing to generate changes what the average
    means, and a reviewer reading 82% deserves to know it covers 24 rows."""
    body = comment(summary={"avg_score": 0.82, "metrics": {}, "gen_failed": 6})
    assert "6 example(s) produced no output" in body


def test_no_warning_when_everything_generated():
    assert "produced no output" not in comment(
        summary={"avg_score": 0.82, "metrics": {}, "gen_failed": 0})


def test_the_task_under_test_is_named():
    """Which prompt version produced this is the first thing a reviewer asks."""
    assert "support v3" in comment()
    assert "gpt-4o" in comment()


def test_pr_number_is_read_from_the_actions_ref(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_REF", "refs/pull/482/merge")
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    assert pr_number_from_env() == 482


def test_pr_number_falls_back_to_the_event_payload(monkeypatch, tmp_path):
    """A workflow_run-triggered job has a branch ref, not a pull ref, but still
    names the PR in its payload."""
    payload = tmp_path / "event.json"
    payload.write_text(json.dumps({"pull_request": {"number": 77}}), encoding="utf-8")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload))
    assert pr_number_from_env() == 77


def test_pr_number_is_none_outside_a_pull_request(monkeypatch):
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    assert pr_number_from_env() is None


def test_a_corrupt_event_file_does_not_raise(monkeypatch, tmp_path):
    """Nothing about posting a comment justifies crashing a CI job that has
    already computed the answer."""
    bad = tmp_path / "event.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(bad))
    assert pr_number_from_env() is None
