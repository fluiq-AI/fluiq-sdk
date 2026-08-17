"""Post an eval result to the pull request that caused it.

A CI gate that only sets an exit code is a gate nobody reads. The failure shows
up as a red cross next to nine other red crosses, and whoever opens the log finds
a wall of `[fluiq.ci]` lines. The reviewer — the person actually deciding whether
this merges — never sees a number.

A comment on the PR is where that decision is made, so that is where the numbers
go: what each metric scored, how it moved against the last run on the base
branch, and which examples got worse.

The comment is **updated in place** rather than appended. A PR pushed to eleven
times should carry one current verdict, not eleven stale ones, and a reviewer
scrolling past ten obsolete tables to find the live one is how a good signal
becomes noise.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

#: Marks our comment so a later run can find and replace it. Invisible in the
#: rendered markdown, which is why it is an HTML comment rather than a heading.
MARKER = "<!-- fluiq-eval-report -->"

GITHUB_API = os.getenv("GITHUB_API_URL", "https://api.github.com")


def _gh(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def pr_number_from_env() -> Optional[int]:
    """The PR this run belongs to, from GitHub Actions' own environment.

    ``GITHUB_REF`` is ``refs/pull/123/merge`` on a pull_request event. The event
    payload is the fallback, because a workflow triggered by ``workflow_run`` or
    ``issue_comment`` has a different ref but still names the PR in its payload.
    """
    ref = os.getenv("GITHUB_REF", "")
    if ref.startswith("refs/pull/"):
        parts = ref.split("/")
        if len(parts) >= 3 and parts[2].isdigit():
            return int(parts[2])

    path = os.getenv("GITHUB_EVENT_PATH")
    if path and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                event = json.load(fh)
        except (OSError, ValueError):
            return None
        for key in ("pull_request", "issue"):
            number = (event.get(key) or {}).get("number")
            if isinstance(number, int):
                return number
    return None


def _pct(value: Any) -> str:
    return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else "—"


def _delta(current: Any, baseline: Any) -> str:
    """A signed delta, or a dash when there is nothing to compare against.

    Deliberately not "0.0%" when a baseline is missing: no previous run and no
    change are different facts, and showing the second for the first would tell
    a reviewer this change is safe on the strength of no evidence.
    """
    if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)):
        return "—"
    diff = (current - baseline) * 100
    if abs(diff) < 0.05:
        return "±0.0"
    return f"{diff:+.1f}"


def _verdict(diff: str, threshold: float = 1.0) -> str:
    """An emoji per row, so the table is scannable without reading numbers."""
    if diff in ("—", "±0.0"):
        return "▫️"
    try:
        value = float(diff)
    except ValueError:
        return "▫️"
    if value <= -threshold:
        return "🔻"
    if value >= threshold:
        return "🔺"
    return "▫️"


def build_comment(
    *,
    run: Dict[str, Any],
    summary: Dict[str, Any],
    items: List[Dict[str, Any]],
    baseline_summary: Optional[Dict[str, Any]],
    gate: float,
    passed: bool,
    dashboard_url: Optional[str] = None,
) -> str:
    """The markdown body. Pure, so its shape is testable without a network."""
    metrics = summary.get("metrics") or {}
    base_metrics = (baseline_summary or {}).get("metrics") or {}
    average = summary.get("avg_score")
    if average is None:
        average = summary.get("avg_run_score")
    base_average = (baseline_summary or {}).get("avg_score") or (
        baseline_summary or {}
    ).get("avg_run_score")

    task = run.get("task") or {}
    task_label = ""
    if task:
        name = task.get("prompt_name") or "inline prompt"
        version = task.get("prompt_version")
        task_label = f"{name}{f' v{version}' if version else ''} on `{task.get('model')}`"

    headline = "✅ Evals passed" if passed else "❌ Evals failed"
    lines: List[str] = [
        MARKER,
        f"### {headline}",
        "",
        f"**{_pct(average)}** overall against a gate of **{_pct(gate)}**"
        + (f" · {_delta(average, base_average)} pts vs base" if base_average is not None else ""),
    ]
    if task_label:
        lines.append(f"Task: {task_label}")
    lines.append("")

    if metrics:
        # The comparison column only appears when there is something to compare
        # against. A column of dashes is a promise the report cannot keep, and it
        # invites the reader to hunt for the run it is silently missing.
        if base_metrics:
            lines += ["| | Metric | Score | vs base |", "|---|---|---:|---:|"]
            for name in sorted(metrics):
                diff = _delta(metrics[name], base_metrics.get(name))
                lines.append(
                    f"| {_verdict(diff)} | {name} | {_pct(metrics[name])} | {diff} |"
                )
        else:
            lines += ["| Metric | Score |", "|---|---:|"]
            for name in sorted(metrics):
                lines.append(f"| {name} | {_pct(metrics[name])} |")
        lines.append("")

    # The failing examples are the actionable part. Capped, because a reviewer
    # scrolling past forty rows is not reading any of them.
    worst = _worst_examples(items, limit=5)
    if worst:
        lines += [
            "<details><summary>Weakest examples</summary>",
            "",
            "| Score | Input |",
            "|---:|---|",
        ]
        for score, text in worst:
            lines.append(f"| {_pct(score)} | {_escape_cell(text)} |")
        lines += ["", "</details>", ""]

    if (summary.get("gen_failed") or 0) > 0:
        lines.append(
            f"> ⚠️ {summary['gen_failed']} example(s) produced no output and were "
            f"not scored, so the numbers above cover the rest."
        )
        lines.append("")

    if dashboard_url:
        lines.append(f"[Open the full run]({dashboard_url})")

    return "\n".join(lines)


def _worst_examples(items: List[Dict[str, Any]], limit: int) -> List[tuple]:
    scored: List[tuple] = []
    for item in items or []:
        result = item.get("result")
        if not isinstance(result, list) or not result:
            continue
        values = [r["score"] for r in result if isinstance(r.get("score"), (int, float))]
        if not values:
            continue
        scored.append((sum(values) / len(values), str(item.get("input") or "")))
    scored.sort(key=lambda pair: pair[0])
    return scored[:limit]


def _escape_cell(text: str) -> str:
    """Keep a table cell a table cell.

    A pipe in a customer's prompt would split the row into extra columns, and a
    newline would end the table early — both silently, and both on exactly the
    inputs most likely to be interesting.
    """
    cleaned = text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    return cleaned[:120] + ("…" if len(cleaned) > 120 else "")


def post(
    *,
    token: str,
    repo: str,
    pr: int,
    body: str,
) -> bool:
    """Create or update the report comment. Returns whether it landed.

    Never raises: a comment is the delivery of a result, not the result. A CI
    job that already knows the answer must not fail because GitHub was slow.
    """
    try:
        existing = requests.get(
            f"{GITHUB_API}/repos/{repo}/issues/{pr}/comments",
            headers=_gh(token), params={"per_page": 100}, timeout=30,
        )
        existing.raise_for_status()
        mine = next(
            (c for c in existing.json() if MARKER in (c.get("body") or "")), None,
        )
        if mine:
            response = requests.patch(
                f"{GITHUB_API}/repos/{repo}/issues/comments/{mine['id']}",
                headers=_gh(token), json={"body": body}, timeout=30,
            )
        else:
            response = requests.post(
                f"{GITHUB_API}/repos/{repo}/issues/{pr}/comments",
                headers=_gh(token), json={"body": body}, timeout=30,
            )
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        print(f"[fluiq.ci] could not post the PR comment: {exc}")
        return False


__all__ = [
    "MARKER",
    "build_comment",
    "post",
    "pr_number_from_env",
]
