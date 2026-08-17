"""CI quality gate: run a Fluiq dataset eval and fail the build on regressions.

Grade what the dataset already holds::

    python -m fluiq.ci --dataset "checkout-agent" --kind metrics \
        --metrics hallucination,relevance,completeness \
        --fail-below 0.7

Or run the prompt *this branch changed* against every example and grade the
fresh output — the form that actually gates a pull request::

    python -m fluiq.ci --dataset "checkout-agent" --kind metrics \
        --task-prompt support-reply --task-model gpt-5-mini \
        --fail-below 0.7

    env:
      FLUIQ_API_KEY: ${{ secrets.FLUIQ_API_KEY }}

Launches a batch eval over every example in the dataset, polls until the report
completes, prints the per-metric summary, and exits non-zero when the average
score is below ``--fail-below`` (or any example is below ``--min-example``),
so the CI job fails before a quality regression ships.

Exit codes: 0 pass · 1 threshold failed · 2 error/timeout.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import requests


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


def _fail(msg: str, code: int = 2) -> int:
    # ::error makes the message surface in the GitHub Actions annotations UI;
    # it's harmless noise on any other CI.
    print(f"::error::{msg}")
    print(f"[fluiq.ci] FAIL: {msg}")
    return code


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fluiq.ci",
        description="Run a Fluiq dataset eval as a CI quality gate.",
    )
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (case-insensitive) or dataset UUID")
    parser.add_argument("--kind", choices=("metrics", "agentic"), default="metrics")
    parser.add_argument("--metrics", default="hallucination,relevance",
                        help="Comma-separated metrics for --kind metrics")
    parser.add_argument("--depth", choices=("fast", "standard", "deep"), default=None,
                        help="Agentic depth for --kind agentic")
    parser.add_argument("--fail-below", type=float, default=0.7,
                        help="Fail when the run's average score is below this (default 0.7)")
    parser.add_argument("--min-example", type=float, default=None,
                        help="Also fail when ANY example's average score is below this")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Seconds to wait for the run to complete (default 600)")
    parser.add_argument("--poll", type=int, default=10,
                        help="Seconds between report polls (default 10)")
    parser.add_argument("--api-key", default=os.getenv("FLUIQ_API_KEY", ""))
    parser.add_argument("--endpoint",
                        default=os.getenv("FLUIQ_API_ENDPOINT", "https://api.getfluiq.com/api"))

    task = parser.add_argument_group(
        "task",
        "Run a prompt against every example and grade the fresh output. Without "
        "these, the run grades the output each example already carries.",
    )
    task.add_argument("--task-prompt", default=None,
                      help="Saved prompt slug (or UUID) to execute as the task")
    task.add_argument("--task-prompt-version", type=int, default=None,
                      help="Pin the saved prompt to a version")
    task.add_argument("--task-template", default=None,
                      help="Inline task template, e.g. 'Answer: {{input}}'")
    task.add_argument("--task-template-file", default=None,
                      help="Read the task template from a file (version it with your code)")
    task.add_argument("--task-model", default=None,
                      help="Model the task runs on (defaults to the saved prompt's model)")
    task.add_argument("--task-system", default=None, help="System prompt for the task")
    task.add_argument("--trials", type=int, default=1, metavar="N",
                      help="Run each example N times and average (default 1). "
                           "Use when a gate keeps flapping — it separates a real "
                           "regression from a model that is simply noisy.")
    task.add_argument("--name", default=os.getenv("GITHUB_REF_NAME") or None,
                      help="Name this run (defaults to the branch on GitHub Actions)")

    report = parser.add_argument_group("pull request report")
    report.add_argument("--pr-comment", action="store_true",
                        help="Post the result as a comment on the pull request")
    report.add_argument("--github-token", default=os.getenv("GITHUB_TOKEN", ""),
                        help="Token with pull-requests: write (GITHUB_TOKEN in Actions)")
    report.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", ""),
                        help="owner/name (set automatically on GitHub Actions)")
    report.add_argument("--pr", type=int, default=None,
                        help="PR number (inferred from the Actions environment)")
    report.add_argument("--baseline-run", default=None,
                        help="Run id to show deltas against; defaults to the "
                             "most recent earlier run on the same dataset")
    report.add_argument("--dashboard-url", default=os.getenv("FLUIQ_DASHBOARD_URL", ""),
                        help="Base dashboard URL, for a link back to the run")

    args = parser.parse_args(argv)

    if not args.api_key:
        return _fail("No API key: pass --api-key or set FLUIQ_API_KEY")

    base = f"{args.endpoint.rstrip('/')}/v1"

    body: dict = {"kind": args.kind}
    ds = args.dataset.strip()
    if len(ds) == 36 and ds.count("-") == 4:
        body["dataset_id"] = ds
    else:
        body["dataset_name"] = ds
    if args.kind == "metrics":
        body["metrics"] = [m.strip() for m in args.metrics.split(",") if m.strip()]
    elif args.depth:
        body["depth"] = args.depth
    if args.name:
        body["name"] = args.name

    template = args.task_template
    if args.task_template_file:
        if template:
            return _fail("Pass --task-template or --task-template-file, not both")
        try:
            with open(args.task_template_file, encoding="utf-8") as fh:
                template = fh.read()
        except OSError as e:
            return _fail(f"Could not read --task-template-file: {e}")

    if args.task_prompt or template:
        if args.task_prompt and template:
            return _fail("Pass a saved prompt or an inline template, not both")
        task_body: dict = {}
        if args.task_prompt:
            ref = args.task_prompt.strip()
            if len(ref) == 36 and ref.count("-") == 4:
                task_body["prompt_id"] = ref
            else:
                task_body["prompt_slug"] = ref
            if args.task_prompt_version is not None:
                task_body["prompt_version"] = args.task_prompt_version
        else:
            task_body["template"] = template
        if args.task_model:
            task_body["model"] = args.task_model
        if args.task_system:
            task_body["system"] = args.task_system
        body["task"] = task_body
        # Only sent with a task. Scoring one recorded output N times measures
        # the judge's variance rather than the model's, and bills N× for it.
        if args.trials and args.trials > 1:
            body["trials"] = max(1, min(int(args.trials), 10))

    try:
        r = requests.post(f"{base}/ci/eval-runs", json=body,
                          headers=_headers(args.api_key), timeout=30)
        if r.status_code >= 400:
            detail = ""
            try:
                detail = r.json().get("detail", "")
            except Exception:
                pass
            return _fail(f"Launch failed ({r.status_code}): {detail or r.text[:200]}")
        run = r.json()
    except requests.RequestException as e:
        return _fail(f"Launch request failed: {e}")

    run_id = run["run_id"]
    total = run.get("total", run.get("item_count", "?"))
    task_info = run.get("task") or {}
    if task_info:
        label = task_info.get("prompt_name") or "inline template"
        version = task_info.get("prompt_version")
        label = f"{label} v{version}" if version else label
        print(f"[fluiq.ci] Task: {label} on {task_info.get('model')}")
    print(f"[fluiq.ci] Launched {args.kind} run {run_id} over {total} example(s)")

    deadline = time.monotonic() + args.timeout
    report = None
    while time.monotonic() < deadline:
        time.sleep(args.poll)
        try:
            r = requests.get(f"{base}/ci/eval-runs/{run_id}",
                             headers=_headers(args.api_key), timeout=30)
            r.raise_for_status()
            report = r.json()
        except requests.RequestException as e:
            print(f"[fluiq.ci] poll failed ({e}); retrying")
            continue
        summary = report.get("summary") or {}
        done, tot = summary.get("completed", 0), summary.get("total", 0)
        print(f"[fluiq.ci] {done}/{tot} examples scored")
        if report.get("run", {}).get("status") != "running":
            break
    else:
        return _fail(f"Timed out after {args.timeout}s waiting for run {run_id}")

    if report is None:
        return _fail("Never received a report")

    summary = report.get("summary") or {}
    avg = summary.get("avg_score") if args.kind == "metrics" else summary.get("avg_run_score")

    print(f"[fluiq.ci] Average score: {_fmt(avg)} (gate: {args.fail_below})")
    for name, score in sorted((summary.get("metrics") or {}).items()):
        print(f"[fluiq.ci]   {name}: {_fmt(score)}")
    for name, score in sorted((summary.get("layers") or {}).items()):
        print(f"[fluiq.ci]   layer {name}: {_fmt(score)}")

    # Examples whose task never produced output are excluded from the scores
    # above, so say so rather than letting a high average hide a half-run task.
    gen_failed = summary.get("gen_failed") or 0
    if gen_failed:
        print(f"::warning::[fluiq.ci] {gen_failed} example(s) failed to generate and were not scored")
        for item in (report.get("items") or []):
            if item.get("gen_error"):
                print(f"[fluiq.ci]   generation failed: {str(item['gen_error'])[:160]}")

    if avg is None:
        return _fail("Run produced no scores")
    if avg < args.fail_below:
        # Reported before failing: a red check with no explanation on the PR is
        # the situation this flag exists to end.
        if args.pr_comment:
            _post_pr_comment(args, base, run_id, report, summary, passed=False)
        return _fail(f"Average score {_fmt(avg)} is below the gate {args.fail_below}", 1)

    if args.min_example is not None:
        weakest: list[tuple[float, str]] = []
        for item in report.get("items") or []:
            result = item.get("result")
            if not isinstance(result, list) or not result:
                continue
            scores = [r["score"] for r in result if isinstance(r.get("score"), (int, float))]
            if scores:
                ex_avg = sum(scores) / len(scores)
                if ex_avg < args.min_example:
                    weakest.append((ex_avg, str(item.get("input") or item.get("example_id"))[:80]))
        if weakest:
            weakest.sort()
            for ex_avg, label in weakest[:10]:
                print(f"[fluiq.ci]   below floor ({_fmt(ex_avg)}): {label}")
            if args.pr_comment:
                _post_pr_comment(args, base, run_id, report, summary, passed=False)
            return _fail(
                f"{len(weakest)} example(s) scored below the per-example floor {args.min_example}", 1,
            )

    if args.pr_comment:
        _post_pr_comment(args, base, run_id, report, summary, passed=True)

    print("[fluiq.ci] PASS")
    return 0


def _post_pr_comment(args, base: str, run_id: str, report: dict, summary: dict, *, passed: bool) -> None:
    """Build and post the PR report. Never fatal — the gate already decided."""
    from fluiq import pr_comment

    if not args.github_token or not args.repo:
        print("::warning::[fluiq.ci] --pr-comment needs GITHUB_TOKEN and GITHUB_REPOSITORY")
        return
    pr = args.pr or pr_comment.pr_number_from_env()
    if not pr:
        print("::warning::[fluiq.ci] --pr-comment could not determine the PR number")
        return

    run = report.get("run") or {}
    baseline_summary = _baseline_summary(args, base, run, args.baseline_run)
    url = None
    if args.dashboard_url:
        url = f"{args.dashboard_url.rstrip('/')}/dashboard/datasets"

    body = pr_comment.build_comment(
        run=run,
        summary=summary,
        items=report.get("items") or [],
        baseline_summary=baseline_summary,
        gate=args.fail_below,
        passed=passed,
        dashboard_url=url,
    )
    if pr_comment.post(token=args.github_token, repo=args.repo, pr=pr, body=body):
        print(f"[fluiq.ci] Posted the report to {args.repo}#{pr}")


def _baseline_summary(args, base: str, run: dict, explicit: str | None) -> dict | None:
    """The run to show deltas against.

    A number on its own says how good the branch is; a delta says whether this
    change made it worse, which is the question a reviewer is actually asking.
    Defaults to the most recent earlier run on the same dataset — usually the
    last run on the base branch, which is exactly the comparison wanted.
    """
    try:
        if explicit:
            r = requests.get(f"{base}/ci/eval-runs/{explicit}",
                             headers=_headers(args.api_key), timeout=30)
            r.raise_for_status()
            return (r.json() or {}).get("summary")

        dataset_id = run.get("dataset_id")
        if not dataset_id:
            return None
        r = requests.get(f"{base}/datasets/{dataset_id}/runs?limit=25",
                         headers=_headers(args.api_key), timeout=30)
        r.raise_for_status()
        for candidate in r.json().get("runs", []):
            if candidate.get("run_id") == run.get("run_id"):
                continue
            if candidate.get("status") != "complete":
                continue
            if candidate.get("kind") != run.get("kind"):
                # Comparing a metrics run against an agentic one would diff two
                # different scales and call the difference a regression.
                continue
            return candidate.get("summary")
    except requests.RequestException:
        # A missing baseline costs the deltas, not the report.
        return None
    return None


if __name__ == "__main__":
    sys.exit(main())
