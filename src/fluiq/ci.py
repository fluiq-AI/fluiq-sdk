"""CI quality gate: run a Fluiq dataset eval and fail the build on regressions.

Usage (e.g. in GitHub Actions):

    python -m fluiq.ci --dataset "checkout-agent" --kind metrics \
        --metrics hallucination,relevance,completeness \
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

    if avg is None:
        return _fail("Run produced no scores")
    if avg < args.fail_below:
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
            return _fail(
                f"{len(weakest)} example(s) scored below the per-example floor {args.min_example}", 1,
            )

    print("[fluiq.ci] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
