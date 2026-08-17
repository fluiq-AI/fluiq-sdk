"""``fluiq`` command line: push resources, run evals, serve them remotely.

    python -m fluiq.cli push     fluiq_resources.py
    python -m fluiq.cli eval     evals/ --fail-below 0.7
    python -m fluiq.cli serve    evals/

``push`` sends code-declared prompts, scorers, and datasets to the platform, so
the version that ships and the version in the repo are the same thing.

``eval`` runs a code-defined eval: **your task executes here**, in your process,
with your dependencies and your secrets, and only inputs and outputs travel to
Fluiq to be scored. That is what lets it evaluate an application that isn't
reachable from the internet, and what stops the eval drifting from the app.

``serve`` keeps a process alive that will run those same evals on request, so a
non-technical teammate can trigger one from the dashboard without needing your
laptop's environment on theirs.

Exit codes: 0 pass · 1 threshold failed · 2 error.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

DEFAULT_ENDPOINT = os.getenv("FLUIQ_API_ENDPOINT", "https://api.getfluiq.com/api")
#: Eval files are discovered by name, so a directory can be handed over whole.
EVAL_PATTERNS = ("*_eval.py", "eval_*.py", "*.eval.py")
RESULT_BATCH = 50


def _headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _fail(message: str, code: int = 2) -> int:
    # ::error surfaces in GitHub Actions annotations; harmless anywhere else.
    print(f"::error::{message}")
    print(f"[fluiq] FAIL: {message}")
    return code


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "—"


# ── Loading user files ────────────────────────────────────────────────────────

def _load(path: Path) -> None:
    """Import a file for its side effect of declaring resources or evals.

    Imported under a unique module name and with its directory on ``sys.path``,
    so an eval file can import the application it is testing the same way a test
    would.
    """
    directory = str(path.parent.resolve())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location(f"_fluiq_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def _discover(target: str) -> List[Path]:
    """Files to load: one path, or every eval-shaped file under a directory."""
    path = Path(target)
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    found: List[Path] = []
    for pattern in EVAL_PATTERNS:
        found.extend(sorted(path.rglob(pattern)))
    # rglob patterns overlap (`x_eval.py` matches two of them), so dedupe while
    # keeping order.
    seen, unique = set(), []
    for item in found:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


# ── push ──────────────────────────────────────────────────────────────────────

def cmd_push(args: argparse.Namespace) -> int:
    from fluiq import resources

    paths = _discover(args.target) if Path(args.target).is_dir() else [Path(args.target)]
    if not paths:
        return _fail(f"No file found at {args.target}")

    for path in paths:
        try:
            _load(path)
        except Exception as exc:  # noqa: BLE001
            return _fail(f"Could not load {path}: {exc!r}")

    payload = resources.payload()
    counts = {k: len(v) for k, v in payload.items()}
    if not any(counts.values()):
        return _fail(
            "Nothing declared. Construct Prompt/Scorer/Dataset at module level "
            "in the file you pushed."
        )

    print(
        f"[fluiq] Pushing {counts['prompts']} prompt(s), "
        f"{counts['scorers']} scorer(s), {counts['datasets']} dataset(s)"
    )
    try:
        response = requests.post(
            f"{args.endpoint.rstrip('/')}/v1/resources/push",
            json=payload, headers=_headers(args.api_key), timeout=60,
        )
    except requests.RequestException as exc:
        return _fail(f"Push request failed: {exc}")
    if response.status_code >= 400:
        return _fail(f"Push failed ({response.status_code}): {response.text[:300]}")

    body = response.json()
    for bucket in ("prompts", "scorers", "datasets"):
        for slug in body.get("created", {}).get(bucket, []):
            print(f"[fluiq]   + {bucket[:-1]} {slug}")
        for slug in body.get("updated", {}).get(bucket, []):
            print(f"[fluiq]   ~ {bucket[:-1]} {slug}")

    errors = body.get("errors") or []
    for message in errors:
        print(f"::warning::[fluiq] {message}")
    if errors and args.strict:
        # Off by default: a partial push that reports what it skipped is more
        # useful than one that refuses everything over a single bad scorer.
        return _fail(f"{len(errors)} resource(s) rejected (--strict)", 1)
    print("[fluiq] Push complete")
    return 0


# ── eval ──────────────────────────────────────────────────────────────────────

def cmd_eval(args: argparse.Namespace) -> int:
    from fluiq.evals import definition

    paths = _discover(args.target)
    if not paths:
        return _fail(
            f"No eval files under {args.target}. Name them "
            f"{', '.join(EVAL_PATTERNS)}."
        )
    for path in paths:
        try:
            _load(path)
        except Exception as exc:  # noqa: BLE001
            return _fail(f"Could not load {path}: {exc!r}")

    evals = definition.declared()
    if not evals:
        return _fail("No Eval(...) declared in the files loaded.")
    if args.name:
        evals = [e for e in evals if e.name == args.name]
        if not evals:
            return _fail(f"No eval named {args.name!r}")

    worst = 0
    for spec in evals:
        code = _run_one(spec, args)
        worst = max(worst, code)
    return worst


def _run_one(spec: Any, args: argparse.Namespace) -> int:
    base = f"{args.endpoint.rstrip('/')}/v1"
    print(f"\n[fluiq] {spec.name}")

    rows = _resolve_data(spec, base, args.api_key)
    if rows is None:
        return 2
    if not rows:
        return _fail(f"{spec.name}: no examples to run")
    if spec.trials > 1:
        # Repeat then average, to blunt judge variance. Expanded here rather than
        # server-side so each trial is a real, independent invocation of the task.
        rows = [row for row in rows for _ in range(spec.trials)]

    try:
        response = requests.post(
            f"{base}/local-runs",
            json={
                "name":          spec.name,
                "description":   spec.description,
                "dataset":       spec.dataset,
                "metrics":       spec.remote_scorers,
                "judge":         spec.judge,
                "task_label":    getattr(spec.task, "__name__", "task"),
            },
            headers=_headers(args.api_key), timeout=30,
        )
        response.raise_for_status()
        run_id = response.json()["run_id"]
    except requests.RequestException as exc:
        return _fail(f"{spec.name}: could not start run: {exc}")

    print(f"[fluiq] Run {run_id} — executing task over {len(rows)} example(s) locally")

    results, failures = _execute(spec, rows, args)
    if failures:
        print(f"::warning::[fluiq] {failures} example(s) raised and were skipped")
    if not results:
        return _fail(f"{spec.name}: every example failed to run")

    for start in range(0, len(results), RESULT_BATCH):
        batch = results[start:start + RESULT_BATCH]
        try:
            requests.post(
                f"{base}/local-runs/{run_id}/results",
                json={"results": batch}, headers=_headers(args.api_key), timeout=60,
            ).raise_for_status()
        except requests.RequestException as exc:
            return _fail(f"{spec.name}: could not submit results: {exc}")
    try:
        requests.post(
            f"{base}/local-runs/{run_id}/finish",
            headers=_headers(args.api_key), timeout=30,
        )
    except requests.RequestException:
        pass  # Scoring is already queued; a missed close only affects the report.

    return _await_report(
        spec, base, run_id, args,
        local_only=not spec.remote_scorers,
        local_scores=_average_local(results),
    )


def _resolve_data(spec: Any, base: str, api_key: str) -> Optional[List[Dict[str, Any]]]:
    """Inline rows, or a dataset's examples pulled from the platform."""
    if spec.data is not None:
        return [dict(row) for row in spec.data]
    try:
        response = requests.get(
            f"{base}/datasets", headers=_headers(api_key), timeout=30,
        )
        response.raise_for_status()
        wanted = (spec.dataset or "").strip().lower()
        match = next(
            (d for d in response.json().get("datasets", [])
             if str(d.get("name") or "").strip().lower() == wanted),
            None,
        )
        if match is None:
            _fail(f"{spec.name}: no dataset named {spec.dataset!r}")
            return None
        examples = requests.get(
            f"{base}/datasets/{match['dataset_id']}/examples?limit=500",
            headers=_headers(api_key), timeout=30,
        )
        examples.raise_for_status()
        return [
            {
                "input":    e.get("input") or "",
                "expected": e.get("expected_output"),
                "metadata": e.get("metadata") or {},
            }
            for e in examples.json().get("examples", [])
        ]
    except requests.RequestException as exc:
        _fail(f"{spec.name}: could not read dataset: {exc}")
        return None


def _execute(spec: Any, rows: List[Dict[str, Any]], args: argparse.Namespace):
    """Run the task over every row, in parallel, isolating failures.

    One example raising is a bug in that example's path, not a reason to abandon
    the other ninety-nine — the run reports how many were skipped.
    """
    results: List[Dict[str, Any]] = []
    failures = 0

    def one(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            output = spec.run_task(row)
        except Exception as exc:  # noqa: BLE001
            if args.verbose:
                print(f"[fluiq]   task raised on {str(row.get('input'))[:60]!r}: {exc!r}")
            return None
        return {
            "input":    str(row.get("input") or ""),
            "output":   output,
            "expected": row.get("expected"),
            "metadata": row.get("metadata") or {},
            "local_scores": spec.score_locally(
                output=output,
                expected=str(row.get("expected") or ""),
                input=str(row.get("input") or ""),
            ),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=spec.concurrency) as pool:
        for outcome in pool.map(one, rows):
            if outcome is None:
                failures += 1
            else:
                results.append(outcome)
    return results, failures


def _await_report(
    spec: Any,
    base: str,
    run_id: str,
    args: argparse.Namespace,
    local_only: bool,
    local_scores: Dict[str, float],
) -> int:
    """Poll for the scored report and apply the gate."""
    if local_only:
        # Nothing was sent to a judge, so there is nothing to wait for.
        return _apply_gate(spec, local_scores, {}, args, run_id)

    deadline = time.monotonic() + args.timeout
    report: Optional[Dict[str, Any]] = None
    while time.monotonic() < deadline:
        time.sleep(args.poll)
        try:
            response = requests.get(
                f"{base}/ci/eval-runs/{run_id}", headers=_headers(args.api_key), timeout=30,
            )
            response.raise_for_status()
            report = response.json()
        except requests.RequestException as exc:
            print(f"[fluiq] poll failed ({exc}); retrying")
            continue
        summary = report.get("summary") or {}
        print(f"[fluiq]   {summary.get('completed', 0)}/{summary.get('total', 0)} scored")
        if (report.get("run") or {}).get("status") != "running":
            break
    else:
        return _fail(f"{spec.name}: timed out after {args.timeout}s waiting for scores")

    summary = (report or {}).get("summary") or {}
    return _apply_gate(spec, local_scores, summary, args, run_id)


def _average_local(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mean of each local scorer across the examples it actually scored.

    Averaged over the examples that produced a score, not over every example: a
    scorer that raised on half the rows should report the half it managed, not
    be diluted toward zero by the half it never saw.
    """
    totals: Dict[str, List[float]] = {}
    for result in results:
        for name, value in (result.get("local_scores") or {}).items():
            totals.setdefault(name, []).append(value)
    return {name: sum(v) / len(v) for name, v in totals.items() if v}


def _apply_gate(
    spec: Any,
    local_scores: Dict[str, float],
    summary: Dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
) -> int:
    remote = {k: v for k, v in (summary.get("metrics") or {}).items() if v is not None}
    combined = {**remote, **local_scores}

    for name, score in sorted(combined.items()):
        print(f"[fluiq]   {name}: {_fmt(score)}")
    if not combined:
        return _fail(f"{spec.name}: produced no scores")

    average = sum(combined.values()) / len(combined)
    print(f"[fluiq] {spec.name}: {_fmt(average)} (gate {args.fail_below}) · run {run_id}")
    if average < args.fail_below:
        return _fail(
            f"{spec.name}: average {_fmt(average)} is below the gate {args.fail_below}", 1,
        )
    return 0


# ── serve (remote evals) ──────────────────────────────────────────────────────

def cmd_serve(args: argparse.Namespace) -> int:
    """Keep code-defined evals runnable from the dashboard.

    The task lives on this machine, so the platform cannot call it — a laptop
    has no address a server can reach. Instead this polls for run requests and
    executes them here, which needs no tunnel, no inbound port, and no exception
    in anyone's firewall.
    """
    from fluiq.evals import definition

    paths = _discover(args.target)
    for path in paths:
        try:
            _load(path)
        except Exception as exc:  # noqa: BLE001
            return _fail(f"Could not load {path}: {exc!r}")

    evals = {e.name: e for e in definition.declared()}
    if not evals:
        return _fail("No Eval(...) declared in the files loaded.")

    base = f"{args.endpoint.rstrip('/')}/v1"
    try:
        requests.post(
            f"{base}/remote-evals/register",
            json={
                "evals": [
                    {
                        "name": e.name,
                        "description": e.description,
                        "dataset": e.dataset,
                        "scorers": e.remote_scorers + [
                            getattr(s, "__name__", "scorer") for s in e.local_scorers
                        ],
                    }
                    for e in evals.values()
                ]
            },
            headers=_headers(args.api_key), timeout=30,
        ).raise_for_status()
    except requests.RequestException as exc:
        return _fail(f"Could not register evals: {exc}")

    print(f"[fluiq] Serving {len(evals)} eval(s); Ctrl-C to stop")
    for name in evals:
        print(f"[fluiq]   · {name}")

    while True:
        try:
            response = requests.get(
                f"{base}/remote-evals/next", headers=_headers(args.api_key), timeout=40,
            )
            if response.status_code == 204:
                continue
            response.raise_for_status()
            request = response.json()
        except requests.RequestException:
            time.sleep(args.poll)
            continue
        except KeyboardInterrupt:
            print("\n[fluiq] Stopped")
            return 0

        spec = evals.get(request.get("eval"))
        if spec is None:
            continue
        print(f"[fluiq] Running {spec.name} on request")
        run_args = argparse.Namespace(**{**vars(args), "fail_below": 0.0})
        try:
            _run_one(spec, run_args)
        except Exception as exc:  # noqa: BLE001
            print(f"[fluiq] run failed: {exc!r}")


# ── entry point ───────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="fluiq", description=__doc__)
    parser.add_argument("--api-key", default=os.getenv("FLUIQ_API_KEY", ""))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    push = sub.add_parser("push", help="Upload code-declared prompts, scorers, datasets")
    push.add_argument("target", help="File or directory declaring resources")
    push.add_argument("--strict", action="store_true",
                      help="Exit non-zero if any resource was rejected")
    push.set_defaults(func=cmd_push)

    ev = sub.add_parser("eval", help="Run code-defined evals; the task runs locally")
    ev.add_argument("target", help="Eval file or directory")
    ev.add_argument("--name", default=None, help="Run only the eval with this name")
    ev.add_argument("--fail-below", type=float, default=0.7)
    ev.add_argument("--timeout", type=int, default=600)
    ev.add_argument("--poll", type=int, default=5)
    ev.set_defaults(func=cmd_eval)

    serve = sub.add_parser("serve", help="Keep evals runnable from the dashboard")
    serve.add_argument("target", help="Eval file or directory")
    serve.add_argument("--poll", type=int, default=5)
    serve.add_argument("--timeout", type=int, default=600)
    serve.add_argument("--fail-below", type=float, default=0.0)
    serve.add_argument("--name", default=None)
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    if not args.api_key:
        return _fail("No API key: pass --api-key or set FLUIQ_API_KEY")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
