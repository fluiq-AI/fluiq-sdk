"""Internal: run a single test script, then summarize new traces."""
import json
import subprocess
import sys
from pathlib import Path

OUTPUT = Path(r"d:\ideas\FluiqAI\source\fluiq-api\output.json")


def reset():
    OUTPUT.write_text("[]", encoding="utf-8")


def summarize():
    try:
        events = json.loads(OUTPUT.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"FAILED to read output.json: {e}")
        return
    if not events:
        print("NO EVENTS CAPTURED")
        return
    for i, entry in enumerate(events):
        e = entry.get("event") or {}
        integ = e.get("integration")
        typ = e.get("type")
        api = e.get("api")
        model = e.get("model")
        fn = e.get("function")
        success = e.get("success")
        latency = e.get("latency")
        tokens = e.get("tokens") or {}
        parent = e.get("parent_id")
        tid = e.get("trace_id")
        resp = e.get("response")
        resp_str = (
            (resp[:80] + "...") if isinstance(resp, str) and len(resp) > 80
            else str(resp)[:80] if resp is not None else "-"
        )
        print(
            f"[{i}] integ={integ} type={typ} api={api or '-'} "
            f"model={model or '-'} fn={fn or '-'} success={success} "
            f"latency={f'{latency:.2f}s' if isinstance(latency, (int, float)) else '-'} "
            f"tokens={tokens.get('total') or tokens.get('prompt') or '-'} "
            f"parent={'Y' if parent else '-'} tid={(tid or '')[:8]} resp={resp_str!r}"
        )


def run(script):
    reset()
    print(f"\n=== RUNNING {script} ===")
    p = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True, timeout=180,
    )
    if p.stdout.strip():
        print("STDOUT:", p.stdout.strip()[:500])
    if p.returncode != 0:
        print("STDERR:", p.stderr.strip()[:1000])
        print(f"EXIT: {p.returncode}")
    print(f"--- TRACES ({script}) ---")
    summarize()


if __name__ == "__main__":
    for script in sys.argv[1:]:
        run(script)
