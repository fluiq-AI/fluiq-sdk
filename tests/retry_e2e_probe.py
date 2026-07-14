"""Probe for the worker seek-back-retry fix: emits one @trace span with a
marker name passed via argv, so the audit can verify it lands in ClickHouse
even when the trace was sent during a ClickHouse outage."""
import sys
import time

from fluiq import instrument, trace

from .keys import FLUIQ_API_KEY

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

marker = sys.argv[1] if len(sys.argv) > 1 else "retry_probe"


@trace(name=marker)
def probe() -> str:
    return f"probe payload for {marker}"


if __name__ == "__main__":
    print(probe())
    time.sleep(3)  # let the background sender flush
