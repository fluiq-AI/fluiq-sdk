"""
fluiq.optimize() — MCP list_tools() caching.

When optimize() is active, ClientSession.list_tools() responses are cached in
Redis keyed by the server URL.  The cache is automatically invalidated when
session.initialize() is called (server restart signal).

On a cache hit the SDK returns the cached tool schema instantly and emits a
type="mcp" / kind="mcp_list_tools" trace with cache_hit=True.
On a cache miss it calls the real server, stores the result, and emits
cache_hit=False.  Both appear in the Optimize dashboard ▸ By cache type.

Run:  python -m tests.optimize.mcp_list_tools_cache_test
"""
import asyncio
import time
from mcp import ClientSession
import fluiq
from ..keys import FLUIQ_API_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.optimize()

# Import AFTER fluiq.instrument() so the module-level patch is applied first,
# giving _get_session_url access to the server URL via the ContextVar.
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = "https://mcp.deepwiki.com/mcp"


async def run():
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # initialize() also invalidates any stale list_tools cache for this URL

            # ── Call 1: cache miss — hits the real MCP server ─────────────────
            t0 = time.perf_counter()
            result1 = await session.list_tools()
            t1 = time.perf_counter()
            tool_names = [t.name for t in (result1.tools or [])]
            print(f"[list_tools call 1] {(t1 - t0) * 1000:.0f} ms  tools={tool_names}")

            # ── Call 2: cache hit — served from Redis ─────────────────────────
            t2 = time.perf_counter()
            result2 = await session.list_tools()
            t3 = time.perf_counter()
            tool_names2 = [t.name for t in (result2.tools or [])]
            print(f"[list_tools call 2] {(t3 - t2) * 1000:.0f} ms  tools={tool_names2}")

            assert tool_names == tool_names2, "Cache returned different tool list!"
            print(f"\nOK — tool list matches.  Call 2 should be significantly faster.")
            print("Check Optimize dashboard ▸ By cache type ▸ mcp_list_tools for hit/miss counts.")


asyncio.run(run())
