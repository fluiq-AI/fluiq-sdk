"""
fluiq.optimize() — MCP call_tool() caching.

When optimize() is active, ClientSession.call_tool() results are cached in
Redis keyed by (server_url, tool_name, sorted_arguments).  Identical calls
skip the MCP server entirely and return the cached result instantly.

Error results (isError=True) are never cached.

Run:  python -m tests.optimize.mcp_call_tool_cache_test
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

            # Discover available tools first
            tools_result = await session.list_tools()
            tools = tools_result.tools or []
            if not tools:
                print("No tools found on the MCP server — exiting.")
                return

            tool = tools[0]
            print(f"Using tool: '{tool.name}'")

            # Build minimal valid arguments from the tool's input schema
            schema = getattr(tool, "inputSchema", None) or {}
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            required = schema.get("required", []) if isinstance(schema, dict) else []

            # Use the first required string property, or fall back to an empty dict
            args: dict = {}
            for prop_name in required:
                prop_schema = props.get(prop_name, {})
                if prop_schema.get("type") == "string":
                    args[prop_name] = "model context protocol"
                    break

            print(f"Arguments: {args}")

            # ── Call 1: cache miss — hits the real MCP server ─────────────────
            t0 = time.perf_counter()
            result1 = await session.call_tool(tool.name, args)
            t1 = time.perf_counter()
            content1 = result1.content[0] if result1.content else None
            text1 = getattr(content1, "text", str(content1))[:100] if content1 else ""
            print(f"\n[call_tool call 1] {(t1 - t0) * 1000:.0f} ms")
            print(f"  isError={result1.isError}  result={text1!r}")

            # ── Call 2: identical args — served from Redis cache ──────────────
            t2 = time.perf_counter()
            result2 = await session.call_tool(tool.name, args)
            t3 = time.perf_counter()
            content2 = result2.content[0] if result2.content else None
            text2 = getattr(content2, "text", str(content2))[:100] if content2 else ""
            print(f"[call_tool call 2] {(t3 - t2) * 1000:.0f} ms")
            print(f"  isError={result2.isError}  result={text2!r}")

            assert text1 == text2, "Cache returned different result content!"
            print(f"\nOK — results match.  Call 2 should be significantly faster.")
            print("Check Optimize dashboard ▸ By cache type ▸ mcp_call for hit/miss counts.")


asyncio.run(run())
