import os
import time
import asyncio
import logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MCP_SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["-m", "src.mcp_servers.tools_server"],
    env=os.environ.copy()   # <-- explicitly pass the parent's full environment from agent to MCP server, so it can see the API key 
                            #     and other env vars
)

# Gemini-facing tool name -> actual MCP server tool name
AGENT_TO_MCP_TOOL_NAME = {
    "get_live_weather_api": "get_live_weather",
    "search_air_quality_db": "search_air_quality_history",
}

TOOL_UNAVAILABLE_MESSAGE = (
    "Tool service is unavailable right now. Do not estimate or invent data; "
    "tell the user the tool couldn't be reached."
)


async def _run_calls(session, calls: list[tuple[str, dict]], results: list[dict] | None = None) -> list[dict]:
    """Runs each call on an open MCP session and returns one result per call,
    in the same order as `calls`. Results are appended to `results` as they
    finish, so a caller still has the completed ones if the session fails later.

    Results used to be a dict keyed by tool name, so two calls to the same tool
    in one turn (e.g. weather for London AND New Delhi) overwrote each other and
    both got the last city's data. A list keeps every call's result separate.

    Each entry: {"name", "result", "duration_s"}. duration_s is recorded for
    later use and is not passed to the model or the API response."""
    if results is None:
        results = []
    for agent_tool_name, args in calls:
        started = time.perf_counter()
        mcp_tool_name = AGENT_TO_MCP_TOOL_NAME.get(agent_tool_name)
        if mcp_tool_name is None:
            text = f"Tool '{agent_tool_name}' has no MCP mapping."
        else:
            try:
                result = await session.call_tool(mcp_tool_name, arguments=args)
                # unwrap TextContent blocks -> plain string, same shape your tools already return
                text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
            except Exception as e:
                text = f"MCP tool call failed: {e}"
        results.append({
            "name": agent_tool_name,
            "result": text,
            "duration_s": time.perf_counter() - started,
        })
    return results


async def _call_tools_via_mcp(calls: list[tuple[str, dict]]) -> list[dict]:
    """Opens ONE MCP session, runs every requested call through it, closes once.
    This matters for the London case (two tools in one turn) — one subprocess
    launch serving both calls, not two separate launches.

    Always returns one result per call. If the MCP process can't start or the
    session fails (including while closing, after calls finished), results that
    were already collected are kept and every call without one gets
    TOOL_UNAVAILABLE_MESSAGE. Only Exception is caught: asyncio.CancelledError
    and other BaseExceptions propagate."""
    results: list[dict] = []
    started = time.perf_counter()
    try:
        async with stdio_client(MCP_SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await _run_calls(session, calls, results)
    except Exception:
        logging.exception(
            f"MCP tool session failed; {len(results)} of {len(calls)} calls had completed"
        )
    elapsed = time.perf_counter() - started
    for agent_tool_name, _ in calls[len(results):]:
        results.append({
            "name": agent_tool_name,
            "result": TOOL_UNAVAILABLE_MESSAGE,
            "duration_s": elapsed,
        })
    return results


def call_tools_via_mcp_sync(calls: list[tuple[str, dict]]) -> list[dict]:
    """Sync bridge — this is the asyncio.run() piece, called from a plain def.
    Returns one result dict per call, in the order of `calls`."""
    return asyncio.run(_call_tools_via_mcp(calls))