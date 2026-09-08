# src/mcp_servers/mcp_client_helper.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MCP_SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["-m", "src.mcp_servers.tools_server"],
)

# Gemini-facing tool name -> actual MCP server tool name
AGENT_TO_MCP_TOOL_NAME = {
    "get_live_weather_api": "get_live_weather",
    "search_air_quality_db": "search_air_quality_history",
}

async def _call_tools_via_mcp(calls: list[tuple[str, dict]]) -> dict[str, str]:
    """Opens ONE MCP session, runs every requested call through it, closes once.
    This matters for the London case (two tools in one turn) — one subprocess
    launch serving both calls, not two separate launches."""
    results = {}
    async with stdio_client(MCP_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for agent_tool_name, args in calls:
                mcp_tool_name = AGENT_TO_MCP_TOOL_NAME.get(agent_tool_name)
                if mcp_tool_name is None:
                    results[agent_tool_name] = f"Tool '{agent_tool_name}' has no MCP mapping."
                    continue
                try:
                    result = await session.call_tool(mcp_tool_name, arguments=args)
                    # unwrap TextContent blocks -> plain string, same shape your tools already return
                    text_parts = [block.text for block in result.content if hasattr(block, "text")]
                    results[agent_tool_name] = "\n".join(text_parts)
                except Exception as e:
                    results[agent_tool_name] = f"MCP tool call failed: {e}"
    return results

def call_tools_via_mcp_sync(calls: list[tuple[str, dict]]) -> dict[str, str]:
    """Sync bridge — this is the asyncio.run() piece, called from a plain def."""
    return asyncio.run(_call_tools_via_mcp(calls))