from src.agent.graph import compiled_agent
from src.mcp_servers.mcp_client_helper import call_tools_via_mcp_sync


results=call_tools_via_mcp_sync([('search_air_quality_db', {'query': 'historical air quality in London'})])
print(f"Results from MCP call: {results}")