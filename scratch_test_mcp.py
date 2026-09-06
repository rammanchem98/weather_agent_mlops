import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import logging


logger= logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

server_params = StdioServerParameters(
    command="python",
    args=["-m", "src.mcp_servers.tools_server"],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            logger.info("Available tools:")
            for tool in tools.tools:
                logger.info(f"  - {tool.name}: {tool.description[:60]}...")

            logger.info("\nCalling get_live_weather('Tokyo')...")
            result = await session.call_tool("get_live_weather", arguments={"city": "Tokyo"})
            logger.info(f"Result: {result.content}")

if __name__ == "__main__":
    asyncio.run(main())