from mcp.server.fastmcp import FastMCP

from src.tools.weather_api import get_live_weather_api
from src.tools.vector_search import search_air_quality_db

mcp = FastMCP("weather-agent-tools")


@mcp.tool()
def get_live_weather(city: str) -> str:
    """Fetches real-time weather metrics — temperature, humidity, and
    conditions — for a given city. Use for CURRENT/LIVE weather only.

    Args:
        city: Name of the city to fetch weather for.
    """
    return get_live_weather_api(city)

@mcp.tool()
def search_air_quality_history(query: str) -> str:
    """Searches a local historical database for STORED, PAST air quality and
    weather records (temperature, humidity, PM2.5) for world cities. Use for
    historical/past/stored data — NOT for current/live conditions.

    Args:
        query: Natural language description of the historical data being requested.
    """
    return search_air_quality_db(query)

if __name__ == "__main__":
    mcp.run(transport="stdio")