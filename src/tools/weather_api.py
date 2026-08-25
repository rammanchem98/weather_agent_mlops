import logging
import requests

from src.common.config import OPENWEATHER_API_KEY as API_KEY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_live_weather_api(city: str) -> str:
    """Fetches real-time weather metrics, temperature, and humidity conditions for a city.

    Args:
        city: The name of the target city.
    """
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        desc = data['weather'][0]['description']

        return f"Current weather in {city}: {temp}°C, Humidity: {humidity}%, Condition: {desc}."

    return f"Error: Unable to fetch weather data for {city}."
