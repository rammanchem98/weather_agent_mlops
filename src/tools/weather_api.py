import logging
import requests

from src.common.config import OPENWEATHER_API_KEY as API_KEY
from src.common.redact import redact_secrets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

WEATHER_UNAVAILABLE_MESSAGE = "Weather service is unavailable right now"


def get_live_weather_api(city: str) -> str:
    """Fetches real-time weather metrics, temperature, and humidity conditions for a city.

    Args:
        city: The name of the target city.
    """
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            desc = data['weather'][0]['description']

            return f"Current weather in {city}: {temp}°C, Humidity: {humidity}%, Condition: {desc}."
    except requests.RequestException as e:
        # requests' error text includes the full URL, and the key is in it (appid=...).
        # Never return it; log a redacted copy only.
        logging.error(f"OpenWeather request failed for {city!r}: {type(e).__name__}: {redact_secrets(str(e))}")
        return WEATHER_UNAVAILABLE_MESSAGE

    return f"Error: Unable to fetch weather data for {city}."
