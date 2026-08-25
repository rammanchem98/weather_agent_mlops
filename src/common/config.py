import os
import yaml
from dotenv import load_dotenv

load_dotenv()

# src/common/config.py -> up 2 levels to reach the project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

_config_cache: dict[str, dict] = {}


def get_config(env: str | None = None) -> dict:
    """Loads config/config-{env}.yaml from the project root, caching per env
    so repeated calls don't re-read the file from disk."""
    env = env or os.getenv("env", "dev")

    if env not in _config_cache:
        config_path = os.path.join(ROOT_DIR, "config", f"config-{env}.yaml")
        with open(config_path, "r") as f:
            _config_cache[env] = yaml.safe_load(f)

    return _config_cache[env]
