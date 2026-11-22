"""API key values must never reach the model, the client or the logs, even when a
tool's error text contains them. No network: requests, MCP and Gemini are faked."""
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from fastapi.testclient import TestClient
from google.genai import types

from src.api.rate_limit import RateLimiter
from src.common.redact import REDACTED, redact_secrets
from src.tools import weather_api
from src.tools.weather_api import WEATHER_UNAVAILABLE_MESSAGE, get_live_weather_api

OW_KEY = "SECRET123"
GEMINI_KEY = "GEMINI-KEY-VALUE-456"
LEAKY_URL = f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={OW_KEY}&units=metric"
CONNECTION_ERROR = requests.exceptions.ConnectionError(
    "HTTPSConnectionPool(host='api.openweathermap.org', port=443): Max retries exceeded with url: "
    f"/data/2.5/weather?q=London&appid={OW_KEY}&units=metric "
    "(Caused by NameResolutionError(\"Failed to resolve 'api.openweathermap.org'\"))"
)


@pytest.fixture
def secrets_env(monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", OW_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setattr(weather_api, "API_KEY", OW_KEY)  # module reads the key at import


def assert_no_secrets(text):
    assert OW_KEY not in text
    assert GEMINI_KEY not in text


# --- redaction helper ------------------------------------------------------------

def test_redact_secrets_removes_env_key_values_and_appid_params(secrets_env):
    text = f"{LEAKY_URL} and appid=OTHERVALUE&x=1 and key {GEMINI_KEY}"
    out = redact_secrets(text)
    assert_no_secrets(out)
    assert "OTHERVALUE" not in out
    assert out.count(REDACTED) >= 3


def test_redact_secrets_ignores_short_placeholder_values(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert redact_secrets("max temp") == "max temp"   # a 1-char "key" must not mangle text


# --- weather tool ------------------------------------------------------------------

@patch("src.tools.weather_api.requests.get", side_effect=CONNECTION_ERROR)
def test_weather_tool_connection_error_returns_generic_message(mock_get, secrets_env, caplog):
    with caplog.at_level(logging.INFO):
        result = get_live_weather_api("London")

    assert OW_KEY in mock_get.call_args.args[0]              # the request really carried the key
    assert result == WEATHER_UNAVAILABLE_MESSAGE
    assert_no_secrets(result)
    assert_no_secrets(caplog.text)
    assert "OpenWeather request failed" in caplog.text       # the real error is still logged...
    assert f"appid={REDACTED}" in caplog.text                # ...with the key redacted


# --- end to end: tool -> helper -> model -> /chat ----------------------------------

class InProcessToolSession:
    """Fake MCP session that runs the real tool functions in-process; anything a tool
    raises is turned into error text the way the MCP server does it."""

    async def initialize(self):
        pass

    async def call_tool(self, name, arguments):
        try:
            if name == "get_live_weather":
                text = get_live_weather_api(arguments["city"])
            else:
                raise RuntimeError(f"tool failed calling {LEAKY_URL} with key {GEMINI_KEY}")
        except Exception as e:
            text = f"Error executing tool {name}: {e}"
        return SimpleNamespace(content=[SimpleNamespace(text=text)])


def _echoing_model(tool_name, args):
    """Worst-case model: requests one tool, then repeats the tool result verbatim."""
    def model(**kwargs):
        last = kwargs["contents"][-1]
        results = [p.function_response.response["result"] for p in (last.parts or []) if p.function_response]
        response = MagicMock()
        if results:
            content = types.Content(role="model", parts=[types.Part.from_text(text=f"Tool said: {results[0]}")])
        else:
            content = types.Content(role="model", parts=[types.Part.from_function_call(name=tool_name, args=args)])
        response.candidates = [MagicMock(content=content)]
        return response
    return model


def _chat(message):
    import server

    @asynccontextmanager
    async def fake_stdio_client(params):
        yield ("read", "write")

    @asynccontextmanager
    async def fake_client_session(read, write):
        yield InProcessToolSession()

    with patch("server.chat_rate_limiter", RateLimiter(max_requests=100, window_seconds=60)), \
         patch("server.check_input", new_callable=AsyncMock, return_value=(True, None)), \
         patch("src.mcp_servers.mcp_client_helper.stdio_client", fake_stdio_client), \
         patch("src.mcp_servers.mcp_client_helper.ClientSession", fake_client_session):
        return TestClient(server.app).post("/chat", json={"message": message})


@patch("src.tools.weather_api.requests.get", side_effect=CONNECTION_ERROR)
@patch("src.agent.nodes.genai_client")
def test_connection_error_with_key_never_reaches_model_client_or_logs(mock_genai, mock_get, secrets_env, caplog):
    mock_genai.models.generate_content.side_effect = _echoing_model("get_live_weather_api", {"city": "London"})

    with caplog.at_level(logging.INFO):
        response = _chat("What's the weather in London?")

    assert response.status_code == 200
    assert OW_KEY in mock_get.call_args.args[0]
    sent_to_model = str(mock_genai.models.generate_content.call_args_list[-1].kwargs["contents"])
    assert_no_secrets(sent_to_model)                          # tool result as the model saw it
    assert WEATHER_UNAVAILABLE_MESSAGE in response.json()["response"]
    assert_no_secrets(response.text)                          # final response
    assert_no_secrets(caplog.text)                            # captured logs


@patch("src.agent.nodes.genai_client")
def test_raw_tool_error_text_with_keys_is_redacted_before_the_model(mock_genai, secrets_env, caplog):
    # A tool without its own handling (MCP wraps the exception text, URL and key included).
    mock_genai.models.generate_content.side_effect = _echoing_model(
        "search_air_quality_db", {"query": "London air quality"})

    with caplog.at_level(logging.INFO):
        response = _chat("Historical air quality in London?")

    sent_to_model = str(mock_genai.models.generate_content.call_args_list[-1].kwargs["contents"])
    assert "Error executing tool" in sent_to_model            # the error did go through...
    assert_no_secrets(sent_to_model)                          # ...redacted
    assert_no_secrets(response.text)
    assert_no_secrets(caplog.text)


@patch("server.chat_rate_limiter", RateLimiter(max_requests=100, window_seconds=60))
@patch("server.check_input", new_callable=AsyncMock, return_value=(True, None))
@patch("server.compiled_agent")
def test_logged_agent_exception_is_redacted(mock_agent, _check, secrets_env, caplog):
    import server
    mock_agent.ainvoke = AsyncMock(side_effect=RuntimeError(f"failed: {LEAKY_URL} {GEMINI_KEY}"))

    with caplog.at_level(logging.ERROR):
        response = TestClient(server.app).post("/chat", json={"message": "Tokyo?"})

    assert response.status_code == 500
    assert "Agent invocation failed" in caplog.text
    assert "RuntimeError" in caplog.text                     # traceback still logged...
    assert_no_secrets(caplog.text)                           # ...without the key values
    assert_no_secrets(response.text)
