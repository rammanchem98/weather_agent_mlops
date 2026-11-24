"""/chat trace and /eval/summary through the real server, guardrail, graph and
MCP helper. No Gemini, MCP subprocess or network: the clients are faked."""
import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai import errors, types

import server
import src.agent.nodes as nodes
from src.agent.nodes import MAX_TOOL_ROUNDS
from src.api.rate_limit import RateLimiter

OW_KEY = "SECRET123"
GEMINI_KEY = "PLANTED-GEMINI-KEY-42"
TRACE_KEYS = {"guardrail", "tool_calls", "rounds", "capped", "gemini_calls", "total_latency_s"}


# --- fakes ---------------------------------------------------------------------

def classifier(text=None, raises=None):
    async def fake(**kwargs):
        if raises:
            raise raises
        return MagicMock(text=text)
    client = MagicMock()
    client.aio.models.generate_content = fake
    return client


SAFE = '{"is_injection": false, "confidence": "low", "reason": "weather question"}'


def gemini(*parts):
    response = MagicMock()
    response.candidates = [MagicMock(content=types.Content(role="model", parts=list(parts)))]
    return response


def weather_call(city):
    return types.Part.from_function_call(name="get_live_weather_api", args={"city": city})


class FakeSession:
    tool_text = "Current weather in {city}: 10°C, Humidity: 50%, Condition: clear."

    async def initialize(self):
        pass

    async def call_tool(self, name, arguments):
        await asyncio.sleep(0.01)
        return SimpleNamespace(content=[SimpleNamespace(text=self.tool_text.format(**arguments))])


def mcp_patches(session_cls=FakeSession):
    @asynccontextmanager
    async def fake_stdio_client(params):
        yield ("read", "write")

    @asynccontextmanager
    async def fake_client_session(read, write):
        yield session_cls()

    return (patch("src.mcp_servers.mcp_client_helper.stdio_client", fake_stdio_client),
            patch("src.mcp_servers.mcp_client_helper.ClientSession", fake_client_session))


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(server, "chat_rate_limiter", RateLimiter(max_requests=100, window_seconds=60))
    monkeypatch.setattr(nodes, "_include_thoughts", True)


def chat(message, model_side_effect, guard=classifier(SAFE), session_cls=FakeSession):
    stdio, session = mcp_patches(session_cls)
    with patch("src.api.guardrails.genai_client", guard), patch("src.agent.nodes.genai_client") as agent, \
         stdio, session:
        agent.models.generate_content.side_effect = model_side_effect
        response = TestClient(server.app).post("/chat", json={"message": message})
    return response, agent


# --- trace shape ----------------------------------------------------------------

def test_normal_turn_trace():
    response, _ = chat("Compare London and New Delhi", [
        gemini(types.Part(text="I need live weather for both cities.", thought=True),
               weather_call("London"), weather_call("New Delhi")),
        gemini(types.Part.from_text(text="London 10°C, New Delhi 10°C.")),
    ])
    body = response.json()

    assert response.status_code == 200
    assert set(body) == {"response", "thread_id", "trace"}          # existing fields unchanged
    assert body["response"] == "London 10°C, New Delhi 10°C."        # no thought text in the answer
    trace = body["trace"]
    assert set(trace) == TRACE_KEYS | {"plan"}
    assert trace["guardrail"] == {"regex": "pass", "classifier": "passed"}
    assert trace["plan"] == "I need live weather for both cities."
    assert [(c["name"], c["args"]) for c in trace["tool_calls"]] == [
        ("get_live_weather_api", {"city": "London"}), ("get_live_weather_api", {"city": "New Delhi"})]
    assert "London" in trace["tool_calls"][0]["result_preview"]
    assert "New Delhi" in trace["tool_calls"][1]["result_preview"]
    assert all(isinstance(c["duration_s"], float) and c["duration_s"] > 0 for c in trace["tool_calls"])
    assert trace["rounds"] == 1 and trace["capped"] is False
    assert trace["gemini_calls"] == 3                                 # classifier + 2 agent steps
    assert trace["total_latency_s"] > 0


def test_no_thought_summary_means_no_plan_field():
    response, _ = chat("Weather in Tokyo?", [gemini(weather_call("Tokyo")),
                                             gemini(types.Part.from_text(text="10°C."))])
    trace = response.json()["trace"]
    assert "plan" not in trace and set(trace) == TRACE_KEYS


def test_capped_turn_trace():
    cities = iter(["Tokyo", "Paris", "Rome", "Oslo", "Lima"])
    response, _ = chat("Weather everywhere", lambda **kw: gemini(weather_call(next(cities))))
    trace = response.json()["trace"]

    assert response.status_code == 200
    assert trace["capped"] is True and trace["rounds"] == MAX_TOOL_ROUNDS
    assert [c["args"]["city"] for c in trace["tool_calls"]] == ["Tokyo", "Paris", "Rome"]  # the 4th never ran
    assert trace["gemini_calls"] == 1 + MAX_TOOL_ROUNDS + 1


def test_classifier_unavailable_trace_counts_the_failed_call():
    quota = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota"}})
    response, _ = chat("Weather in Tokyo?", [gemini(types.Part.from_text(text="10°C."))],
                       guard=classifier(raises=quota))
    trace = response.json()["trace"]

    assert response.status_code == 200
    assert trace["guardrail"] == {"regex": "pass", "classifier": "unavailable"}
    assert trace["gemini_calls"] == 2                                 # failed classifier + 1 agent step


def test_planted_keys_never_appear_in_the_trace(monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", OW_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)

    class LeakySession(FakeSession):
        tool_text = (f"Error: https://api.openweathermap.org/data/2.5/weather?q={{city}}&appid={OW_KEY} "
                     f"used key {GEMINI_KEY}")

    response, _ = chat("Weather in London?", [
        gemini(types.Part(text=f"Thinking about {GEMINI_KEY} and {OW_KEY}", thought=True),
               types.Part.from_function_call(name="get_live_weather_api", args={"city": f"London {OW_KEY}"})),
        gemini(types.Part.from_text(text="Done.")),
    ], session_cls=LeakySession)

    raw = response.text
    assert response.status_code == 200
    assert OW_KEY not in raw and GEMINI_KEY not in raw
    assert "[REDACTED]" in json.dumps(response.json()["trace"])


# --- concurrency -----------------------------------------------------------------

def test_overlapping_requests_get_their_own_traces():
    both_in_flight = threading.Barrier(2, timeout=20)

    def model(**kwargs):
        contents = kwargs["contents"]
        city = contents[0].parts[0].text
        if len(contents) == 1:
            both_in_flight.wait()                 # both requests are inside the agent at once
            return gemini(weather_call(city))
        return gemini(types.Part.from_text(text=f"done {city}"))

    async def both():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await asyncio.gather(*(client.post("/chat", json={"message": c}) for c in ("London", "New Delhi")))

    stdio, session = mcp_patches()
    with patch("src.api.guardrails.genai_client", classifier(SAFE)), patch("src.agent.nodes.genai_client") as agent, \
         stdio, session:
        agent.models.generate_content.side_effect = model
        london, delhi = asyncio.run(both())

    for response, city in ((london, "London"), (delhi, "New Delhi")):
        trace = response.json()["trace"]
        assert [c["args"]["city"] for c in trace["tool_calls"]] == [city]
        assert trace["gemini_calls"] == 3 and trace["rounds"] == 1


# --- /eval/summary ---------------------------------------------------------------

@pytest.fixture
def eval_file(tmp_path, monkeypatch):
    path = tmp_path / "results.json"
    monkeypatch.setattr(server, "EVAL_RESULTS_PATH", str(path))
    return path


def test_eval_summary_absent_is_404(eval_file):
    response = TestClient(server.app).get("/eval/summary")
    assert response.status_code == 404 and response.json() == {"detail": "No eval results."}


@pytest.mark.parametrize("content", [
    "not json", "[]", '{"passed": 2, "total": 3}', '{"run_at": "yesterday", "passed": 2, "total": 3}',
    '{"run_at": "2026-11-20", "passed": "2", "total": 3}', '{"run_at": "2026-11-20", "passed": true, "total": 3}',
    '{"run_at": "2026-11-20", "passed": 4, "total": 3}', '{"run_at": "2026-11-20", "passed": 0, "total": 0}',
])
def test_eval_summary_malformed_is_404_without_details(eval_file, content):
    eval_file.write_text(content, encoding="utf-8")
    response = TestClient(server.app).get("/eval/summary")
    assert response.status_code == 404
    assert response.json() == {"detail": "No eval results."}
    assert str(eval_file) not in response.text and "results.json" not in response.text


def test_eval_summary_returns_only_the_three_fields(eval_file):
    eval_file.write_text(json.dumps({"run_at": "2026-11-20T10:00:00Z", "passed": 2, "total": 3,
                                     "results": [{"query": "secret internals"}]}), encoding="utf-8")
    response = TestClient(server.app).get("/eval/summary")
    assert response.status_code == 200
    assert response.json() == {"run_at": "2026-11-20T10:00:00Z", "passed": 2, "total": 3}
