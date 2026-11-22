"""Regression tests for the tool-result overwrite bug, the tool-loop cap and the
500 response body. No Gemini, MCP subprocess or network calls: the MCP session
and the Gemini client are faked."""
import logging
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from google.genai import types

from src.agent.graph import compiled_agent
from src.agent.nodes import MAX_TOOL_ROUNDS, TOOL_LIMIT_MESSAGE, execute_tools_node
from src.api.rate_limit import RateLimiter


# --- helpers -----------------------------------------------------------------

def model_calls(*calls: tuple[str, dict]) -> types.Content:
    """A model turn that asks for the given (tool name, args) calls."""
    return types.Content(role="model", parts=[
        types.Part.from_function_call(name=name, args=args) for name, args in calls
    ])


def model_text(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part.from_text(text=text)])


def gemini_response(content: types.Content) -> MagicMock:
    response = MagicMock()
    response.candidates = [MagicMock(content=content)]
    return response


def fake_tool_results(calls):
    return [{"name": name, "result": f"result for {args}", "duration_s": 0.0} for name, args in calls]


def run_turn(question: str, thread_id: str) -> list[types.Content]:
    user = types.Content(role="user", parts=[types.Part.from_text(text=question)])
    result = compiled_agent.invoke(
        {"messages": [user]}, config={"configurable": {"thread_id": thread_id}}
    )
    return result["messages"]


def assert_every_call_has_a_response(messages):
    """Gemini rejects a history where a function call isn't followed by its responses."""
    for i, message in enumerate(messages):
        calls = [p.function_call for p in (message.parts or []) if p.function_call]
        if calls:
            nxt = messages[i + 1]
            responses = [p.function_response for p in (nxt.parts or []) if p.function_response]
            assert nxt.role == "user"
            assert [r.name for r in responses] == [c.name for c in calls]


# --- 1. same tool twice in one turn -------------------------------------------

class FakeMcpSession:
    def __init__(self):
        self.calls = []

    async def initialize(self):
        pass

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return SimpleNamespace(content=[SimpleNamespace(text=f"Current weather in {arguments['city']}")])


def test_two_calls_to_the_same_tool_get_two_different_responses():
    session = FakeMcpSession()

    @asynccontextmanager
    async def fake_stdio_client(params):
        yield ("read", "write")

    @asynccontextmanager
    async def fake_client_session(read, write):
        yield session

    message = model_calls(
        ("get_live_weather_api", {"city": "London"}),
        ("get_live_weather_api", {"city": "New Delhi"}),
    )
    with patch("src.mcp_servers.mcp_client_helper.stdio_client", fake_stdio_client), \
         patch("src.mcp_servers.mcp_client_helper.ClientSession", fake_client_session):
        out = execute_tools_node({"messages": [message]})["messages"][0]

    results = [part.function_response.response["result"] for part in out.parts]
    assert results == ["Current weather in London", "Current weather in New Delhi"]
    assert [part.function_response.name for part in out.parts] == ["get_live_weather_api"] * 2
    assert session.calls == [
        ("get_live_weather", {"city": "London"}),
        ("get_live_weather", {"city": "New Delhi"}),
    ]


@patch("src.agent.nodes.call_tools_via_mcp_sync", side_effect=fake_tool_results)
def test_call_id_is_echoed_only_when_set(_mock_tools):
    with_id = types.Part(function_call=types.FunctionCall(id="call-1", name="get_live_weather_api", args={"city": "Oslo"}))
    without_id = types.Part(function_call=types.FunctionCall(name="get_live_weather_api", args={"city": "Lima"}))
    message = types.Content(role="model", parts=[with_id, without_id])

    out = execute_tools_node({"messages": [message]})["messages"][0]

    assert [part.function_response.id for part in out.parts] == ["call-1", None]


# --- 2. tool-loop cap ------------------------------------------------------------

@patch("src.agent.nodes.call_tools_via_mcp_sync", side_effect=fake_tool_results)
@patch("src.agent.nodes.genai_client")
def test_identical_repeated_batch_stops_early(mock_genai, mock_tools):
    # The production failure: the model asks for the same pair over and over.
    same_pair = model_calls(
        ("get_live_weather_api", {"city": "London"}),
        ("get_live_weather_api", {"city": "New Delhi"}),
    )
    mock_genai.models.generate_content.return_value = gemini_response(same_pair)

    messages = run_turn("Compare the weather in London and New Delhi.", str(uuid.uuid4()))

    assert messages[-1].parts[0].text == TOOL_LIMIT_MESSAGE
    assert mock_genai.models.generate_content.call_count == 2  # asked twice, second time is the repeat
    assert mock_tools.call_count == 1                          # the repeat is never executed
    assert_every_call_has_a_response(messages)


@patch("src.agent.nodes.call_tools_via_mcp_sync", side_effect=fake_tool_results)
@patch("src.agent.nodes.genai_client")
def test_tool_rounds_are_capped(mock_genai, mock_tools):
    cities = ["Tokyo", "Paris", "Rome", "Oslo", "Lima", "Cairo"]
    mock_genai.models.generate_content.side_effect = [
        gemini_response(model_calls(("get_live_weather_api", {"city": c}))) for c in cities
    ]

    messages = run_turn("Weather everywhere please.", str(uuid.uuid4()))

    assert messages[-1].parts[0].text == TOOL_LIMIT_MESSAGE
    assert mock_tools.call_count == MAX_TOOL_ROUNDS                          # 3 rounds run
    assert mock_genai.models.generate_content.call_count == MAX_TOOL_ROUNDS + 1  # 4th request is refused
    assert_every_call_has_a_response(messages)


@patch("src.agent.nodes.call_tools_via_mcp_sync", side_effect=fake_tool_results)
@patch("src.agent.nodes.genai_client")
def test_cap_is_per_question_and_normal_answers_are_untouched(mock_genai, mock_tools):
    thread_id = str(uuid.uuid4())
    stuck = model_calls(("get_live_weather_api", {"city": "London"}))
    mock_genai.models.generate_content.side_effect = [
        gemini_response(stuck), gemini_response(stuck),                 # question 1: stuck, stopped
        gemini_response(model_calls(("get_live_weather_api", {"city": "Tokyo"}))),
        gemini_response(model_text("It is 10°C in Tokyo.")),             # question 2: normal
    ]

    first = run_turn("London?", thread_id)
    second = run_turn("And Tokyo?", thread_id)

    assert first[-1].parts[0].text == TOOL_LIMIT_MESSAGE
    assert second[-1].parts[0].text == "It is 10°C in Tokyo."
    assert mock_tools.call_count == 2  # one round per question; question 2 is not counted against question 1
    assert_every_call_has_a_response(second)


# --- 3. 500 body ---------------------------------------------------------------

@patch("server.chat_rate_limiter", RateLimiter(max_requests=100, window_seconds=60))
@patch("server.check_input", new_callable=AsyncMock, return_value=(True, None))
@patch("server.compiled_agent")
def test_500_body_does_not_contain_the_exception_text(mock_agent, _mock_check_input, caplog):
    import server

    secret = "429 RESOURCE_EXHAUSTED quota detail at /src/agent/nodes.py"
    mock_agent.ainvoke = AsyncMock(side_effect=RuntimeError(secret))
    client = TestClient(server.app)

    with caplog.at_level(logging.ERROR):
        response = client.post("/chat", json={"message": "What is the weather in Tokyo?"})

    assert response.status_code == 500
    assert response.json() == {"detail": server.AGENT_ERROR_MESSAGE}
    for fragment in ("RESOURCE_EXHAUSTED", "RuntimeError", "/src/", "Agent error"):
        assert fragment not in response.text
    assert secret in caplog.text  # the real exception is still logged server-side
