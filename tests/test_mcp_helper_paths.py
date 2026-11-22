"""Contract: call_tools_via_mcp_sync returns one {name, result, duration_s} per
call, in call order, on EVERY path, including when the MCP process can't start
or the session fails. No subprocess, Gemini or network: the MCP client is faked."""
import asyncio
import logging
import re
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from src.agent.graph import compiled_agent
from src.mcp_servers.mcp_client_helper import TOOL_UNAVAILABLE_MESSAGE, call_tools_via_mcp_sync

CALLS = [
    ("get_live_weather_api", {"city": "London"}),
    ("get_live_weather_api", {"city": "New Delhi"}),
]


def assert_contract(results):
    assert isinstance(results, list)
    assert len(results) == len(CALLS)
    for (name, _), entry in zip(CALLS, results):
        assert set(entry) == {"name", "result", "duration_s"}
        assert entry["name"] == name
        assert isinstance(entry["result"], str) and entry["result"]
        assert isinstance(entry["duration_s"], float)


class OkSession:
    async def initialize(self):
        pass

    async def call_tool(self, name, arguments):
        return SimpleNamespace(content=[SimpleNamespace(text=f"Current weather in {arguments['city']}")])


def test_process_fails_to_start_returns_a_result_per_call():
    @asynccontextmanager
    async def failing_stdio_client(params):
        raise FileNotFoundError("[WinError 2] The system cannot find the file specified")
        yield  # pragma: no cover

    with patch("src.mcp_servers.mcp_client_helper.stdio_client", failing_stdio_client):
        results = call_tools_via_mcp_sync(CALLS)

    assert_contract(results)


def test_session_initialize_error_returns_a_result_per_call():
    class BrokenSession(OkSession):
        async def initialize(self):
            raise ExceptionGroup("unhandled errors in a TaskGroup", [ConnectionError("server exited")])

    @asynccontextmanager
    async def fake_stdio_client(params):
        yield ("read", "write")

    @asynccontextmanager
    async def fake_client_session(read, write):
        yield BrokenSession()

    with patch("src.mcp_servers.mcp_client_helper.stdio_client", fake_stdio_client), \
         patch("src.mcp_servers.mcp_client_helper.ClientSession", fake_client_session):
        results = call_tools_via_mcp_sync(CALLS)

    assert_contract(results)


def test_session_error_on_close_keeps_completed_results():
    @asynccontextmanager
    async def fake_stdio_client(params):
        yield ("read", "write")
        raise ConnectionResetError("pipe closed during shutdown")

    @asynccontextmanager
    async def fake_client_session(read, write):
        yield OkSession()

    with patch("src.mcp_servers.mcp_client_helper.stdio_client", fake_stdio_client), \
         patch("src.mcp_servers.mcp_client_helper.ClientSession", fake_client_session):
        results = call_tools_via_mcp_sync(CALLS)

    assert_contract(results)
    assert [r["result"] for r in results] == ["Current weather in London", "Current weather in New Delhi"]


def test_tool_error_inside_server_returns_a_result_per_call():
    class ErroringSession(OkSession):
        async def call_tool(self, name, arguments):
            if arguments["city"] == "London":
                raise RuntimeError("tool crashed")
            return await super().call_tool(name, arguments)

    @asynccontextmanager
    async def fake_stdio_client(params):
        yield ("read", "write")

    @asynccontextmanager
    async def fake_client_session(read, write):
        yield ErroringSession()

    with patch("src.mcp_servers.mcp_client_helper.stdio_client", fake_stdio_client), \
         patch("src.mcp_servers.mcp_client_helper.ClientSession", fake_client_session):
        results = call_tools_via_mcp_sync(CALLS)

    assert_contract(results)
    assert results[1]["result"] == "Current weather in New Delhi"  # order kept, the other call unaffected


# --- fallback text, logging, cancellation --------------------------------------

@asynccontextmanager
async def _failing_stdio_client(params):
    raise FileNotFoundError("[WinError 2] The system cannot find the file specified")
    yield  # pragma: no cover


def test_failure_uses_the_no_invention_message_and_is_logged(caplog):
    with patch("src.mcp_servers.mcp_client_helper.stdio_client", _failing_stdio_client), \
         caplog.at_level(logging.ERROR):
        results = call_tools_via_mcp_sync(CALLS)

    assert [r["result"] for r in results] == [TOOL_UNAVAILABLE_MESSAGE] * 2
    record = next(r for r in caplog.records if "MCP tool session failed" in r.getMessage())
    assert record.exc_info is not None              # logged with logging.exception (traceback attached)
    assert "FileNotFoundError" in caplog.text


def test_cancellation_during_startup_propagates():
    @asynccontextmanager
    async def cancelled_stdio_client(params):
        raise asyncio.CancelledError()
        yield  # pragma: no cover

    with patch("src.mcp_servers.mcp_client_helper.stdio_client", cancelled_stdio_client):
        with pytest.raises(asyncio.CancelledError):
            call_tools_via_mcp_sync(CALLS)


def test_cancellation_during_a_call_propagates():
    class CancelledSession(OkSession):
        async def call_tool(self, name, arguments):
            raise asyncio.CancelledError()

    @asynccontextmanager
    async def fake_stdio_client(params):
        yield ("read", "write")

    @asynccontextmanager
    async def fake_client_session(read, write):
        yield CancelledSession()

    with patch("src.mcp_servers.mcp_client_helper.stdio_client", fake_stdio_client), \
         patch("src.mcp_servers.mcp_client_helper.ClientSession", fake_client_session):
        with pytest.raises(asyncio.CancelledError):
            call_tools_via_mcp_sync(CALLS)


# --- a tool that always fails, through the whole agent ------------------------

def _gemini(content):
    response = MagicMock()
    response.candidates = [MagicMock(content=content)]
    return response


def _ask_weather(city):
    return types.Content(role="model", parts=[
        types.Part.from_function_call(name="get_live_weather_api", args={"city": city})
    ])


def _run(question):
    user = types.Content(role="user", parts=[types.Part.from_text(text=question)])
    out = compiled_agent.invoke({"messages": [user]}, config={"configurable": {"thread_id": str(uuid.uuid4())}})
    return out["messages"]


@patch("src.mcp_servers.mcp_client_helper.stdio_client", _failing_stdio_client)
@patch("src.agent.nodes.genai_client")
def test_always_failing_tool_with_a_stubborn_model_ends_via_the_cap(mock_genai):
    # Worst case: the model ignores the message and keeps asking (different args each time,
    # so the repeat check can't help; only the round cap can).
    cities = iter(["London", "London, UK", "London GB", "Londres", "LDN"])
    mock_genai.models.generate_content.side_effect = lambda **kw: _gemini(_ask_weather(next(cities)))

    messages = _run("What's the weather in London?")
    final = messages[-1].parts[0].text

    assert mock_genai.models.generate_content.call_count <= 4
    assert not re.search(r"\d", final)                       # no invented temperatures/humidity
    seen = [p.function_response.response["result"] for m in messages for p in (m.parts or []) if p.function_response]
    assert TOOL_UNAVAILABLE_MESSAGE in seen                  # the instruction did reach the model


@patch("src.mcp_servers.mcp_client_helper.stdio_client", _failing_stdio_client)
@patch("src.agent.nodes.genai_client")
def test_always_failing_tool_with_an_obedient_model_gets_an_honest_answer(mock_genai):
    honest = "I couldn't reach the weather service, so I can't give London's current weather."

    def model(**kwargs):
        last = kwargs["contents"][-1]
        told = any(p.function_response and "Do not estimate" in p.function_response.response["result"]
                   for p in (last.parts or []))
        return _gemini(types.Content(role="model", parts=[types.Part.from_text(text=honest)]) if told
                       else _ask_weather("London"))

    mock_genai.models.generate_content.side_effect = model

    messages = _run("What's the weather in London?")

    assert messages[-1].parts[0].text == honest
    assert mock_genai.models.generate_content.call_count == 2
    assert not re.search(r"\d", messages[-1].parts[0].text)
