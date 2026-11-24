"""TraceRecorder wiring through the graph. No Gemini, MCP or network calls."""
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors, types

import src.agent.nodes as nodes
from src.agent.graph import compiled_agent
from src.agent.trace import PLAN_MAX_CHARS, RESULT_PREVIEW_CHARS, TRACE_CONFIG_KEY, TraceRecorder, build_trace


def gemini(*parts):
    response = MagicMock()
    response.candidates = [MagicMock(content=types.Content(role="model", parts=list(parts)))]
    return response


def thought(text):
    return types.Part(text=text, thought=True)


def call(city, signature=None):
    part = types.Part.from_function_call(name="get_live_weather_api", args={"city": city})
    if signature:
        part.thought_signature = signature
    return part


def text(value):
    return types.Part.from_text(text=value)


def fake_tools(calls):
    return [{"name": n, "result": f"Current weather in {a['city']}: 10°C.", "duration_s": 0.25} for n, a in calls]


def run(question, recorder=None, thread_id=None):
    configurable = {"thread_id": thread_id or str(uuid.uuid4())}
    if recorder is not None:
        configurable[TRACE_CONFIG_KEY] = recorder
    user = types.Content(role="user", parts=[types.Part.from_text(text=question)])
    return compiled_agent.invoke({"messages": [user]}, config={"configurable": configurable})["messages"]


@pytest.fixture(autouse=True)
def thoughts_flag_on(monkeypatch):
    monkeypatch.setattr(nodes, "_include_thoughts", True)


@patch("src.agent.nodes.call_tools_via_mcp_sync", side_effect=fake_tools)
@patch("src.agent.nodes.genai_client")
def test_recorder_captures_a_normal_tool_turn(mock_genai, _tools):
    mock_genai.models.generate_content.side_effect = [
        gemini(thought("I need live weather for London."), call("London", b"sig-1")),
        gemini(thought("Got it, answering."), text("It is 10°C in London.")),
    ]
    recorder = TraceRecorder()

    messages = run("Weather in London?", recorder)

    assert recorder.gemini_calls == 2
    assert recorder.rounds == 1 and recorder.capped is False
    assert recorder.tool_calls == [{"name": "get_live_weather_api", "args": {"city": "London"},
                                    "result": "Current weather in London: 10°C.", "duration_s": 0.25}]
    assert recorder.thoughts == ["I need live weather for London.", "Got it, answering."]
    # thought text is not in the stored history or the answer; the call keeps its signature
    assert not any(p.thought for m in messages for p in (m.parts or []))
    assert messages[-1].parts[0].text == "It is 10°C in London."
    assert messages[1].parts[0].thought_signature == b"sig-1"
    # the request asked for thought summaries
    sent = mock_genai.models.generate_content.call_args_list[0].kwargs["config"]
    assert sent.thinking_config.include_thoughts is True


@patch("src.agent.nodes.call_tools_via_mcp_sync", side_effect=fake_tools)
@patch("src.agent.nodes.genai_client")
def test_two_turn_tool_conversation_with_thoughts_and_memory_saver(mock_genai, _tools):
    mock_genai.models.generate_content.side_effect = [
        gemini(thought("plan A"), call("London", b"sig-1")),
        gemini(thought("done A"), text("London is 10°C.")),
        gemini(thought("plan B"), call("Tokyo", b"sig-2")),
        gemini(text("Tokyo is 10°C.")),
    ]
    thread = str(uuid.uuid4())
    first, second = TraceRecorder(), TraceRecorder()

    run("London?", first, thread)
    messages = run("And Tokyo?", second, thread)          # MemorySaver replays turn 1

    assert messages[-1].parts[0].text == "Tokyo is 10°C."
    turn2_request = mock_genai.models.generate_content.call_args_list[2].kwargs["contents"]
    assert len(turn2_request) == 5                         # q1, call, result, answer, q2
    assert not any(p.thought for m in turn2_request for p in (m.parts or []))
    assert turn2_request[1].parts[0].thought_signature == b"sig-1"
    # each turn's recorder holds only its own turn
    assert [c["args"]["city"] for c in first.tool_calls] == ["London"]
    assert [c["args"]["city"] for c in second.tool_calls] == ["Tokyo"]
    assert second.thoughts == ["plan B"] and second.gemini_calls == 2


@patch("src.agent.nodes.call_tools_via_mcp_sync", side_effect=fake_tools)
@patch("src.agent.nodes.genai_client")
def test_recorder_marks_a_capped_turn(mock_genai, _tools):
    cities = iter(["Tokyo", "Paris", "Rome", "Oslo"])
    mock_genai.models.generate_content.side_effect = lambda **kw: gemini(call(next(cities)))
    recorder = TraceRecorder()

    run("Weather everywhere", recorder)

    assert recorder.capped is True
    assert recorder.rounds == nodes.MAX_TOOL_ROUNDS
    assert recorder.gemini_calls == nodes.MAX_TOOL_ROUNDS + 1
    assert recorder.thoughts == []


@patch("src.agent.nodes.call_tools_via_mcp_sync", side_effect=fake_tools)
@patch("src.agent.nodes.genai_client")
def test_rejected_include_thoughts_falls_back_silently(mock_genai, _tools):
    rejected = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                                  "message": "Thinking is not supported for this model."}})
    mock_genai.models.generate_content.side_effect = [rejected, gemini(text("Sunny."))]
    recorder = TraceRecorder()

    messages = run("Weather?", recorder)

    assert messages[-1].parts[0].text == "Sunny."
    assert recorder.gemini_calls == 2                      # the failed attempt is counted
    assert recorder.thoughts == []
    retry_config = mock_genai.models.generate_content.call_args_list[1].kwargs["config"]
    assert retry_config.thinking_config is None


@patch("src.agent.nodes.genai_client")
def test_other_client_errors_still_raise(mock_genai):
    mock_genai.models.generate_content.side_effect = errors.ClientError(
        429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota"}})
    recorder = TraceRecorder()

    with pytest.raises(errors.ClientError):
        run("Weather?", recorder)
    assert recorder.gemini_calls == 1


@patch("src.agent.nodes.call_tools_via_mcp_sync", side_effect=fake_tools)
@patch("src.agent.nodes.genai_client")
def test_overlapping_runs_never_share_a_recorder(mock_genai, _tools):
    both_in_flight = threading.Barrier(2, timeout=20)

    def model(**kwargs):
        contents = kwargs["contents"]
        question = contents[0].parts[0].text
        if len(contents) == 1:            # first call of each run: wait until both are inside
            both_in_flight.wait()
            return gemini(call(question))
        return gemini(text(f"done {question}"))

    mock_genai.models.generate_content.side_effect = model
    recorders = {"London": TraceRecorder(), "New Delhi": TraceRecorder()}
    threads = [threading.Thread(target=run, args=(city, rec)) for city, rec in recorders.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    for city, rec in recorders.items():
        assert [c["args"]["city"] for c in rec.tool_calls] == [city]
        assert rec.gemini_calls == 2 and rec.rounds == 1


def test_build_trace_sanitizes_truncates_and_omits_empty_plan():
    recorder = TraceRecorder(guardrail={"regex": "pass", "classifier": "passed"})
    recorder.add_tool_round([{"name": "t", "args": {"q": "SECRET-x", "n": 2},
                              "result": "SECRET-y " + "z" * 1000, "duration_s": 0.123456}])
    trace = build_trace(recorder, 1.23456, sanitize=lambda s: s.replace("SECRET", "[REDACTED]"))

    assert set(trace) == {"guardrail", "tool_calls", "rounds", "capped", "gemini_calls", "total_latency_s"}
    call = trace["tool_calls"][0]
    assert call["args"] == {"q": "[REDACTED]-x", "n": 2}
    assert call["result_preview"].startswith("[REDACTED]-y") and len(call["result_preview"]) == RESULT_PREVIEW_CHARS
    assert call["duration_s"] == 0.123 and trace["total_latency_s"] == 1.235

    recorder.add_thought("SECRET plan")
    assert build_trace(recorder, 0, sanitize=lambda s: s.replace("SECRET", "[REDACTED]"))["plan"] == "[REDACTED] plan"


def test_plan_is_capped_at_2000_chars_with_an_ellipsis_after_sanitizing():
    recorder = TraceRecorder()
    recorder.add_thought("SECRET " + "x" * 5000)
    plan = build_trace(recorder, 0, sanitize=lambda s: s.replace("SECRET", "[REDACTED]"))["plan"]

    assert len(plan) <= PLAN_MAX_CHARS and plan.endswith("…")
    assert plan.startswith("[REDACTED] ")

    short = TraceRecorder()
    short.add_thought("A short plan.")
    assert build_trace(short, 0, sanitize=str)["plan"] == "A short plan."
