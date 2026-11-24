import json
import logging
from google import genai
from google.genai import errors, types
from google.genai.types import AutomaticFunctionCallingConfig
from langchain_core.runnables import RunnableConfig

from src.agent.state import AgentState
from src.agent.trace import get_recorder
from src.mcp_servers.mcp_client_helper import call_tools_via_mcp_sync
from src.tools.weather_api import get_live_weather_api
from src.tools.vector_search import search_air_quality_db


genai_client = genai.Client()

# Central tool registry
TOOL_REGISTRY = {
    "get_live_weather_api": get_live_weather_api,
    "search_air_quality_db": search_air_quality_db
}

AVAILABLE_TOOLS = list(TOOL_REGISTRY.values())


def get_function_calls(content: types.Content):
    """Extracts function_call parts from a Content message (Content has no
    .function_calls attribute — only GenerateContentResponse does)."""
    if not content or not content.parts:
        return []
    return [part.function_call for part in content.parts if part.function_call is not None]

def is_thought(part: types.Part) -> bool:
    """Thought-summary parts (returned because of include_thoughts) are for the
    trace only: never part of the answer, the logs or the stored history."""
    return bool(getattr(part, "thought", None))


# Set to False the first time the API rejects include_thoughts; later calls skip it.
_include_thoughts = True


def _generate(contents, recorder):
    """One agent generateContent call. Every attempt is counted, failed ones too.
    If the API rejects include_thoughts, retry once without it (no summary)."""
    global _include_thoughts
    while True:
        config = types.GenerateContentConfig(
            tools=AVAILABLE_TOOLS,
            temperature=0.0,
            automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(include_thoughts=True) if _include_thoughts else None,
        )
        if recorder:
            recorder.count_gemini_call()
        try:
            return genai_client.models.generate_content(
                model="gemini-2.5-flash", contents=contents, config=config,
            )
        except errors.ClientError as e:
            message = str(e).lower()
            if _include_thoughts and e.code == 400 and ("thought" in message or "thinking" in message):
                logging.warning("include_thoughts rejected by the API; continuing without reasoning summaries")
                _include_thoughts = False
                continue
            raise


def agent_brain(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """Reasoning Node: Passes conversation state to Gemini 2.5 Flash."""
    recorder = get_recorder(config)
    response = _generate(state["messages"], recorder)

    candidate = response.candidates[0].content
    parts = candidate.parts or []
    if recorder:
        for part in parts:
            if is_thought(part) and part.text:
                recorder.add_thought(part.text)

    # Keep thought text out of the thread history (it is replayed to Gemini on the
    # next question); function_call parts keep their thought_signature untouched.
    visible = [part for part in parts if not is_thought(part)]
    if visible and len(visible) != len(parts):
        candidate = types.Content(role=candidate.role, parts=visible)

    function_calls= get_function_calls(candidate)
    if function_calls:
        logging.info(f"Function calls detected:{[(f.name,f.args) for f in function_calls]}")
    else:
        text=",".join([part.text for part in (candidate.parts or []) if part.text and not is_thought(part)])
        logging.info(f"No function calls detected. Response: {text}")
    return {"messages": [candidate]}

def execute_tools_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """Action Node: Intercepts function_calls, runs them via MCP, and builds responses."""
    last_message = state["messages"][-1]
    tool_responses = []
    function_calls = get_function_calls(last_message)

    if function_calls:
        calls = [(call.name, call.args or {}) for call in function_calls]
        results = call_tools_via_mcp_sync(calls)  # ONE call, handles all tools in this turn

        # Results come back in the same order as the calls, so match by position.
        # (Matching by tool name made two calls to the same tool share one result.)
        recorded = []
        for i, call in enumerate(function_calls):
            if i < len(results):
                tool_output = results[i]["result"]
                duration_s = results[i].get("duration_s", 0.0)
            else:
                tool_output = f"Tool '{call.name}' returned no result."
                duration_s = 0.0
            tool_responses.append(_function_response_part(call, {"result": tool_output}))
            recorded.append({"name": call.name, "args": dict(call.args or {}),
                             "result": tool_output, "duration_s": duration_s})

        recorder = get_recorder(config)
        if recorder:
            recorder.add_tool_round(recorded)

    return {"messages": [types.Content(role="user", parts=tool_responses)]}


def _function_response_part(call: types.FunctionCall, response: dict) -> types.Part:
    """Builds the function_response for one call. The Gemini API often leaves
    call.id empty; when it is set, echo it back so the response is tied to its call."""
    return types.Part(function_response=types.FunctionResponse(
        id=call.id or None, name=call.name, response=response,
    ))


# --- Tool-loop protection ---------------------------------------------------

MAX_TOOL_ROUNDS = 3

TOOL_LIMIT_MESSAGE = (
    "Sorry, I couldn't finish answering that. I kept needing more tool calls "
    "without getting to an answer, so I stopped. Try asking about one city at a time."
)


def current_turn_tool_batches(messages) -> list[list[tuple[str, str]]]:
    """Tool-call batches the model has made since the user's latest question,
    oldest first. Each batch is a list of (tool name, args as sorted JSON).

    The checkpointer keeps the whole thread, so this walks back only to the most
    recent user message that contains text (tool results are also role="user",
    but carry function_response parts, not text)."""
    batches = []
    for message in reversed(messages):
        parts = message.parts or []
        if message.role == "user" and any(part.text for part in parts):
            break
        calls = get_function_calls(message)
        if calls:
            batches.append([
                (call.name, json.dumps(call.args or {}, sort_keys=True)) for call in calls
            ])
    batches.reverse()
    return batches


def should_stop_tool_loop(messages) -> bool:
    """True if the latest tool-call batch would exceed MAX_TOOL_ROUNDS for this
    turn, or exactly repeats the batch just before it (the model is stuck)."""
    batches = current_turn_tool_batches(messages)
    if len(batches) > MAX_TOOL_ROUNDS:
        return True
    return len(batches) >= 2 and batches[-1] == batches[-2]


def tool_limit_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """Ends the turn with a short, honest message instead of running more tools.

    The pending function calls still get (not-executed) responses first: the
    thread history is replayed to Gemini on the next question, and a function
    call with no response after it would make that request invalid."""
    last_message = state["messages"][-1]
    calls = get_function_calls(last_message)
    logging.warning(
        f"Tool loop stopped after {len(current_turn_tool_batches(state['messages']))} "
        f"batches; pending calls: {[(c.name, c.args) for c in calls]}"
    )
    recorder = get_recorder(config)
    if recorder:
        recorder.capped = True
    not_run = [
        _function_response_part(call, {"result": "Not executed: tool-call limit reached."})
        for call in calls
    ]
    return {"messages": [
        types.Content(role="user", parts=not_run),
        types.Content(role="model", parts=[types.Part.from_text(text=TOOL_LIMIT_MESSAGE)]),
    ]}
