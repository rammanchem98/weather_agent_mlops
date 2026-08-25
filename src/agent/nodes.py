import logging
from google import genai
from google.genai import types
from google.genai.types import AutomaticFunctionCallingConfig

from src.agent.state import AgentState
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

def agent_brain(state: AgentState) -> dict:
    """Reasoning Node: Passes conversation state to Gemini 2.5 Flash."""
    config = types.GenerateContentConfig(
        tools=AVAILABLE_TOOLS,
        temperature=0.0,
        automatic_function_calling=AutomaticFunctionCallingConfig(disable=True)
    )

    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=state["messages"],
        config=config
    )

    candidate = response.candidates[0].content
    function_calls= get_function_calls(candidate)
    if function_calls:
        logging.info(f"Function calls detected:{[(f.name,f.args) for f in function_calls]}")
    else:
        text=",".join([part.text for part in candidate.parts if part.text])
        logging.info(f"No function calls detected. Response: {text}")
    return {"messages": [candidate]}

def execute_tools_node(state: AgentState) -> dict:
    """Action Node: Intercepts function_calls, runs them locally, and builds responses."""
    last_message = state["messages"][-1]
    tool_responses = []
    function_calls = get_function_calls(last_message)

    if function_calls:
        for call in function_calls:
            tool_name = call.name
            tool_args = call.args or {}

            # Execute tool locally
            if tool_name in TOOL_REGISTRY:
                try:
                    tool_output = TOOL_REGISTRY[tool_name](**tool_args)
                except Exception as e:
                    tool_output = f"Tool execution failed: {e}"
            else:
                tool_output = f"Tool '{tool_name}' not recognized."

            # Package as FunctionResponse
            tool_responses.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": tool_output}
                )
            )

    return {"messages": [types.Content(role="user", parts=tool_responses)]}
