from typing import Annotated, Sequence
from typing_extensions import TypedDict
from google.genai import types

def append_messages(
    existing: Sequence[types.Content],
    new_items: Sequence[types.Content]
) -> Sequence[types.Content]:
    """Reducer that appends incoming messages to conversation history."""
    return list(existing) + list(new_items)

class AgentState(TypedDict):
    """The central message state of the agent."""
    messages: Annotated[Sequence[types.Content], append_messages]
