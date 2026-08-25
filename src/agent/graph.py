from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.nodes import agent_brain, execute_tools_node, get_function_calls

def route_decision(state: AgentState) -> str:
    """Inspects LLM output: if tool calls exist, branch to tools; otherwise END."""
    last_message = state["messages"][-1]
    if get_function_calls(last_message):
        return "execute_tools"
    return END

# 1. Initialize Graph
workflow = StateGraph(AgentState)

# 2. Add Nodes
workflow.add_node("agent_brain", agent_brain)
workflow.add_node("execute_tools", execute_tools_node)

# 3. Add Edges
workflow.add_edge(START, "agent_brain")
workflow.add_conditional_edges(
    "agent_brain",
    route_decision,
    {
        "execute_tools": "execute_tools",
        END: END
    }
)
# ReAct Loop: Return tool observations back to LLM
workflow.add_edge("execute_tools", "agent_brain")

# 4. Compile with Checkpointer for Memory
memory = MemorySaver()
compiled_agent = workflow.compile(checkpointer=memory)
