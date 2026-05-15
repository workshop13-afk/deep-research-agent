from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.pregel import Pregel
from langgraph.prebuilt import ToolNode


def build_react_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
) -> Pregel:
    """Return a compiled ReAct graph: agent ↔ tools loop until no tool calls remain."""
    llm_with_tools = llm.bind_tools(tools)
    system_message = SystemMessage(content=system_prompt)

    def agent_node(state: MessagesState) -> dict:
        # Prepend system message on each call so it is never stored in persistent state
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal["tools", END]:
        last = state["messages"][-1]
        if last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()
