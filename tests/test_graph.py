"""Tests for graph.py — build_react_graph ReAct loop wiring."""
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from graph import build_react_graph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def echo_tool():
    @tool
    def echo(text: str) -> str:
        """Echo the input text back."""
        return f"echoed:{text}"
    return echo


@pytest.fixture
def mock_llm():
    """Mock LLM whose bind_tools returns a mock with controllable invoke."""
    llm = MagicMock()
    llm_with_tools = MagicMock()
    llm.bind_tools.return_value = llm_with_tools
    return llm, llm_with_tools


def _tool_call(name, args, call_id="c1"):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

class TestBuildReactGraph:
    def test_returns_invocable_graph(self, mock_llm):
        llm, llm_wt = mock_llm
        llm_wt.invoke.return_value = AIMessage(content="done")
        graph = build_react_graph(llm, [], "System")
        assert callable(getattr(graph, "invoke", None))

    def test_binds_tools_to_llm(self, mock_llm, echo_tool):
        llm, llm_wt = mock_llm
        llm_wt.invoke.return_value = AIMessage(content="done")
        build_react_graph(llm, [echo_tool], "System")
        llm.bind_tools.assert_called_once_with([echo_tool])

    def test_empty_tool_list_accepted(self, mock_llm):
        llm, llm_wt = mock_llm
        llm_wt.invoke.return_value = AIMessage(content="ok")
        graph = build_react_graph(llm, [], "System")
        result = graph.invoke({"messages": [HumanMessage(content="hi")]})
        assert result["messages"][-1].content == "ok"


# ---------------------------------------------------------------------------
# Termination (no tool calls)
# ---------------------------------------------------------------------------

class TestTermination:
    def test_terminates_immediately_on_final_answer(self, mock_llm):
        llm, llm_wt = mock_llm
        llm_wt.invoke.return_value = AIMessage(content="Final answer")
        graph = build_react_graph(llm, [], "System")
        result = graph.invoke({"messages": [HumanMessage(content="query")]})
        assert result["messages"][-1].content == "Final answer"
        llm_wt.invoke.assert_called_once()

    def test_last_message_is_ai_message(self, mock_llm):
        llm, llm_wt = mock_llm
        llm_wt.invoke.return_value = AIMessage(content="done")
        graph = build_react_graph(llm, [], "System")
        result = graph.invoke({"messages": [HumanMessage(content="q")]})
        assert isinstance(result["messages"][-1], AIMessage)

    def test_user_message_preserved_in_state(self, mock_llm):
        llm, llm_wt = mock_llm
        llm_wt.invoke.return_value = AIMessage(content="done")
        graph = build_react_graph(llm, [], "System")
        result = graph.invoke({"messages": [HumanMessage(content="hello")]})
        assert result["messages"][0].content == "hello"


# ---------------------------------------------------------------------------
# System message injection
# ---------------------------------------------------------------------------

class TestSystemMessage:
    def test_system_message_prepended_to_llm_call(self, mock_llm):
        llm, llm_wt = mock_llm
        llm_wt.invoke.return_value = AIMessage(content="done")
        graph = build_react_graph(llm, [], "My system prompt")
        graph.invoke({"messages": [HumanMessage(content="hi")]})
        first_msg = llm_wt.invoke.call_args[0][0][0]
        assert isinstance(first_msg, SystemMessage)
        assert first_msg.content == "My system prompt"

    def test_user_message_follows_system_in_llm_call(self, mock_llm):
        llm, llm_wt = mock_llm
        llm_wt.invoke.return_value = AIMessage(content="done")
        graph = build_react_graph(llm, [], "System")
        graph.invoke({"messages": [HumanMessage(content="my query")]})
        call_msgs = llm_wt.invoke.call_args[0][0]
        assert isinstance(call_msgs[1], HumanMessage)
        assert call_msgs[1].content == "my query"

    def test_system_message_not_stored_in_graph_state(self, mock_llm):
        """System message is injected on each call but must not accumulate in state."""
        llm, llm_wt = mock_llm
        llm_wt.invoke.return_value = AIMessage(content="ok")
        graph = build_react_graph(llm, [], "System")
        result = graph.invoke({"messages": [HumanMessage(content="hi")]})
        stored_types = [type(m).__name__ for m in result["messages"]]
        assert "SystemMessage" not in stored_types

    def test_different_system_prompts_passed_correctly(self, mock_llm):
        llm, llm_wt = mock_llm
        llm_wt.invoke.return_value = AIMessage(content="done")
        graph = build_react_graph(llm, [], "Specialist prompt for testing")
        graph.invoke({"messages": [HumanMessage(content="q")]})
        first_msg = llm_wt.invoke.call_args[0][0][0]
        assert first_msg.content == "Specialist prompt for testing"


# ---------------------------------------------------------------------------
# Tool routing
# ---------------------------------------------------------------------------

class TestToolRouting:
    def test_routes_to_tool_then_terminates(self, mock_llm, echo_tool):
        llm, llm_wt = mock_llm
        llm_wt.invoke.side_effect = [
            AIMessage(content="", tool_calls=[_tool_call("echo", {"text": "hello"})]),
            AIMessage(content="Final after tool"),
        ]
        graph = build_react_graph(llm, [echo_tool], "System")
        result = graph.invoke({"messages": [HumanMessage(content="q")]})
        assert result["messages"][-1].content == "Final after tool"
        assert llm_wt.invoke.call_count == 2

    def test_tool_result_appears_in_message_history(self, mock_llm, echo_tool):
        llm, llm_wt = mock_llm
        llm_wt.invoke.side_effect = [
            AIMessage(content="", tool_calls=[_tool_call("echo", {"text": "hi"})]),
            AIMessage(content="done"),
        ]
        graph = build_react_graph(llm, [echo_tool], "System")
        result = graph.invoke({"messages": [HumanMessage(content="q")]})
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1

    def test_tool_result_content_correct(self, mock_llm, echo_tool):
        llm, llm_wt = mock_llm
        llm_wt.invoke.side_effect = [
            AIMessage(content="", tool_calls=[_tool_call("echo", {"text": "world"})]),
            AIMessage(content="done"),
        ]
        graph = build_react_graph(llm, [echo_tool], "System")
        result = graph.invoke({"messages": [HumanMessage(content="q")]})
        tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        assert "echoed:world" in tool_msg.content

    def test_tool_result_fed_back_before_second_llm_call(self, mock_llm, echo_tool):
        llm, llm_wt = mock_llm
        responses = [
            AIMessage(content="", tool_calls=[_tool_call("echo", {"text": "x"})]),
            AIMessage(content="done"),
        ]
        call_log = []

        def recording_invoke(msgs):
            call_log.append(list(msgs))
            return responses[len(call_log) - 1]

        llm_wt.invoke.side_effect = recording_invoke

        graph = build_react_graph(llm, [echo_tool], "System")
        graph.invoke({"messages": [HumanMessage(content="q")]})

        second_msgs = call_log[1]  # messages passed to the second LLM call
        msg_types = [type(m).__name__ for m in second_msgs]
        assert "ToolMessage" in msg_types

    def test_multiple_tool_calls_in_single_turn(self, mock_llm, echo_tool):
        llm, llm_wt = mock_llm
        llm_wt.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("echo", {"text": "a"}, "c1"),
                    _tool_call("echo", {"text": "b"}, "c2"),
                ],
            ),
            AIMessage(content="done"),
        ]
        graph = build_react_graph(llm, [echo_tool], "System")
        result = graph.invoke({"messages": [HumanMessage(content="q")]})
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 2

    def test_multi_turn_tool_use(self, mock_llm, echo_tool):
        llm, llm_wt = mock_llm
        llm_wt.invoke.side_effect = [
            AIMessage(content="", tool_calls=[_tool_call("echo", {"text": "first"}, "c1")]),
            AIMessage(content="", tool_calls=[_tool_call("echo", {"text": "second"}, "c2")]),
            AIMessage(content="All done"),
        ]
        graph = build_react_graph(llm, [echo_tool], "System")
        result = graph.invoke({"messages": [HumanMessage(content="q")]})
        assert result["messages"][-1].content == "All done"
        assert llm_wt.invoke.call_count == 3
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 2
