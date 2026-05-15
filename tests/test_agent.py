"""Tests for agent.py — DeepResearchAgent init and research orchestration."""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from prompts import DEFAULT_PROMPT, SYSTEM_PROMPTS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_graph(content="## Report\n\nFindings."):
    graph = MagicMock()
    graph.invoke.return_value = {"messages": [AIMessage(content=content)]}
    return graph


def _make_dataset_tools(sources=None, search_count=2):
    dt = MagicMock()
    dt.build.return_value = []
    dt.sources = sources or [{"title": "T", "url": "f.csv", "published_date": "", "score": 1.0}]
    dt.search_count = search_count
    return dt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_graph():
    return _make_graph()


@pytest.fixture
def mock_dataset_tools():
    return _make_dataset_tools()


@pytest.fixture
def agent(mock_graph, mock_dataset_tools):
    with (
        patch("agent.ChatOpenAI"),
        patch("agent._console"),
        patch("agent.Progress"),
        patch("agent.DatasetTools", return_value=mock_dataset_tools),
        patch("agent.build_react_graph", return_value=mock_graph),
    ):
        from agent import DeepResearchAgent
        yield DeepResearchAgent(system_prompt_name="general")


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_stores_system_prompt_name(self, agent):
        assert agent.system_prompt_name == "general"

    def test_stores_system_prompt_text(self, agent):
        assert agent.system_prompt == SYSTEM_PROMPTS["general"]

    def test_prompt_text_non_empty(self, agent):
        assert len(agent.system_prompt) > 50

    def test_custom_mode_stored(self):
        with (
            patch("agent.ChatOpenAI"),
            patch("agent._console"),
        ):
            from agent import DeepResearchAgent
            a = DeepResearchAgent(system_prompt_name="tech")
        assert a.system_prompt_name == "tech"
        assert a.system_prompt == SYSTEM_PROMPTS["tech"]

    def test_unknown_mode_falls_back_to_default_prompt(self):
        with (
            patch("agent.ChatOpenAI"),
            patch("agent._console"),
        ):
            from agent import DeepResearchAgent
            a = DeepResearchAgent(system_prompt_name="totally_made_up")
        assert a.system_prompt == SYSTEM_PROMPTS[DEFAULT_PROMPT]

    def test_unknown_mode_keeps_given_name(self):
        with (
            patch("agent.ChatOpenAI"),
            patch("agent._console"),
        ):
            from agent import DeepResearchAgent
            a = DeepResearchAgent(system_prompt_name="totally_made_up")
        assert a.system_prompt_name == "totally_made_up"

    def test_llm_constructed_with_base_url(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "http://myhost:1234")
        with patch("agent.ChatOpenAI") as mock_chat, patch("agent._console"):
            from agent import DeepResearchAgent
            DeepResearchAgent()
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["base_url"] == "http://myhost:1234"

    def test_llm_constructed_with_model(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "mistral")
        with patch("agent.ChatOpenAI") as mock_chat, patch("agent._console"):
            from agent import DeepResearchAgent
            DeepResearchAgent()
        assert mock_chat.call_args.kwargs["model"] == "mistral"


# ---------------------------------------------------------------------------
# research — result structure
# ---------------------------------------------------------------------------

class TestResearchResultStructure:
    def test_returns_all_required_keys(self, agent):
        result = agent.research("test query")
        for key in ("query", "report", "sources", "search_count", "system_prompt_name", "timestamp"):
            assert key in result

    def test_query_matches_input(self, agent):
        result = agent.research("my research topic")
        assert result["query"] == "my research topic"

    def test_system_prompt_name_in_result(self, agent):
        result = agent.research("q")
        assert result["system_prompt_name"] == "general"

    def test_timestamp_is_valid_iso_string(self, agent):
        result = agent.research("q")
        datetime.fromisoformat(result["timestamp"])  # raises ValueError if invalid


# ---------------------------------------------------------------------------
# research — report extraction
# ---------------------------------------------------------------------------

class TestResearchReport:
    def test_report_from_last_message_content(self, mock_graph, mock_dataset_tools):
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="First"), AIMessage(content="Final report")]
        }
        with (
            patch("agent.ChatOpenAI"),
            patch("agent._console"),
            patch("agent.Progress"),
            patch("agent.DatasetTools", return_value=mock_dataset_tools),
            patch("agent.build_react_graph", return_value=mock_graph),
        ):
            from agent import DeepResearchAgent
            result = DeepResearchAgent().research("q")
        assert result["report"] == "Final report"

    def test_strips_think_tags(self, mock_graph, mock_dataset_tools):
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="<think>internal reasoning</think>Clean report.")]
        }
        with (
            patch("agent.ChatOpenAI"),
            patch("agent._console"),
            patch("agent.Progress"),
            patch("agent.DatasetTools", return_value=mock_dataset_tools),
            patch("agent.build_react_graph", return_value=mock_graph),
        ):
            from agent import DeepResearchAgent
            result = DeepResearchAgent().research("q")
        assert result["report"] == "Clean report."
        assert "<think>" not in result["report"]

    def test_strips_multiline_think_blocks(self, mock_graph, mock_dataset_tools):
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="<think>\nline1\nline2\n</think>Output.")]
        }
        with (
            patch("agent.ChatOpenAI"),
            patch("agent._console"),
            patch("agent.Progress"),
            patch("agent.DatasetTools", return_value=mock_dataset_tools),
            patch("agent.build_react_graph", return_value=mock_graph),
        ):
            from agent import DeepResearchAgent
            result = DeepResearchAgent().research("q")
        assert result["report"] == "Output."

    def test_empty_messages_yields_empty_report(self, mock_dataset_tools):
        graph = MagicMock()
        graph.invoke.return_value = {"messages": []}
        with (
            patch("agent.ChatOpenAI"),
            patch("agent._console"),
            patch("agent.Progress"),
            patch("agent.DatasetTools", return_value=mock_dataset_tools),
            patch("agent.build_react_graph", return_value=graph),
        ):
            from agent import DeepResearchAgent
            result = DeepResearchAgent().research("q")
        assert result["report"] == ""


# ---------------------------------------------------------------------------
# research — tool and graph wiring
# ---------------------------------------------------------------------------

class TestResearchWiring:
    def test_sources_come_from_dataset_tools(self, agent, mock_dataset_tools):
        result = agent.research("q")
        assert result["sources"] is mock_dataset_tools.sources

    def test_search_count_comes_from_dataset_tools(self, agent, mock_dataset_tools):
        mock_dataset_tools.search_count = 7
        result = agent.research("q")
        assert result["search_count"] == 7

    def test_graph_invoked_with_user_message(self, agent, mock_graph):
        agent.research("my query")
        call_input = mock_graph.invoke.call_args[0][0]
        assert call_input["messages"][0]["role"] == "user"
        assert call_input["messages"][0]["content"] == "my query"

    def test_recursion_limit_passed_to_graph(self, agent, mock_graph):
        from agent import MAX_ITERATIONS
        agent.research("q")
        config = mock_graph.invoke.call_args[1]["config"]
        assert config["recursion_limit"] == MAX_ITERATIONS * 3

    def test_build_react_graph_called_with_system_prompt(self, mock_dataset_tools):
        graph = _make_graph()
        with (
            patch("agent.ChatOpenAI"),
            patch("agent._console"),
            patch("agent.Progress"),
            patch("agent.DatasetTools", return_value=mock_dataset_tools),
            patch("agent.build_react_graph", return_value=graph) as mock_build,
        ):
            from agent import DeepResearchAgent
            a = DeepResearchAgent(system_prompt_name="tech")
            a.research("q")
        _, kwargs = mock_build.call_args
        # system_prompt is the third positional arg
        assert SYSTEM_PROMPTS["tech"] in mock_build.call_args[0]

    def test_dataset_tools_instantiated_per_research_call(self, mock_graph):
        with (
            patch("agent.ChatOpenAI"),
            patch("agent._console"),
            patch("agent.Progress"),
            patch("agent.DatasetTools") as mock_dt_cls,
            patch("agent.build_react_graph", return_value=mock_graph),
        ):
            mock_dt_cls.return_value = _make_dataset_tools()
            from agent import DeepResearchAgent
            a = DeepResearchAgent()
            a.research("q1")
            a.research("q2")
        assert mock_dt_cls.call_count == 2

    def test_on_action_passed_to_dataset_tools(self, mock_graph):
        callback = MagicMock()
        with (
            patch("agent.ChatOpenAI"),
            patch("agent._console"),
            patch("agent.Progress"),
            patch("agent.DatasetTools") as mock_dt_cls,
            patch("agent.build_react_graph", return_value=mock_graph),
        ):
            mock_dt_cls.return_value = _make_dataset_tools()
            from agent import DeepResearchAgent
            a = DeepResearchAgent()
            a.research("q", on_action=callback)
        assert mock_dt_cls.call_args.kwargs["on_action"] is callback
