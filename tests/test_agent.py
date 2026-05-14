"""Tests for agent.py — DeepResearchAgent init, tools, and research loop."""
import json
import pytest
from unittest.mock import MagicMock, patch, call

from helpers import (
    SAMPLE_TAVILY_RESULTS,
    make_end_turn_response,
    make_text_block,
    make_tool_block,
    make_tool_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_anthropic_instance():
    return MagicMock()


@pytest.fixture
def mock_tavily_instance():
    instance = MagicMock()
    instance.search.return_value = SAMPLE_TAVILY_RESULTS
    instance.extract.return_value = {
        "results": [{"url": "https://example.com", "raw_content": "full content"}]
    }
    return instance


@pytest.fixture
def agent(mock_anthropic_instance, mock_tavily_instance):
    """Fully patched agent — patches remain active for the entire test via yield."""
    with (
        patch("anthropic.Anthropic", return_value=mock_anthropic_instance),
        patch("agent.TavilyClient", return_value=mock_tavily_instance),
        patch("agent.console"),
        patch("agent.Progress"),
    ):
        from agent import DeepResearchAgent
        a = DeepResearchAgent(system_prompt_name="general")
        yield a


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_uses_provided_prompt_name(self, agent):
        assert agent.system_prompt_name == "general"

    def test_prompt_text_populated(self, agent):
        assert len(agent.system_prompt) > 50

    def test_custom_mode_stored(self, mock_anthropic_instance, mock_tavily_instance):
        with (
            patch("anthropic.Anthropic", return_value=mock_anthropic_instance),
            patch("agent.TavilyClient", return_value=mock_tavily_instance),
            patch("agent.console"),
            patch("agent.Progress"),
        ):
            from agent import DeepResearchAgent
            a = DeepResearchAgent(system_prompt_name="tech")
            assert a.system_prompt_name == "tech"

    def test_unknown_prompt_name_falls_back_to_general(self, mock_anthropic_instance, mock_tavily_instance):
        with (
            patch("anthropic.Anthropic", return_value=mock_anthropic_instance),
            patch("agent.TavilyClient", return_value=mock_tavily_instance),
            patch("agent.console"),
            patch("agent.Progress"),
        ):
            from agent import DeepResearchAgent
            from prompts import DEFAULT_PROMPT, SYSTEM_PROMPTS
            a = DeepResearchAgent(system_prompt_name="definitely_not_a_real_mode")
            assert a.system_prompt == SYSTEM_PROMPTS[DEFAULT_PROMPT]

    def test_initial_search_count_zero(self, agent):
        assert agent.search_count == 0

    def test_initial_sources_empty(self, agent):
        assert agent.sources == []


# ---------------------------------------------------------------------------
# _search_web
# ---------------------------------------------------------------------------

class TestSearchWeb:
    def test_calls_tavily_with_correct_params(self, agent, mock_tavily_instance):
        agent._search_web("quantum computing", max_results=3)
        mock_tavily_instance.search.assert_called_once_with(
            query="quantum computing",
            search_depth="advanced",
            max_results=3,
            include_published_date=True,
        )

    def test_returns_tavily_results(self, agent, mock_tavily_instance):
        result = agent._search_web("AI")
        assert result == SAMPLE_TAVILY_RESULTS

    def test_increments_search_count(self, agent):
        agent._search_web("q1")
        agent._search_web("q2")
        assert agent.search_count == 2

    def test_tracks_new_sources(self, agent):
        agent._search_web("AI news")
        assert len(agent.sources) == 2
        urls = {s["url"] for s in agent.sources}
        assert "https://example.com/article-1" in urls
        assert "https://example.com/article-2" in urls

    def test_source_fields_populated(self, agent):
        agent._search_web("test")
        src = agent.sources[0]
        assert src["title"] == "Article One"
        assert src["score"] == 0.95
        assert src["published_date"] == "2025-01-15"

    def test_deduplicates_sources_across_calls(self, agent, mock_tavily_instance):
        agent._search_web("q1")
        agent._search_web("q2")  # same URLs returned
        assert len(agent.sources) == 2  # not 4

    def test_caps_max_results_at_10(self, agent, mock_tavily_instance):
        agent._search_web("query", max_results=99)
        assert mock_tavily_instance.search.call_args.kwargs["max_results"] == 10

    def test_skips_results_without_url(self, agent, mock_tavily_instance):
        mock_tavily_instance.search.return_value = {
            "results": [{"title": "No URL Article", "content": "x", "score": 0.5}]
        }
        agent._search_web("test")
        assert agent.sources == []

    def test_handles_missing_published_date(self, agent, mock_tavily_instance):
        mock_tavily_instance.search.return_value = {
            "results": [{"title": "T", "url": "https://example.com/no-date", "score": 0.5}]
        }
        agent._search_web("test")
        assert agent.sources[0]["published_date"] == ""


# ---------------------------------------------------------------------------
# _get_page_content
# ---------------------------------------------------------------------------

class TestGetPageContent:
    def test_calls_tavily_extract_with_url_list(self, agent, mock_tavily_instance):
        agent._get_page_content("https://example.com/article")
        mock_tavily_instance.extract.assert_called_once_with(
            urls=["https://example.com/article"]
        )

    def test_returns_extract_result(self, agent, mock_tavily_instance):
        mock_tavily_instance.extract.return_value = {"results": [{"raw_content": "full text"}]}
        result = agent._get_page_content("https://example.com/article")
        assert result == {"results": [{"raw_content": "full text"}]}

    def test_returns_error_dict_on_exception(self, agent, mock_tavily_instance):
        mock_tavily_instance.extract.side_effect = Exception("Connection timeout")
        result = agent._get_page_content("https://example.com/broken")
        assert result["error"] == "Connection timeout"
        assert result["url"] == "https://example.com/broken"

    def test_does_not_raise_on_exception(self, agent, mock_tavily_instance):
        mock_tavily_instance.extract.side_effect = RuntimeError("unexpected")
        result = agent._get_page_content("https://example.com")
        assert "error" in result


# ---------------------------------------------------------------------------
# _run_tool
# ---------------------------------------------------------------------------

class TestRunTool:
    def test_dispatches_search_web(self, agent, mock_tavily_instance):
        raw = agent._run_tool("search_web", {"query": "AI", "max_results": 3})
        result = json.loads(raw)
        assert "results" in result

    def test_dispatches_get_page_content(self, agent, mock_tavily_instance):
        mock_tavily_instance.extract.return_value = {"results": [{"raw_content": "text"}]}
        raw = agent._run_tool("get_page_content", {"url": "https://example.com"})
        result = json.loads(raw)
        assert "results" in result

    def test_returns_error_for_unknown_tool(self, agent):
        raw = agent._run_tool("nonexistent_tool", {})
        result = json.loads(raw)
        assert "error" in result
        assert "nonexistent_tool" in result["error"]

    def test_search_uses_default_max_results_when_omitted(self, agent, mock_tavily_instance):
        agent._run_tool("search_web", {"query": "test"})
        assert mock_tavily_instance.search.call_args.kwargs["max_results"] == 5

    def test_returns_valid_json_string(self, agent):
        raw = agent._run_tool("nonexistent_tool", {})
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# research — agent loop
# ---------------------------------------------------------------------------

class TestResearch:
    def test_end_turn_single_iteration(self, agent, mock_anthropic_instance):
        report_text = "## Executive Summary\n\nKey findings."
        mock_anthropic_instance.messages.create.return_value = make_end_turn_response(report_text)

        result = agent.research("test query")

        assert result["report"] == report_text
        mock_anthropic_instance.messages.create.assert_called_once()

    def test_result_contains_all_keys(self, agent, mock_anthropic_instance):
        mock_anthropic_instance.messages.create.return_value = make_end_turn_response()

        result = agent.research("test query")

        for key in ("query", "report", "sources", "search_count", "system_prompt_name", "timestamp"):
            assert key in result

    def test_result_query_matches_input(self, agent, mock_anthropic_instance):
        mock_anthropic_instance.messages.create.return_value = make_end_turn_response()
        result = agent.research("specific research question")
        assert result["query"] == "specific research question"

    def test_result_system_prompt_name(self, agent, mock_anthropic_instance):
        mock_anthropic_instance.messages.create.return_value = make_end_turn_response()
        result = agent.research("q")
        assert result["system_prompt_name"] == "general"

    def test_tool_use_then_end_turn(self, agent, mock_anthropic_instance, mock_tavily_instance):
        mock_anthropic_instance.messages.create.side_effect = [
            make_tool_response("search_web", "t1", {"query": "AI news", "max_results": 5}),
            make_end_turn_response("## Final Report\n\nComplete."),
        ]

        result = agent.research("AI developments")

        assert result["report"] == "## Final Report\n\nComplete."
        assert agent.search_count == 1
        assert mock_anthropic_instance.messages.create.call_count == 2

    def test_get_page_content_tool_in_loop(self, agent, mock_anthropic_instance, mock_tavily_instance):
        mock_anthropic_instance.messages.create.side_effect = [
            make_tool_response("get_page_content", "t2", {"url": "https://example.com"}),
            make_end_turn_response("## Report from page."),
        ]

        result = agent.research("deep dive")

        assert result["report"] == "## Report from page."
        mock_tavily_instance.extract.assert_called_once()

    def test_max_iterations_respected(self, agent, mock_anthropic_instance):
        mock_anthropic_instance.messages.create.return_value = make_tool_response(
            "search_web", "t_loop", {"query": "loop forever"}
        )

        result = agent.research("infinite search")

        from agent import MAX_ITERATIONS
        assert mock_anthropic_instance.messages.create.call_count == MAX_ITERATIONS
        assert result["report"] == ""

    def test_message_history_starts_with_user_query(self, agent, mock_anthropic_instance):
        mock_anthropic_instance.messages.create.return_value = make_end_turn_response()
        agent.research("my research query")

        first_call_messages = mock_anthropic_instance.messages.create.call_args.kwargs["messages"]
        assert first_call_messages[0]["role"] == "user"
        assert first_call_messages[0]["content"] == "my research query"

    def test_tool_result_appended_to_message_history(self, agent, mock_anthropic_instance, mock_tavily_instance):
        mock_anthropic_instance.messages.create.side_effect = [
            make_tool_response("search_web", "t1", {"query": "q"}),
            make_end_turn_response("done"),
        ]

        agent.research("query")

        second_call_messages = mock_anthropic_instance.messages.create.call_args_list[1].kwargs["messages"]
        roles = [m["role"] for m in second_call_messages]
        assert "assistant" in roles
        # tool results go back as user role
        user_msgs = [m for m in second_call_messages if m["role"] == "user"]
        # last user message should contain tool_result content
        last_user_content = user_msgs[-1]["content"]
        assert isinstance(last_user_content, list)
        assert last_user_content[0]["type"] == "tool_result"
        assert last_user_content[0]["tool_use_id"] == "t1"

    def test_multiple_tool_blocks_in_one_response(self, agent, mock_anthropic_instance, mock_tavily_instance):
        multi_tool_response = MagicMock()
        multi_tool_response.stop_reason = "tool_use"
        multi_tool_response.content = [
            make_tool_block("search_web", "t1", {"query": "q1"}),
            make_tool_block("search_web", "t2", {"query": "q2"}),
        ]
        mock_anthropic_instance.messages.create.side_effect = [
            multi_tool_response,
            make_end_turn_response("done"),
        ]

        result = agent.research("multi-tool query")

        assert agent.search_count == 2
        assert result["report"] == "done"

    def test_sources_accumulated_across_searches(self, agent, mock_anthropic_instance, mock_tavily_instance):
        second_batch = {
            "results": [
                {"title": "C", "url": "https://example.com/article-3", "content": "c", "score": 0.7, "published_date": ""}
            ]
        }
        mock_tavily_instance.search.side_effect = [SAMPLE_TAVILY_RESULTS, second_batch]
        mock_anthropic_instance.messages.create.side_effect = [
            make_tool_response("search_web", "t1", {"query": "q1"}),
            make_tool_response("search_web", "t2", {"query": "q2"}),
            make_end_turn_response("report"),
        ]

        result = agent.research("multi-search")

        assert len(result["sources"]) == 3

    def test_system_prompt_passed_to_api(self, agent, mock_anthropic_instance):
        mock_anthropic_instance.messages.create.return_value = make_end_turn_response()
        agent.research("q")

        call_kwargs = mock_anthropic_instance.messages.create.call_args.kwargs
        assert call_kwargs["system"] == agent.system_prompt

    def test_tools_list_passed_to_api(self, agent, mock_anthropic_instance):
        from agent import TOOLS
        mock_anthropic_instance.messages.create.return_value = make_end_turn_response()
        agent.research("q")

        call_kwargs = mock_anthropic_instance.messages.create.call_args.kwargs
        assert call_kwargs["tools"] == TOOLS
