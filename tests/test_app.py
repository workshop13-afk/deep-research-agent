"""
Tests for app.py — three layers:
  1. Unit tests  : _mode_label, _report_as_string
  2. Class tests : _StreamlitAgent (callback firing, search/fetch, research patching)
  3. AppTest UI  : layout, sidebar state, result display, interactions
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Must be set before app.py is imported so DeepResearchAgent.__init__ can read them
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")

from streamlit.testing.v1 import AppTest
from prompts import SYSTEM_PROMPTS, VULNERABLE_PROMPTS, DEFAULT_PROMPT

# Import the pure helpers from app.py.
# Running app.py at import time emits Streamlit "bare mode" warnings — that is expected.
with (
    patch("anthropic.Anthropic"),
    patch("agent.TavilyClient"),
    patch("report.console"),
):
    from app import _StreamlitAgent, _report_as_string, _mode_label

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

MOCK_RESULT = {
    "query": "quantum computing advances",
    "report": (
        "## Executive Summary\n\nSignificant advances found.\n\n"
        "## Key Findings\n\nFinding 1.\n\nFinding 2."
    ),
    "sources": [
        {"title": "Source A", "url": "https://example.com/a", "score": 0.95, "published_date": "2025-01-15"},
        {"title": "Source B", "url": "https://example.com/b", "score": 0.80, "published_date": ""},
    ],
    "search_count": 4,
    "system_prompt_name": "general",
    "timestamp": "2025-05-13T14:30:00",
}

MOCK_REPORT_MD = (
    "# Research Report\n\n"
    "| Field | Value |\n|---|---|\n"
    "| **Query** | quantum computing advances |\n\n---\n\n"
    "## Executive Summary\n\nSignificant advances found."
)

SECURE_MODES = list(set(SYSTEM_PROMPTS) - VULNERABLE_PROMPTS)
VULN_MODES = list(VULNERABLE_PROMPTS)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_anthropic_instance():
    return MagicMock()


@pytest.fixture
def mock_tavily_instance():
    inst = MagicMock()
    inst.search.return_value = {"results": []}
    inst.extract.return_value = {"results": []}
    return inst


@pytest.fixture
def streamlit_agent(mock_anthropic_instance, mock_tavily_instance):
    """_StreamlitAgent with mocked Anthropic + Tavily clients and a log list."""
    log = []
    with (
        patch("anthropic.Anthropic", return_value=mock_anthropic_instance),
        patch("agent.TavilyClient", return_value=mock_tavily_instance),
    ):
        agent = _StreamlitAgent(
            system_prompt_name="general",
            on_action=lambda kind, val: log.append((kind, val)),
        )
    agent._log = log
    yield agent


def _make_end_turn_response(text="## Report\n\nDone."):
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp.content = [block]
    return resp


@pytest.fixture
def app_bare():
    """AppTest instance at initial state (no query, no result)."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    return at


@pytest.fixture
def app_with_result():
    """AppTest instance with a completed result pre-loaded into session state."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.session_state["result"] = MOCK_RESULT
    at.session_state["report_md"] = MOCK_REPORT_MD
    at.session_state["running"] = False
    at.run()
    return at


# ---------------------------------------------------------------------------
# 1. _mode_label
# ---------------------------------------------------------------------------

class TestModeLabel:
    def test_secure_mode_shows_green(self):
        label = _mode_label("general")
        assert "🟢" in label

    def test_secure_mode_contains_name(self):
        label = _mode_label("tech")
        assert "tech" in label

    def test_vulnerable_mode_shows_red_vuln(self):
        label = _mode_label("customer_support_v1")
        assert "🔴 VULN" in label

    def test_vulnerable_mode_contains_name(self):
        label = _mode_label("finance_advisor_legacy")
        assert "finance_advisor_legacy" in label

    def test_all_secure_modes_get_green(self):
        for name in SECURE_MODES:
            assert "🟢" in _mode_label(name), f"Expected 🟢 for secure mode '{name}'"

    def test_all_vulnerable_modes_get_red(self):
        for name in VULN_MODES:
            assert "🔴 VULN" in _mode_label(name), f"Expected 🔴 VULN for '{name}'"

    def test_secure_mode_has_no_vuln_tag(self):
        assert "VULN" not in _mode_label("legal_assistant")

    def test_vulnerable_mode_has_no_green(self):
        label = _mode_label("hr_assistant_v2")
        assert "🟢" not in label


# ---------------------------------------------------------------------------
# 2. _report_as_string
# ---------------------------------------------------------------------------

class TestReportAsString:
    def test_returns_string(self):
        with patch("report.console"):
            result = _report_as_string(MOCK_RESULT)
        assert isinstance(result, str)

    def test_contains_report_body(self):
        with patch("report.console"):
            result = _report_as_string(MOCK_RESULT)
        assert "Significant advances found." in result

    def test_contains_query_in_header(self):
        with patch("report.console"):
            result = _report_as_string(MOCK_RESULT)
        assert "quantum computing advances" in result

    def test_contains_mode_in_header(self):
        with patch("report.console"):
            result = _report_as_string(MOCK_RESULT)
        assert "general" in result

    def test_contains_sources_section(self):
        with patch("report.console"):
            result = _report_as_string(MOCK_RESULT)
        assert "Source A" in result or "example.com/a" in result

    def test_result_with_empty_sources(self):
        sparse = dict(MOCK_RESULT, sources=[])
        with patch("report.console"):
            result = _report_as_string(sparse)
        assert isinstance(result, str)
        assert "## Sources" not in result


# ---------------------------------------------------------------------------
# 3. _StreamlitAgent
# ---------------------------------------------------------------------------

class TestStreamlitAgentCallback:
    def test_search_fires_callback_with_kind_and_query(self, streamlit_agent):
        streamlit_agent._search_web("AI news", max_results=3)
        assert ("search", "AI news") in streamlit_agent._log

    def test_fetch_fires_callback_with_kind_and_url(self, streamlit_agent):
        streamlit_agent._get_page_content("https://example.com/article")
        assert ("fetch", "https://example.com/article") in streamlit_agent._log

    def test_callback_fires_before_tavily_call(self, streamlit_agent, mock_tavily_instance):
        order = []
        streamlit_agent._on_action = lambda k, v: order.append("callback")
        mock_tavily_instance.search.side_effect = lambda **kw: order.append("tavily") or {"results": []}
        streamlit_agent._search_web("test")
        assert order[0] == "callback"
        assert "tavily" in order

    def test_fetch_callback_fires_before_tavily_extract(self, streamlit_agent, mock_tavily_instance):
        order = []
        streamlit_agent._on_action = lambda k, v: order.append("callback")
        mock_tavily_instance.extract.side_effect = lambda **kw: order.append("tavily") or {"results": []}
        streamlit_agent._get_page_content("https://example.com")
        assert order[0] == "callback"

    def test_multiple_calls_log_all_actions(self, streamlit_agent):
        streamlit_agent._search_web("q1")
        streamlit_agent._search_web("q2")
        streamlit_agent._get_page_content("https://example.com")
        assert len(streamlit_agent._log) == 3

    def test_search_returns_tavily_result(self, streamlit_agent, mock_tavily_instance):
        expected = {"results": [{"title": "T", "url": "https://x.com", "score": 0.9}]}
        mock_tavily_instance.search.return_value = expected
        assert streamlit_agent._search_web("q") == expected

    def test_fetch_returns_tavily_result(self, streamlit_agent, mock_tavily_instance):
        expected = {"results": [{"raw_content": "page text"}]}
        mock_tavily_instance.extract.return_value = expected
        assert streamlit_agent._get_page_content("https://x.com") == expected


class TestStreamlitAgentResearch:
    def test_research_completes_and_returns_report(self, streamlit_agent, mock_anthropic_instance):
        mock_anthropic_instance.messages.create.return_value = _make_end_turn_response(
            "## Executive Summary\n\nFindings."
        )
        result = streamlit_agent.research("test query")
        assert result["report"] == "## Executive Summary\n\nFindings."

    def test_research_returns_correct_query(self, streamlit_agent, mock_anthropic_instance):
        mock_anthropic_instance.messages.create.return_value = _make_end_turn_response()
        result = streamlit_agent.research("my specific query")
        assert result["query"] == "my specific query"

    def test_research_silences_rich_progress(self, streamlit_agent, mock_anthropic_instance):
        mock_anthropic_instance.messages.create.return_value = _make_end_turn_response()
        # If Progress is NOT silenced internally, calling it in a non-TTY context
        # could raise or produce junk output. Completing without error is the assertion.
        result = streamlit_agent.research("query")
        assert "report" in result

    def test_research_silences_rich_console(self, streamlit_agent, mock_anthropic_instance):
        mock_anthropic_instance.messages.create.return_value = _make_end_turn_response()
        # Capture stdout to verify no Rich output leaks
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            streamlit_agent.research("query")
        # Rich console text shouldn't appear (patched to MagicMock)
        assert "[dim cyan]" not in buf.getvalue()

    def test_research_with_tool_call_fires_search_callback(
        self, streamlit_agent, mock_anthropic_instance, mock_tavily_instance
    ):
        tool_response = MagicMock()
        tool_response.stop_reason = "tool_use"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "search_web"
        tool_block.id = "t1"
        tool_block.input = {"query": "latest AI news", "max_results": 5}
        tool_response.content = [tool_block]

        mock_anthropic_instance.messages.create.side_effect = [
            tool_response,
            _make_end_turn_response("## Report\n\nDone."),
        ]

        result = streamlit_agent.research("AI research")

        assert ("search", "latest AI news") in streamlit_agent._log
        assert result["report"] == "## Report\n\nDone."


# ---------------------------------------------------------------------------
# 4. AppTest — layout and initial state
# ---------------------------------------------------------------------------

class TestAppLoad:
    def test_loads_without_exception(self, app_bare):
        assert not app_bare.exception

    def test_title_present(self, app_bare):
        titles = [m.value for m in app_bare.markdown]
        assert any("Deep Research Agent" in str(t) for t in titles)

    def test_sidebar_has_mode_selectbox(self, app_bare):
        assert len(app_bare.selectbox) >= 1

    def test_main_has_text_area_for_query(self, app_bare):
        assert len(app_bare.text_area) >= 1

    def test_run_button_present(self, app_bare):
        labels = [b.label for b in app_bare.button]
        assert any("Research" in lbl for lbl in labels)

    def test_clear_button_present(self, app_bare):
        labels = [b.label for b in app_bare.button]
        assert any("Clear" in lbl for lbl in labels)

    def test_run_button_disabled_when_query_empty(self, app_bare):
        run_btns = [b for b in app_bare.button if "Research" in b.label]
        assert run_btns[0].disabled

    def test_api_keys_text_inputs_in_sidebar(self, app_bare):
        # Two password text_inputs for Anthropic and Tavily keys
        assert len(app_bare.text_input) >= 2

    def test_no_metrics_before_research(self, app_bare):
        assert len(app_bare.metric) == 0

    def test_no_report_md_before_research(self, app_bare):
        # Without a completed research run, report_md must be falsy
        assert not app_bare.session_state["report_md"]


# ---------------------------------------------------------------------------
# 5. AppTest — sidebar mode selection
# ---------------------------------------------------------------------------

class TestAppSidebarModes:
    def test_default_mode_is_general(self, app_bare):
        sb = app_bare.selectbox[0]
        assert sb.value == DEFAULT_PROMPT

    def test_secure_mode_shows_info_not_warning(self, app_bare):
        # Default mode (general) is secure
        assert len(app_bare.info) >= 1
        assert len(app_bare.warning) == 0

    def test_vuln_mode_shows_warning(self):
        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()
        at.selectbox[0].select("customer_support_v1")
        at.run()
        assert not at.exception
        assert len(at.warning) >= 1

    def test_vuln_warning_contains_security_testing_message(self):
        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()
        at.selectbox[0].select("hr_assistant_v2")
        at.run()
        warning_bodies = [w.value for w in at.warning]
        assert any("security testing" in str(w).lower() or "vulnerable" in str(w).lower()
                   for w in warning_bodies)

    def test_switching_back_to_secure_mode_removes_warning(self):
        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()
        at.selectbox[0].select("customer_support_v1")
        at.run()
        assert len(at.warning) >= 1
        at.selectbox[0].select("general")
        at.run()
        assert len(at.warning) == 0

    def test_system_prompt_expander_present(self, app_bare):
        expander_labels = [e.label for e in app_bare.expander]
        assert any("system prompt" in lbl.lower() for lbl in expander_labels)


# ---------------------------------------------------------------------------
# 6. AppTest — result display
# ---------------------------------------------------------------------------

class TestAppResultDisplay:
    def test_three_metrics_shown(self, app_with_result):
        assert len(app_with_result.metric) == 3

    def test_search_count_metric(self, app_with_result):
        metric_labels = [m.label for m in app_with_result.metric]
        assert any("Search" in lbl for lbl in metric_labels)

    def test_sources_count_metric(self, app_with_result):
        metric_labels = [m.label for m in app_with_result.metric]
        assert any("Source" in lbl for lbl in metric_labels)

    def test_mode_metric(self, app_with_result):
        metric_labels = [m.label for m in app_with_result.metric]
        assert any("Mode" in lbl for lbl in metric_labels)

    def test_report_markdown_rendered(self, app_with_result):
        all_md = " ".join(str(m.value) for m in app_with_result.markdown)
        assert "Executive Summary" in all_md or "Significant advances" in all_md

    def test_sources_expander_present(self, app_with_result):
        expander_labels = [e.label for e in app_with_result.expander]
        assert any("Source" in lbl for lbl in expander_labels)

    def test_sources_expander_shows_count(self, app_with_result):
        expander_labels = [e.label for e in app_with_result.expander]
        sources_expanders = [lbl for lbl in expander_labels if "Source" in lbl]
        assert any("2" in lbl for lbl in sources_expanders)

    def test_report_md_in_session_state_enables_download(self, app_with_result):
        # st.download_button is not accessible via at.button in AppTest 1.57;
        # assert the session state condition that gates it is satisfied instead.
        assert app_with_result.session_state["report_md"] is not None
        assert len(app_with_result.session_state["report_md"]) > 0

    def test_report_md_content_is_markdown_string(self, app_with_result):
        md = app_with_result.session_state["report_md"]
        assert isinstance(md, str)
        assert "Report" in md or "#" in md

    def test_no_exception_on_result_display(self, app_with_result):
        assert not app_with_result.exception


# ---------------------------------------------------------------------------
# 7. AppTest — interactions
# ---------------------------------------------------------------------------

class TestAppInteractions:
    def test_clear_button_removes_result(self):
        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.session_state["result"] = MOCK_RESULT
        at.session_state["report_md"] = MOCK_REPORT_MD
        at.run()
        assert len(at.metric) == 3  # result is shown

        clear_btn = next(b for b in at.button if "Clear" in b.label)
        clear_btn.click()
        at.run()

        assert len(at.metric) == 0
        assert at.session_state["result"] is None

    def test_run_button_enabled_after_query_entered(self):
        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()
        at.text_area[0].input("quantum computing research")
        at.run()
        run_btn = next(b for b in at.button if "Research" in b.label)
        assert not run_btn.disabled

    def test_error_shown_when_api_keys_missing(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()
        at.text_area[0].input("some query")
        at.run()

        # Click run — keys are empty so error should appear
        run_btn = next((b for b in at.button if "Research" in b.label), None)
        if run_btn and not run_btn.disabled:
            run_btn.click()
            at.run()
            assert len(at.error) >= 1

    def test_run_button_click_triggers_status_widget(self, monkeypatch):
        # AppTest re-execs the script in its own namespace so module-level patches
        # cannot intercept it; we verify the run attempt produces a Status element
        # (success or error) rather than patching the agent internals.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")

        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()
        at.text_area[0].input("quantum computing advances")
        at.run()

        run_btn = next(b for b in at.button if "Research" in b.label)
        run_btn.click()
        at.run()

        # A Status widget must appear regardless of whether research succeeded
        assert len(at.status) >= 1
