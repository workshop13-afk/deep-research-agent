"""
Tests for app.py — three layers:
  1. Unit tests  : _mode_label, _report_as_string
  2. Class tests : _StreamlitAgent (callback wiring, quiet console, research delegation)
  3. AppTest UI  : layout, sidebar state, result display, interactions
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from streamlit.testing.v1 import AppTest
from prompts import SYSTEM_PROMPTS, VULNERABLE_PROMPTS, DEFAULT_PROMPT

with patch("report.console"):
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
        {"title": "Source A", "url": "file_a.csv", "score": 0.95, "published_date": "2025-01-15"},
        {"title": "Source B", "url": "file_b.csv", "score": 0.80, "published_date": ""},
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
def mock_graph():
    graph = MagicMock()
    graph.invoke.return_value = {"messages": [AIMessage(content="## Report\n\nDone.")]}
    return graph


@pytest.fixture
def mock_dataset_tools():
    dt = MagicMock()
    dt.build.return_value = []
    dt.sources = []
    dt.search_count = 0
    return dt


@pytest.fixture
def streamlit_agent(mock_graph, mock_dataset_tools):
    log = []
    with (
        patch("agent.ChatOpenAI"),
        patch("agent.DatasetTools", return_value=mock_dataset_tools),
        patch("agent.build_react_graph", return_value=mock_graph),
        patch("agent.Progress"),
    ):
        agent = _StreamlitAgent(
            system_prompt_name="general",
            on_action=lambda kind, val: log.append((kind, val)),
        )
    agent._log = log
    return agent


@pytest.fixture
def app_bare():
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    return at


@pytest.fixture
def app_with_result():
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
        assert "🟢" in _mode_label("general")

    def test_secure_mode_contains_name(self):
        assert "tech" in _mode_label("tech")

    def test_vulnerable_mode_shows_red_vuln(self):
        assert "🔴 VULN" in _mode_label("customer_support_v1")

    def test_vulnerable_mode_contains_name(self):
        assert "finance_advisor_legacy" in _mode_label("finance_advisor_legacy")

    def test_all_secure_modes_get_green(self):
        for name in SECURE_MODES:
            assert "🟢" in _mode_label(name), f"Expected 🟢 for secure mode '{name}'"

    def test_all_vulnerable_modes_get_red(self):
        for name in VULN_MODES:
            assert "🔴 VULN" in _mode_label(name), f"Expected 🔴 VULN for '{name}'"

    def test_secure_mode_has_no_vuln_tag(self):
        assert "VULN" not in _mode_label("legal_assistant")

    def test_vulnerable_mode_has_no_green(self):
        assert "🟢" not in _mode_label("hr_assistant_v2")


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
        assert "Source A" in result or "file_a.csv" in result

    def test_result_with_empty_sources(self):
        sparse = dict(MOCK_RESULT, sources=[])
        with patch("report.console"):
            result = _report_as_string(sparse)
        assert isinstance(result, str)
        assert "## Sources" not in result


# ---------------------------------------------------------------------------
# 3. _StreamlitAgent
# ---------------------------------------------------------------------------

class TestStreamlitAgentInit:
    def test_stores_on_action_callback(self):
        on_action = MagicMock()
        with patch("agent.ChatOpenAI"), patch("agent.console"):
            agent = _StreamlitAgent(system_prompt_name="general", on_action=on_action)
        assert agent._on_action is on_action

    def test_uses_quiet_console(self):
        with patch("agent.ChatOpenAI"), patch("agent.console"):
            agent = _StreamlitAgent(system_prompt_name="general", on_action=lambda k, v: None)
        assert agent._console.quiet

    def test_stores_system_prompt_name(self):
        with patch("agent.ChatOpenAI"), patch("agent.console"):
            agent = _StreamlitAgent(system_prompt_name="tech", on_action=lambda k, v: None)
        assert agent.system_prompt_name == "tech"


class TestStreamlitAgentResearch:
    def test_research_returns_report(self, streamlit_agent, mock_graph):
        with (
            patch("agent.DatasetTools", return_value=MagicMock(build=lambda: [], sources=[], search_count=0)),
            patch("agent.build_react_graph", return_value=mock_graph),
        ):
            result = streamlit_agent.research("test query")
        assert "report" in result

    def test_research_passes_on_action_to_parent(self, mock_graph, mock_dataset_tools):
        captured = {}

        def fake_research(query, on_action=None):
            captured["on_action"] = on_action
            return {"query": query, "report": "done", "sources": [], "search_count": 0,
                    "system_prompt_name": "general", "timestamp": "2025-01-01T00:00:00"}

        log = []
        with (
            patch("agent.ChatOpenAI"),
            patch("agent.console"),
        ):
            agent = _StreamlitAgent("general", on_action=lambda k, v: log.append((k, v)))

        with patch.object(type(agent).__bases__[0], "research", fake_research):
            agent.research("my query")

        assert captured["on_action"] is agent._on_action

    def test_on_action_fires_when_dataset_searched(self, mock_graph):
        """Integration: on_action callback reaches DatasetTools via the research chain."""
        log = []
        on_action = lambda kind, val: log.append((kind, val))

        with (
            patch("agent.ChatOpenAI"),
            patch("agent.build_react_graph", return_value=mock_graph),
            patch("agent._console"),
        ):
            from agent import DeepResearchAgent
            dt_real = __import__("tools").DatasetTools(on_action=on_action)
            with patch("agent.DatasetTools", return_value=dt_real):
                a = DeepResearchAgent(system_prompt_name="general")
                dt_real._on_action = on_action
                dt_real.search_datasets("gdp")

        assert ("search", "gdp") in log

    def test_research_produces_no_rich_stdout(self, mock_graph, mock_dataset_tools):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with (
            patch("agent.DatasetTools", return_value=mock_dataset_tools),
            patch("agent.build_react_graph", return_value=mock_graph),
        ):
            with redirect_stdout(buf):
                streamlit_agent = _StreamlitAgent.__new__(_StreamlitAgent)
                from prompts import SYSTEM_PROMPTS
                from rich.console import Console
                streamlit_agent._console = Console(quiet=True)
                streamlit_agent.system_prompt_name = "general"
                streamlit_agent.system_prompt = SYSTEM_PROMPTS["general"]
                streamlit_agent.llm = MagicMock()
                streamlit_agent._on_action = None
                streamlit_agent.research("query")
        assert "[dim cyan]" not in buf.getvalue()


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

    def test_connection_inputs_in_sidebar(self, app_bare):
        assert len(app_bare.text_input) >= 2

    def test_no_metrics_before_research(self, app_bare):
        assert len(app_bare.metric) == 0

    def test_no_report_md_before_research(self, app_bare):
        assert not app_bare.session_state["report_md"]


# ---------------------------------------------------------------------------
# 5. AppTest — sidebar mode selection
# ---------------------------------------------------------------------------

class TestAppSidebarModes:
    def test_default_mode_is_general(self, app_bare):
        assert app_bare.selectbox[0].value == DEFAULT_PROMPT

    def test_secure_mode_shows_info_not_warning(self, app_bare):
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
        assert len(at.metric) == 3

        next(b for b in at.button if "Clear" in b.label).click()
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

    def test_error_shown_when_llm_config_missing(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)

        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()
        at.text_area[0].input("some query")
        at.run()

        run_btn = next((b for b in at.button if "Research" in b.label), None)
        if run_btn and not run_btn.disabled:
            run_btn.click()
            at.run()
            assert len(at.error) >= 1

    def test_run_button_click_triggers_status_widget(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("LLM_MODEL", "llama3")

        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()
        at.text_area[0].input("quantum computing advances")
        at.run()

        next(b for b in at.button if "Research" in b.label).click()
        at.run()

        assert len(at.status) >= 1
