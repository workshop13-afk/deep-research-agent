"""Tests for main.py — _check_env, _print_modes, _interactive_prompt, and main()."""
import sys
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Shared mock result used across main() tests
# ---------------------------------------------------------------------------

def _mock_result(report="## Report\n\nDone.", mode="general", query="test topic"):
    return {
        "report": report,
        "sources": [{"title": "A", "url": "https://example.com/a", "score": 0.9, "published_date": ""}],
        "search_count": 3,
        "system_prompt_name": mode,
        "timestamp": "2025-05-13T10:00:00",
        "query": query,
    }


# ---------------------------------------------------------------------------
# _check_env
# ---------------------------------------------------------------------------

class TestCheckEnv:
    def test_passes_with_both_keys_set(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k1")
        monkeypatch.setenv("TAVILY_API_KEY", "k2")
        from main import _check_env
        with patch("main.console"):
            _check_env()  # must not raise

    def test_exits_when_anthropic_key_missing(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        from main import _check_env
        with patch("main.console"), pytest.raises(SystemExit):
            _check_env()

    def test_exits_when_tavily_key_missing(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        from main import _check_env
        with patch("main.console"), pytest.raises(SystemExit):
            _check_env()

    def test_exits_when_both_keys_missing(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        from main import _check_env
        with patch("main.console"), pytest.raises(SystemExit):
            _check_env()

    def test_prints_missing_key_names_on_failure(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        from main import _check_env
        with patch("main.console") as mock_console, pytest.raises(SystemExit):
            _check_env()
        output = str(mock_console.print.call_args_list)
        assert "ANTHROPIC_API_KEY" in output


# ---------------------------------------------------------------------------
# _print_modes
# ---------------------------------------------------------------------------

class TestPrintModes:
    def test_table_has_row_for_every_mode(self):
        from main import _print_modes
        from prompts import SYSTEM_PROMPTS
        with patch("main.console") as mock_console:
            _print_modes()
        table = mock_console.print.call_args[0][0]
        assert table.row_count == len(SYSTEM_PROMPTS)

    def test_default_mode_label_present(self):
        from main import _print_modes
        from prompts import DEFAULT_PROMPT
        with patch("main.console") as mock_console:
            _print_modes()
        # The table's cells are built with the default label; ensure console.print was called
        mock_console.print.assert_called_once()

    def test_runs_without_error(self):
        from main import _print_modes
        with patch("main.console"):
            _print_modes()  # should not raise


# ---------------------------------------------------------------------------
# _interactive_prompt
# ---------------------------------------------------------------------------

class TestInteractivePrompt:
    def test_returns_query_and_mode(self):
        from main import _interactive_prompt
        with patch("main.console") as mock_console, patch("main._print_modes"):
            mock_console.input.side_effect = ["my research query", "tech"]
            query, mode = _interactive_prompt("general")
        assert query == "my research query"
        assert mode == "tech"

    def test_empty_mode_input_uses_default(self):
        from main import _interactive_prompt
        with patch("main.console") as mock_console, patch("main._print_modes"):
            mock_console.input.side_effect = ["some query", ""]
            _, mode = _interactive_prompt("science")
        assert mode == "science"

    def test_unknown_mode_falls_back_to_default(self):
        from main import _interactive_prompt
        with patch("main.console") as mock_console, patch("main._print_modes"):
            mock_console.input.side_effect = ["my query", "not_a_real_mode"]
            _, mode = _interactive_prompt("general")
        assert mode == "general"

    def test_empty_query_triggers_sys_exit(self):
        from main import _interactive_prompt
        with patch("main.console") as mock_console, patch("main._print_modes"):
            mock_console.input.side_effect = ["", "general"]
            with pytest.raises(SystemExit):
                _interactive_prompt("general")

    def test_whitespace_only_query_triggers_sys_exit(self):
        from main import _interactive_prompt
        with patch("main.console") as mock_console, patch("main._print_modes"):
            mock_console.input.side_effect = ["   ", "general"]
            with pytest.raises(SystemExit):
                _interactive_prompt("general")


# ---------------------------------------------------------------------------
# main() — CLI entry point
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_list_modes_flag_prints_and_returns(self):
        from main import main
        with patch("sys.argv", ["main.py", "--list-modes"]), \
             patch("main._print_modes") as mock_pm, \
             patch("main._check_env"):
            main()
        mock_pm.assert_called_once()

    def test_query_from_positional_arg(self):
        from main import main
        with patch("sys.argv", ["main.py", "test topic"]), \
             patch("main._check_env"), \
             patch("main.DeepResearchAgent") as MockAgent, \
             patch("main.save_report", return_value="reports/r.md"), \
             patch("main.console"):
            MockAgent.return_value.research.return_value = _mock_result()
            main()
        MockAgent.return_value.research.assert_called_once_with("test topic")

    def test_default_mode_is_general(self):
        from main import main
        with patch("sys.argv", ["main.py", "any query"]), \
             patch("main._check_env"), \
             patch("main.DeepResearchAgent") as MockAgent, \
             patch("main.save_report", return_value="r.md"), \
             patch("main.console"):
            MockAgent.return_value.research.return_value = _mock_result()
            main()
        MockAgent.assert_called_once_with(system_prompt_name="general")

    def test_mode_flag_forwarded_to_agent(self):
        from main import main
        with patch("sys.argv", ["main.py", "--mode", "tech", "rust lang"]), \
             patch("main._check_env"), \
             patch("main.DeepResearchAgent") as MockAgent, \
             patch("main.save_report", return_value="r.md"), \
             patch("main.console"):
            MockAgent.return_value.research.return_value = _mock_result(mode="tech")
            main()
        MockAgent.assert_called_once_with(system_prompt_name="tech")

    def test_short_mode_flag(self):
        from main import main
        with patch("sys.argv", ["main.py", "-m", "market", "EV trends"]), \
             patch("main._check_env"), \
             patch("main.DeepResearchAgent") as MockAgent, \
             patch("main.save_report", return_value="r.md"), \
             patch("main.console"):
            MockAgent.return_value.research.return_value = _mock_result(mode="market")
            main()
        MockAgent.assert_called_once_with(system_prompt_name="market")

    def test_output_dir_forwarded_to_save_report(self):
        from main import main
        with patch("sys.argv", ["main.py", "--output", "/tmp/my_reports", "topic"]), \
             patch("main._check_env"), \
             patch("main.DeepResearchAgent") as MockAgent, \
             patch("main.save_report") as mock_save, \
             patch("main.console"):
            MockAgent.return_value.research.return_value = _mock_result()
            mock_save.return_value = "/tmp/my_reports/report.md"
            main()
        mock_save.assert_called_once()
        _, save_kwargs = mock_save.call_args
        assert save_kwargs.get("output_dir") == "/tmp/my_reports"

    def test_empty_report_triggers_sys_exit(self):
        from main import main
        with patch("sys.argv", ["main.py", "topic"]), \
             patch("main._check_env"), \
             patch("main.DeepResearchAgent") as MockAgent, \
             patch("main.save_report"), \
             patch("main.console"):
            MockAgent.return_value.research.return_value = _mock_result(report="")
            with pytest.raises(SystemExit):
                main()

    def test_interactive_path_when_no_query_arg(self):
        from main import main
        with patch("sys.argv", ["main.py"]), \
             patch("main._check_env"), \
             patch("main._interactive_prompt", return_value=("interactive q", "general")) as mock_ip, \
             patch("main.DeepResearchAgent") as MockAgent, \
             patch("main.save_report", return_value="r.md"), \
             patch("main.console"):
            MockAgent.return_value.research.return_value = _mock_result(query="interactive q")
            main()
        mock_ip.assert_called_once()
        MockAgent.return_value.research.assert_called_once_with("interactive q")

    def test_save_report_called_with_full_result(self):
        from main import main
        full_result = _mock_result()
        with patch("sys.argv", ["main.py", "topic"]), \
             patch("main._check_env"), \
             patch("main.DeepResearchAgent") as MockAgent, \
             patch("main.save_report") as mock_save, \
             patch("main.console"):
            MockAgent.return_value.research.return_value = full_result
            mock_save.return_value = "reports/r.md"
            main()
        mock_save.assert_called_once()
        saved_result = mock_save.call_args[0][0]
        assert saved_result == full_result
