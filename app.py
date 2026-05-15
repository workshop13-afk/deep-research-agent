import os
import sys
import tempfile
import threading
from collections.abc import Callable

import streamlit as st
from dotenv import load_dotenv
from rich.console import Console
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

from agent import DeepResearchAgent, ResearchResult
from prompts import DEFAULT_PROMPT, PROMPT_DESCRIPTIONS, SYSTEM_PROMPTS, VULNERABLE_PROMPTS
from report import save_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

_QUIET_CONSOLE = Console(quiet=True)


# ── Streamlit-aware agent ────────────────────────────────────────────────────
class _StreamlitAgent(DeepResearchAgent):
    """Subclass that fires a UI callback on tool actions and silences Rich output."""

    def __init__(self, system_prompt_name: str, on_action: Callable[[str, str], None]) -> None:
        super().__init__(system_prompt_name, console=_QUIET_CONSOLE)
        self._on_action = on_action

    def research(self, query: str, on_thinking: Callable[[str], None] | None = None) -> ResearchResult:
        return super().research(query, on_action=self._on_action, on_thinking=on_thinking)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _report_as_string(result: ResearchResult) -> str:
    with tempfile.TemporaryDirectory() as td:
        path = save_report(result, output_dir=td)
        return open(path, encoding="utf-8").read()


def _mode_label(name: str) -> str:
    tag = "🔴 VULN" if name in VULNERABLE_PROMPTS else "🟢"
    return f"{tag}  {name}"


# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [("result", None), ("report_md", None), ("running", False), ("error", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    with st.expander("🔑 Connection", expanded=not (
        os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_MODEL")
    )):
        llm_base_url = st.text_input(
            "LLM Base URL",
            value=os.environ.get("LLM_BASE_URL", ""),
            placeholder="http://localhost:11434/v1",
        )
        llm_model = st.text_input(
            "Model Name",
            value=os.environ.get("LLM_MODEL", ""),
            placeholder="llama3",
        )
        llm_api_key = st.text_input(
            "API Key (optional)",
            value=os.environ.get("LLM_API_KEY", ""),
            type="password",
            placeholder="not-needed",
        )

    st.markdown("---")
    st.markdown("**Research Mode**")

    mode_names = list(SYSTEM_PROMPTS)
    selected_mode = st.selectbox(
        "mode",
        mode_names,
        index=mode_names.index(DEFAULT_PROMPT),
        format_func=_mode_label,
        label_visibility="collapsed",
    )

    desc = PROMPT_DESCRIPTIONS.get(selected_mode, "")
    if selected_mode in VULNERABLE_PROMPTS:
        st.warning(f"**Intentionally vulnerable** — for security testing.\n\n{desc}", icon="⚠️")
    else:
        st.info(desc, icon="ℹ️")

    with st.expander("View system prompt"):
        st.code(SYSTEM_PROMPTS[selected_mode], language=None)

    st.markdown("---")
    st.caption("Powered by local LLM + local datasets")


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("# 🔍 Deep Research Agent")
st.caption("Search local datasets, synthesise findings, and generate a structured report.")

query = st.text_area(
    "Research query",
    placeholder="e.g.  What are the global GDP trends over the last decade?",
    height=90,
    disabled=st.session_state.running,
    label_visibility="collapsed",
)

col_run, col_clear = st.columns([1, 5])
run_clicked = col_run.button(
    "🚀 Research",
    type="primary",
    disabled=st.session_state.running or not query.strip(),
)
if col_clear.button("✖ Clear", disabled=st.session_state.running):
    st.session_state.result = None
    st.session_state.report_md = None
    st.rerun()


# ── Research execution ────────────────────────────────────────────────────────
if run_clicked and query.strip():
    if not llm_base_url or not llm_model:
        st.error("LLM Base URL and Model Name are required. Enter them in the sidebar ▶ Connection.")
        st.stop()

    os.environ["LLM_BASE_URL"] = llm_base_url
    os.environ["LLM_MODEL"] = llm_model
    os.environ["LLM_API_KEY"] = llm_api_key

    st.session_state.result = None
    st.session_state.report_md = None
    st.session_state.running = True

    # Live thinking display — created before research so tokens stream in immediately
    with st.expander("🧠 Model reasoning", expanded=True):
        think_slot = st.empty()
    _think_buf: list[str] = []

    with st.status("Researching…", expanded=False) as status:
        _ctx = get_script_run_ctx()

        def on_thinking(token: str) -> None:
            add_script_run_ctx(threading.current_thread(), _ctx)
            _think_buf.append(token)
            think_slot.markdown("".join(_think_buf))

        def on_action(kind: str, value: str) -> None:
            add_script_run_ctx(threading.current_thread(), _ctx)
            icon = "🔎" if kind == "search" else "📄"
            status.write(f"{icon} **{kind}:** {value}")

        try:
            agent = _StreamlitAgent(system_prompt_name=selected_mode, on_action=on_action)
            result = agent.research(query.strip(), on_thinking=on_thinking)

            if result["report"]:
                status.update(
                    label=f"✅ Done — {result['search_count']} searches · {len(result['sources'])} sources",
                    state="complete",
                    expanded=False,
                )
                st.session_state.result = result
                st.session_state.report_md = _report_as_string(result)
                st.session_state.error = None
            else:
                st.session_state.error = "Agent did not produce a report. Try a more specific query."
                status.update(label="⚠️ Agent did not produce a report.", state="error")

        except Exception as exc:
            st.session_state.error = f"{type(exc).__name__}: {exc}"
            status.update(label=f"❌ {exc}", state="error")

    st.session_state.running = False
    st.rerun()


# ── Error display ────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(st.session_state.error)

# ── Report display ────────────────────────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result

    m1, m2, m3 = st.columns(3)
    m1.metric("Searches", result["search_count"])
    m2.metric("Sources", len(result["sources"]))
    m3.metric("Mode", result["system_prompt_name"])

    st.divider()

    if result.get("thinking"):
        with st.expander("🧠 Model reasoning", expanded=False):
            st.markdown(result["thinking"])

    st.markdown(result["report"])

    if result["sources"]:
        with st.expander(f"📚 Sources ({len(result['sources'])})"):
            for i, src in enumerate(
                sorted(result["sources"], key=lambda x: -x.get("score", 0)), 1
            ):
                title = src.get("title") or src["url"]
                pub = f" — {src['published_date']}" if src.get("published_date") else ""
                st.markdown(f"{i}. [{title}]({src['url']}){pub}")

    st.divider()

    if st.session_state.report_md:
        safe = "".join(
            c if c.isalnum() or c in " -_" else "_" for c in query[:40]
        ).strip().replace(" ", "_")
        st.download_button(
            label="⬇️ Download Report (.md)",
            data=st.session_state.report_md,
            file_name=f"report_{safe}.md",
            mime="text/markdown",
        )
