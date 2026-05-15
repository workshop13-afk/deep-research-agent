import os
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypedDict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from rich.console import Console

from graph import build_react_graph
from prompts import DEFAULT_PROMPT, SYSTEM_PROMPTS
from tools import DatasetTools

_console = Console()

DEFAULT_MODEL = "llama3"
MAX_TOKENS = 8096
MAX_ITERATIONS = 12


def _feed_think(token: str, state: dict, callback: Callable[[str], None]) -> None:
    """State machine: fires callback for each token inside <think>...</think> blocks."""
    OPEN, CLOSE = "<think>", "</think>"
    buf = state["buf"] + token

    while buf:
        if not state["in_think"]:
            idx = buf.find(OPEN)
            if idx == -1:
                tail = min(len(buf), len(OPEN) - 1)
                state["buf"] = buf[-tail:] if tail else ""
                return
            buf = buf[idx + len(OPEN):]
            state["in_think"] = True
        else:
            idx = buf.find(CLOSE)
            if idx == -1:
                # Keep only the minimal tail needed to detect a partial close tag
                tail = 0
                for n in range(min(len(buf), len(CLOSE) - 1), 0, -1):
                    if buf[-n:] == CLOSE[:n]:
                        tail = n
                        break
                emit = buf[:-tail] if tail else buf
                if emit:
                    callback(emit)
                state["buf"] = buf[-tail:] if tail else ""
                return
            if idx > 0:
                callback(buf[:idx])
            buf = buf[idx + len(CLOSE):]
            state["in_think"] = False

    state["buf"] = ""


class _ThinkStreamHandler(BaseCallbackHandler):
    """LangChain callback that fires on_thinking for tokens inside <think> blocks."""

    def __init__(self, on_thinking: Callable[[str], None]) -> None:
        super().__init__()
        self._on_thinking = on_thinking
        self._state: dict = {"in_think": False, "buf": ""}

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        _feed_think(token, self._state, self._on_thinking)


class ResearchResult(TypedDict):
    query: str
    report: str
    thinking: str
    sources: list[dict]
    search_count: int
    system_prompt_name: str
    timestamp: str


class DeepResearchAgent:
    def __init__(
        self,
        system_prompt_name: str = DEFAULT_PROMPT,
        console: Console | None = None,
    ) -> None:
        self.llm = ChatOpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ.get("LLM_API_KEY", "not-needed"),
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            max_tokens=MAX_TOKENS,
            streaming=True,
        )
        self.system_prompt_name = system_prompt_name
        self.system_prompt = SYSTEM_PROMPTS.get(system_prompt_name, SYSTEM_PROMPTS[DEFAULT_PROMPT])
        self._console = console if console is not None else _console

    def research(
        self,
        query: str,
        on_action: Callable[[str, str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ResearchResult:
        self._console.print(
            f"\n[bold cyan]Deep Research Agent[/bold cyan]  mode=[bold]{self.system_prompt_name}[/bold]"
        )
        self._console.print(f"[bold]Query:[/bold] {query}\n")

        dataset_tools = DatasetTools(on_action=on_action, console=self._console)
        tools = dataset_tools.build()
        graph = build_react_graph(self.llm, tools, self.system_prompt)

        self._console.print("[dim]Researching…[/dim]")
        invoke_config: dict = {"recursion_limit": MAX_ITERATIONS * 3}
        if on_thinking:
            invoke_config["callbacks"] = [_ThinkStreamHandler(on_thinking)]
        result = graph.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config=invoke_config,
        )
        raw = result["messages"][-1].content if result.get("messages") else ""
        thinking_blocks = re.findall(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
        thinking = "\n\n---\n\n".join(b.strip() for b in thinking_blocks if b.strip())
        final_report = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        self._console.print("[dim]Research complete.[/dim]")

        return ResearchResult(
            query=query,
            report=final_report,
            thinking=thinking,
            sources=dataset_tools.sources,
            search_count=dataset_tools.search_count,
            system_prompt_name=self.system_prompt_name,
            timestamp=datetime.now().isoformat(),
        )
