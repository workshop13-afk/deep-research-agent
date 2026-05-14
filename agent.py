import json
import os
import re
from datetime import datetime

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from prompts import DEFAULT_PROMPT, SYSTEM_PROMPTS

console = Console()

DEFAULT_MODEL = "llama3"
MAX_TOKENS = 8096
MAX_ITERATIONS = 12
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


class DeepResearchAgent:
    def __init__(self, system_prompt_name: str = DEFAULT_PROMPT):
        self.llm = ChatOpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ.get("LLM_API_KEY", "not-needed"),
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            max_tokens=MAX_TOKENS,
        )
        self.system_prompt_name = system_prompt_name
        self.system_prompt = SYSTEM_PROMPTS.get(system_prompt_name, SYSTEM_PROMPTS[DEFAULT_PROMPT])
        self.search_count = 0
        self.sources: list[dict] = []

    def _search_datasets(self, query: str, max_results: int = 5) -> dict:
        self.search_count += 1
        console.print(f"    [dim cyan]search:[/dim cyan] {query}")
        keywords = [w.lower() for w in query.split() if len(w) > 2]
        results = []

        for fname in sorted(os.listdir(DATASETS_DIR)):
            if not fname.endswith(".csv"):
                continue
            filepath = os.path.join(DATASETS_DIR, fname)
            try:
                with open(filepath, encoding="utf-8") as f:
                    lines = f.read().strip().splitlines()
                if not lines:
                    continue
                file_text = "\n".join(lines).lower()
                score = sum(1 for kw in keywords if kw in file_text)
                if score == 0:
                    continue
                matching_rows = [
                    line for line in lines[1:]
                    if any(kw in line.lower() for kw in keywords)
                ][:5]
                results.append({
                    "filename": fname,
                    "title": fname.replace("_", " ").replace(".csv", ""),
                    "headers": lines[0],
                    "sample_rows": matching_rows,
                    "total_rows": len(lines) - 1,
                    "score": score,
                })
            except Exception:
                continue

        results.sort(key=lambda x: -x["score"])
        results = results[:max_results]

        seen = {s["url"] for s in self.sources}
        for r in results:
            if r["filename"] not in seen:
                seen.add(r["filename"])
                self.sources.append({
                    "title": r["title"],
                    "url": r["filename"],
                    "published_date": "",
                    "score": r["score"],
                })

        return {"results": results, "query": query}

    def _get_dataset(self, filename: str) -> dict:
        safe_name = os.path.basename(filename)  # prevent path traversal
        filepath = os.path.join(DATASETS_DIR, safe_name)
        console.print(f"    [dim cyan]read:[/dim cyan] {safe_name}")
        if not os.path.exists(filepath):
            return {"error": f"Dataset '{safe_name}' not found."}
        try:
            with open(filepath, encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
            return {
                "filename": safe_name,
                "headers": lines[0] if lines else "",
                "rows": lines[1:] if len(lines) > 1 else [],
                "total_rows": len(lines) - 1,
            }
        except Exception as exc:
            return {"error": str(exc), "filename": safe_name}

    def _build_tools(self) -> list:
        class _SearchInput(BaseModel):
            query: str = Field(description="Keywords or topic to search for across all datasets")
            max_results: int = Field(default=5, description="Maximum number of datasets to return (1-10)")

        class _GetInput(BaseModel):
            filename: str = Field(description="Dataset filename e.g. '01_global_gdp_by_country.csv'")

        return [
            StructuredTool.from_function(
                func=lambda query, max_results=5: json.dumps(
                    self._search_datasets(query, max_results), ensure_ascii=False
                ),
                name="search_datasets",
                description=(
                    "Search the local research datasets for data matching a topic or keyword. "
                    "Returns matching dataset filenames, their column headers, and sample rows. "
                    "Call this multiple times with different keywords to find all relevant data."
                ),
                args_schema=_SearchInput,
            ),
            StructuredTool.from_function(
                func=lambda filename: json.dumps(
                    self._get_dataset(filename), ensure_ascii=False
                ),
                name="get_dataset",
                description=(
                    "Read the full contents of a specific dataset file. "
                    "Use the filename returned by search_datasets to retrieve all rows and columns."
                ),
                args_schema=_GetInput,
            ),
        ]

    def research(self, query: str) -> dict:
        console.print(f"\n[bold cyan]Deep Research Agent[/bold cyan]  mode=[bold]{self.system_prompt_name}[/bold]")
        console.print(f"[bold]Query:[/bold] {query}\n")

        tools = self._build_tools()
        agent = create_agent(self.llm, tools, system_prompt=self.system_prompt)

        final_report = ""
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
            task = progress.add_task("Researching...", total=None)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"recursion_limit": MAX_ITERATIONS * 3},
            )
            raw = result["messages"][-1].content if result.get("messages") else ""
            # strip model thinking blocks if present
            final_report = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            progress.update(task, description="Research complete.")

        return {
            "query": query,
            "report": final_report,
            "sources": self.sources,
            "search_count": self.search_count,
            "system_prompt_name": self.system_prompt_name,
            "timestamp": datetime.now().isoformat(),
        }
