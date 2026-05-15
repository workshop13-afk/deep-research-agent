import json
import os
from collections.abc import Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from rich.console import Console

_console = Console()

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


class _SearchInput(BaseModel):
    query: str = Field(description="Keywords or topic to search for across all datasets")
    max_results: int = Field(default=5, description="Maximum number of datasets to return (1-10)")


class _GetInput(BaseModel):
    filename: str = Field(description="Dataset filename e.g. '01_global_gdp_by_country.csv'")


class DatasetTools:
    """Stateful dataset tool implementations that track sources and search count."""

    def __init__(
        self,
        datasets_dir: str = DATASETS_DIR,
        on_action: Callable[[str, str], None] | None = None,
        console: Console | None = None,
    ) -> None:
        self.datasets_dir = datasets_dir
        self._on_action = on_action
        self._console = console if console is not None else _console
        self.search_count: int = 0
        self.sources: list[dict] = []

    def search_datasets(self, query: str, max_results: int = 5) -> dict:
        self.search_count += 1
        if self._on_action:
            self._on_action("search", query)
        self._console.print(f"    [dim cyan]search:[/dim cyan] {query}")
        keywords = [w.lower() for w in query.split() if len(w) > 2]
        results = []

        for fname in sorted(os.listdir(self.datasets_dir)):
            if not fname.endswith(".csv"):
                continue
            filepath = os.path.join(self.datasets_dir, fname)
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

    def get_dataset(self, filename: str) -> dict:
        safe_name = os.path.basename(filename)  # prevent path traversal
        filepath = os.path.join(self.datasets_dir, safe_name)
        if self._on_action:
            self._on_action("read", safe_name)
        self._console.print(f"    [dim cyan]read:[/dim cyan] {safe_name}")
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

    def _search_json(self, query: str, max_results: int = 5) -> str:
        return json.dumps(self.search_datasets(query, max_results), ensure_ascii=False)

    def _get_json(self, filename: str) -> str:
        return json.dumps(self.get_dataset(filename), ensure_ascii=False)

    def build(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self._search_json,
                name="search_datasets",
                description=(
                    "Search the local research datasets for data matching a topic or keyword. "
                    "Returns matching dataset filenames, their column headers, and sample rows. "
                    "Call this multiple times with different keywords to find all relevant data."
                ),
                args_schema=_SearchInput,
            ),
            StructuredTool.from_function(
                func=self._get_json,
                name="get_dataset",
                description=(
                    "Read the full contents of a specific dataset file. "
                    "Use the filename returned by search_datasets to retrieve all rows and columns."
                ),
                args_schema=_GetInput,
            ),
        ]
