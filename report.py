import os
from datetime import datetime

from rich.console import Console

from agent import ResearchResult

console = Console()


def save_report(result: ResearchResult, output_dir: str = "reports") -> str:
    query = result["query"]
    body = result["report"]
    sources = result["sources"]
    timestamp = result["timestamp"]
    mode = result["system_prompt_name"]
    search_count = result["search_count"]

    dt = datetime.fromisoformat(timestamp)
    date_str = dt.strftime("%Y-%m-%d %H:%M")

    header = (
        f"# Research Report\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| **Query** | {query} |\n"
        f"| **Date** | {date_str} |\n"
        f"| **Mode** | {mode} |\n"
        f"| **Searches performed** | {search_count} |\n"
        f"| **Sources collected** | {len(sources)} |\n\n"
        f"---\n\n"
    )

    # Append a sources appendix only if the report body doesn't already contain one
    sources_section = ""
    if sources and "## Sources" not in body:
        ranked = sorted(sources, key=lambda x: -x.get("score", 0))
        lines = ["\n\n---\n\n## Sources\n"]
        for i, src in enumerate(ranked, 1):
            title = src.get("title") or src["url"]
            pub = f" — {src['published_date']}" if src.get("published_date") else ""
            lines.append(f"{i}. [{title}]({src['url']}){pub}")
        sources_section = "\n".join(lines)

    full_report = header + body + sources_section

    os.makedirs(output_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in query[:50]).strip()
    filename = f"report_{dt.strftime('%Y%m%d_%H%M%S')}_{safe.replace(' ', '_')}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(full_report)

    console.print(f"\n[bold green]Report saved →[/bold green] {filepath}")
    return filepath
