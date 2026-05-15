"""Shared test data and factory helpers."""


def make_research_result(
    query="test query",
    report="## Report\n\nFindings.",
    sources=None,
    search_count=2,
    system_prompt_name="general",
    timestamp="2025-05-13T10:00:00",
) -> dict:
    return {
        "query": query,
        "report": report,
        "sources": sources if sources is not None else [
            {"title": "Source A", "url": "file_a.csv", "published_date": "", "score": 0.9},
        ],
        "search_count": search_count,
        "system_prompt_name": system_prompt_name,
        "timestamp": timestamp,
    }
