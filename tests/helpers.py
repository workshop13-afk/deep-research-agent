"""Shared mock-building helpers for all test modules."""
from unittest.mock import MagicMock

SAMPLE_TAVILY_RESULTS = {
    "query": "test query",
    "results": [
        {
            "title": "Article One",
            "url": "https://example.com/article-1",
            "content": "Detailed content of article one.",
            "score": 0.95,
            "published_date": "2025-01-15",
        },
        {
            "title": "Article Two",
            "url": "https://example.com/article-2",
            "content": "Detailed content of article two.",
            "score": 0.82,
            "published_date": "2025-01-10",
        },
    ],
}

def make_text_block(text="## Report\n\nFindings."):
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def make_tool_block(name="search_web", tool_id="tool_abc", inputs=None):
    b = MagicMock()
    b.type = "tool_use"
    b.name = name
    b.id = tool_id
    b.input = inputs or {"query": "test", "max_results": 5}
    return b


def make_end_turn_response(text="## Report\n\nFindings."):
    r = MagicMock()
    r.stop_reason = "end_turn"
    r.content = [make_text_block(text)]
    return r


def make_tool_response(name="search_web", tool_id="tool_abc", inputs=None):
    r = MagicMock()
    r.stop_reason = "tool_use"
    r.content = [make_tool_block(name, tool_id, inputs)]
    return r
