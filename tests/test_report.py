"""Tests for report.py — save_report() covering file creation, content, and edge cases."""
import os
import pytest
from unittest.mock import patch


SAMPLE_RESULT = {
    "query": "quantum computing advances",
    "report": "## Executive Summary\n\nSignificant advances.\n\n## Key Findings\n\nFinding 1.",
    "sources": [
        {"title": "Source A", "url": "https://example.com/a", "published_date": "2025-01-15", "score": 0.95},
        {"title": "Source B", "url": "https://example.com/b", "published_date": "2025-01-10", "score": 0.80},
    ],
    "search_count": 4,
    "system_prompt_name": "tech",
    "timestamp": "2025-05-13T14:30:00",
}


@pytest.fixture
def result():
    return dict(SAMPLE_RESULT)


def run_save(result_dict, tmp_path):
    from report import save_report
    with patch("report.console"):
        return save_report(result_dict, output_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------

class TestFileCreation:
    def test_creates_output_directory(self, result, tmp_path):
        output_dir = str(tmp_path / "new_subdir")
        from report import save_report
        with patch("report.console"):
            save_report(result, output_dir=output_dir)
        assert os.path.isdir(output_dir)

    def test_returns_filepath_string(self, result, tmp_path):
        path = run_save(result, tmp_path)
        assert isinstance(path, str)

    def test_file_has_md_extension(self, result, tmp_path):
        path = run_save(result, tmp_path)
        assert path.endswith(".md")

    def test_file_exists_after_save(self, result, tmp_path):
        path = run_save(result, tmp_path)
        assert os.path.isfile(path)

    def test_filename_contains_timestamp(self, result, tmp_path):
        path = run_save(result, tmp_path)
        assert "20250513_143000" in os.path.basename(path)

    def test_filename_contains_query_slug(self, result, tmp_path):
        path = run_save(result, tmp_path)
        filename = os.path.basename(path).lower()
        assert "quantum" in filename

    def test_long_query_truncated_in_filename(self, result, tmp_path):
        result["query"] = "a" * 200
        path = run_save(result, tmp_path)
        # The slug portion (after timestamp prefix) must be derived from at most 50 chars
        filename = os.path.basename(path)
        # Strip "report_YYYYMMDD_HHMMSS_" prefix and ".md" suffix
        parts = filename.split("_", 3)
        slug = parts[3].replace(".md", "") if len(parts) > 3 else ""
        assert len(slug) <= 55

    def test_special_chars_sanitized_in_filename(self, result, tmp_path):
        result["query"] = "What is 2+2? <AI> & 'test'"
        path = run_save(result, tmp_path)
        filename = os.path.basename(path)
        for ch in "<>&'\"":
            assert ch not in filename


# ---------------------------------------------------------------------------
# Report content
# ---------------------------------------------------------------------------

class TestReportContent:
    def test_header_contains_query(self, result, tmp_path):
        path = run_save(result, tmp_path)
        content = open(path).read()
        assert "quantum computing advances" in content

    def test_header_contains_date(self, result, tmp_path):
        path = run_save(result, tmp_path)
        content = open(path).read()
        assert "2025-05-13" in content

    def test_header_contains_mode(self, result, tmp_path):
        path = run_save(result, tmp_path)
        content = open(path).read()
        assert "tech" in content

    def test_header_contains_search_count(self, result, tmp_path):
        path = run_save(result, tmp_path)
        content = open(path).read()
        assert "4" in content

    def test_header_contains_source_count(self, result, tmp_path):
        path = run_save(result, tmp_path)
        content = open(path).read()
        assert "2" in content

    def test_report_body_included(self, result, tmp_path):
        path = run_save(result, tmp_path)
        content = open(path).read()
        assert "Significant advances." in content


# ---------------------------------------------------------------------------
# Sources appendix
# ---------------------------------------------------------------------------

class TestSourcesAppendix:
    def test_sources_section_appended_when_not_in_body(self, result, tmp_path):
        path = run_save(result, tmp_path)
        content = open(path).read()
        assert "## Sources" in content
        assert "https://example.com/a" in content
        assert "https://example.com/b" in content

    def test_sources_not_duplicated_when_already_in_body(self, result, tmp_path):
        result["report"] = (
            "## Report\n\nFindings.\n\n## Sources\n\n1. [A](https://example.com/a)"
        )
        path = run_save(result, tmp_path)
        content = open(path).read()
        assert content.count("## Sources") == 1

    def test_no_sources_section_when_sources_empty(self, result, tmp_path):
        result["sources"] = []
        path = run_save(result, tmp_path)
        content = open(path).read()
        assert "## Sources" not in content

    def test_sources_sorted_by_score_descending(self, result, tmp_path):
        path = run_save(result, tmp_path)
        content = open(path).read()
        pos_a = content.find("example.com/a")
        pos_b = content.find("example.com/b")
        assert pos_a < pos_b  # higher score (A=0.95) appears before lower (B=0.80)

    def test_source_with_no_title_uses_url_as_link_text(self, result, tmp_path):
        result["sources"] = [
            {"title": "", "url": "https://example.com/notitle", "published_date": "", "score": 0.9}
        ]
        path = run_save(result, tmp_path)
        content = open(path).read()
        # URL used as link text when title is empty
        assert "https://example.com/notitle" in content

    def test_source_without_published_date_omits_date_suffix(self, result, tmp_path):
        result["sources"] = [
            {"title": "No Date", "url": "https://example.com/nodate", "published_date": "", "score": 0.8}
        ]
        path = run_save(result, tmp_path)
        content = open(path).read()
        # The " — <date>" suffix must not appear for this entry
        assert "No Date" in content

    def test_published_date_included_in_sources(self, result, tmp_path):
        path = run_save(result, tmp_path)
        content = open(path).read()
        assert "2025-01-15" in content
