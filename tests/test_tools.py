"""Tests for tools.py — DatasetTools search, read, and LangChain tool building."""
import json
import pytest
from unittest.mock import patch

from tools import DatasetTools


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def csv_dir(tmp_path):
    (tmp_path / "01_gdp.csv").write_text(
        "country,year,gdp\nUSA,2020,21000\nChina,2020,14000\n"
    )
    (tmp_path / "02_climate.csv").write_text(
        "region,climate_anomaly\nGlobal climate,0.5\nArctic climate,2.1\n"
    )
    (tmp_path / "readme.txt").write_text("not a csv — must be ignored")
    return tmp_path


@pytest.fixture
def dt(csv_dir):
    return DatasetTools(datasets_dir=str(csv_dir))


# ---------------------------------------------------------------------------
# search_datasets
# ---------------------------------------------------------------------------

class TestSearchDatasets:
    def test_increments_search_count_each_call(self, dt):
        dt.search_datasets("gdp")
        dt.search_datasets("climate")
        assert dt.search_count == 2

    def test_returns_matching_result(self, dt):
        result = dt.search_datasets("gdp")
        filenames = [r["filename"] for r in result["results"]]
        assert "01_gdp.csv" in filenames

    def test_no_match_returns_empty_results(self, dt):
        result = dt.search_datasets("unicorns_xyz_nonexistent")
        assert result["results"] == []

    def test_respects_max_results(self, csv_dir):
        for i in range(3, 8):
            (csv_dir / f"0{i}_gdp_extra.csv").write_text("country,gdp\nX,100\n")
        dt = DatasetTools(datasets_dir=str(csv_dir))
        result = dt.search_datasets("gdp", max_results=2)
        assert len(result["results"]) <= 2

    def test_result_contains_required_fields(self, dt):
        result = dt.search_datasets("gdp")
        r = result["results"][0]
        for key in ("filename", "title", "headers", "sample_rows", "total_rows", "score"):
            assert key in r

    def test_query_echoed_in_return_value(self, dt):
        result = dt.search_datasets("gdp")
        assert result["query"] == "gdp"

    def test_accumulates_sources_across_calls(self, dt):
        dt.search_datasets("gdp")
        dt.search_datasets("climate")
        urls = {s["url"] for s in dt.sources}
        assert "01_gdp.csv" in urls
        assert "02_climate.csv" in urls

    def test_deduplicates_sources_on_repeat_match(self, dt):
        dt.search_datasets("gdp")
        dt.search_datasets("gdp")
        assert len(dt.sources) == 1

    def test_source_fields_populated(self, dt):
        dt.search_datasets("gdp")
        src = dt.sources[0]
        for key in ("title", "url", "published_date", "score"):
            assert key in src

    def test_source_url_equals_filename(self, dt):
        dt.search_datasets("gdp")
        assert dt.sources[0]["url"] == "01_gdp.csv"

    def test_skips_non_csv_files(self, dt):
        result = dt.search_datasets("not")
        filenames = [r["filename"] for r in result["results"]]
        assert "readme.txt" not in filenames

    def test_skips_empty_csv_without_raising(self, csv_dir):
        (csv_dir / "99_empty.csv").write_text("")
        dt = DatasetTools(datasets_dir=str(csv_dir))
        result = dt.search_datasets("empty")
        assert isinstance(result["results"], list)

    def test_handles_unreadable_file_gracefully(self, dt, csv_dir):
        real_open = open
        def selective_open(path, *args, **kwargs):
            if "01_gdp" in str(path):
                raise OSError("permission denied")
            return real_open(path, *args, **kwargs)
        with patch("builtins.open", side_effect=selective_open):
            result = dt.search_datasets("gdp")
        assert isinstance(result["results"], list)

    def test_results_sorted_by_score_descending(self, csv_dir):
        # 03_multi has "country" three times → higher score than 01_gdp (once)
        (csv_dir / "03_multi.csv").write_text(
            "country,country_code,country_name\nUSA,US,United States\n"
        )
        dt = DatasetTools(datasets_dir=str(csv_dir))
        result = dt.search_datasets("country")
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_sample_rows_limited_to_five(self, csv_dir):
        rows = "\n".join(f"USA,{i},{i*1000}" for i in range(20))
        (csv_dir / "03_big.csv").write_text(f"country,year,gdp\n{rows}\n")
        dt = DatasetTools(datasets_dir=str(csv_dir))
        result = dt.search_datasets("USA")
        for r in result["results"]:
            assert len(r["sample_rows"]) <= 5


# ---------------------------------------------------------------------------
# get_dataset
# ---------------------------------------------------------------------------

class TestGetDataset:
    def test_returns_headers(self, dt):
        result = dt.get_dataset("01_gdp.csv")
        assert result["headers"] == "country,year,gdp"

    def test_returns_data_rows(self, dt):
        result = dt.get_dataset("01_gdp.csv")
        assert len(result["rows"]) == 2

    def test_total_rows_correct(self, dt):
        result = dt.get_dataset("01_gdp.csv")
        assert result["total_rows"] == 2

    def test_filename_in_result(self, dt):
        result = dt.get_dataset("01_gdp.csv")
        assert result["filename"] == "01_gdp.csv"

    def test_missing_file_returns_error_dict(self, dt):
        result = dt.get_dataset("nonexistent.csv")
        assert "error" in result

    def test_path_traversal_prevented(self, dt, tmp_path):
        secret = tmp_path.parent / "secret.csv"
        secret.write_text("sensitive,data\nyes,yes\n")
        result = dt.get_dataset("../secret.csv")
        assert "error" in result

    def test_double_dot_in_path_prevented(self, dt):
        result = dt.get_dataset("../../etc/passwd")
        assert "error" in result

    def test_read_error_returns_error_with_filename(self, dt):
        with patch("builtins.open", side_effect=OSError("disk error")):
            result = dt.get_dataset("01_gdp.csv")
        assert "error" in result
        assert result["filename"] == "01_gdp.csv"

    def test_no_rows_key_on_missing_file(self, dt):
        result = dt.get_dataset("ghost.csv")
        assert "rows" not in result


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

class TestOnActionCallback:
    def test_search_fires_callback(self, csv_dir):
        log = []
        dt = DatasetTools(datasets_dir=str(csv_dir), on_action=lambda k, v: log.append((k, v)))
        dt.search_datasets("gdp")
        assert ("search", "gdp") in log

    def test_get_fires_callback_with_safe_name(self, csv_dir):
        log = []
        dt = DatasetTools(datasets_dir=str(csv_dir), on_action=lambda k, v: log.append((k, v)))
        dt.get_dataset("01_gdp.csv")
        assert ("read", "01_gdp.csv") in log

    def test_callback_called_before_result_returned(self, csv_dir):
        order = []
        def on_action(k, v):
            order.append("callback")
        dt = DatasetTools(datasets_dir=str(csv_dir), on_action=on_action)
        dt.search_datasets("gdp")
        assert order[0] == "callback"

    def test_no_callback_when_none(self, dt):
        dt.search_datasets("gdp")  # must not raise when on_action is None
        dt.get_dataset("01_gdp.csv")

    def test_callback_receives_safe_filename_not_traversal(self, csv_dir):
        log = []
        dt = DatasetTools(datasets_dir=str(csv_dir), on_action=lambda k, v: log.append((k, v)))
        dt.get_dataset("../secret.csv")
        # callback must receive the basename, not the raw traversal path
        assert all("/" not in v for _, v in log)


class TestBuild:
    def test_returns_two_tools(self, dt):
        assert len(dt.build()) == 2

    def test_tool_names(self, dt):
        names = {t.name for t in dt.build()}
        assert names == {"search_datasets", "get_dataset"}

    def test_search_tool_returns_valid_json(self, dt):
        search_tool = next(t for t in dt.build() if t.name == "search_datasets")
        raw = search_tool.func(query="gdp", max_results=5)
        parsed = json.loads(raw)
        assert "results" in parsed

    def test_get_tool_returns_valid_json(self, dt):
        get_tool = next(t for t in dt.build() if t.name == "get_dataset")
        raw = get_tool.func(filename="01_gdp.csv")
        parsed = json.loads(raw)
        assert "filename" in parsed or "error" in parsed

    def test_search_tool_wires_to_search_datasets(self, dt):
        search_tool = next(t for t in dt.build() if t.name == "search_datasets")
        search_tool.func(query="gdp", max_results=5)
        assert dt.search_count == 1

    def test_get_tool_wires_to_get_dataset(self, dt):
        get_tool = next(t for t in dt.build() if t.name == "get_dataset")
        raw = get_tool.func(filename="01_gdp.csv")
        parsed = json.loads(raw)
        assert parsed.get("filename") == "01_gdp.csv"

    def test_build_calls_are_independent(self, dt):
        tools_a = dt.build()
        tools_b = dt.build()
        assert tools_a is not tools_b
