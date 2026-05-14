import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # project modules
sys.path.insert(0, os.path.dirname(__file__))  # tests/helpers

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
