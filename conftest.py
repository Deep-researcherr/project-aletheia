"""Test bootstrap: make the repository root importable.

Tests import ``backend.research.*``, which requires the *repository root* on
``sys.path`` -- not ``backend/``. Relying on pytest's implicit rootdir insertion
works from the repository root but breaks when pytest is invoked from
elsewhere, from an installed console script, or under a different import mode.
Both paths are added explicitly so the suite is location-independent.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = REPO_ROOT / "backend"

for path in (REPO_ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: hits a real paid API; keep credentials visible to the test",
    )


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, request, tmp_path):
    """Keep ambient credentials from leaking into tests.

    Tests marked ``live`` need the real key, so SERPAPI_KEY is left alone.
    """
    if request.node.get_closest_marker("live") is None:
        monkeypatch.delenv("SERPAPI_KEY", raising=False)
        # Paid extraction fallbacks: a real key in the ambient env would let
        # an offline test make real, billed network calls. OPENROUTER_API_KEY
        # belongs here too -- the service dashboard reads its credit balance at
        # the start of every run, so leaving it set made offline tests hit a
        # paid provider and added minutes to the suite.
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    # Mock mode is opt-in; an ambient LLM_MOCK/SEARCH_MOCK would silently turn
    # real requests into canned data and make a run look like it passed.
    if request.node.get_closest_marker("live") is None:
        for name in ("LLM_MOCK", "SEARCH_MOCK",
                     "LLM_MOCK_FAIL", "SEARCH_MOCK_FAIL"):
            monkeypatch.delenv(name, raising=False)

    # Model health is file-backed in production; point it at a per-test file
    # so cooldowns/ceilings never leak between cases via .cache/.
    monkeypatch.setenv("MODEL_HEALTH_PATH", str(tmp_path / "model_health.json"))
    try:
        from backend.research import llm as llm_module
        llm_module.reset_health(clear_file=True)
    except Exception:
        pass

    # Wikipedia is a live network baseline; keep the suite offline.
    monkeypatch.setenv("WIKI_ENABLED", "0")
    # Query expansion hits real SerpApi autocomplete calls; keep the suite
    # offline and deterministic by defaulting it off here.
    monkeypatch.setenv("SEARCH_EXPAND_QUERIES", "0")

    # Pacing is a runtime policy, not a correctness one. Leaving it on would add
    # real seconds to every test that exercises the model path; the tests that
    # verify throttling itself turn it back on with a fast rate.
    from backend.research import throttle as throttle_module

    throttle_module.reset_all()
    for name in ("THROTTLE_LLM", "THROTTLE_SEARCH"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("THROTTLE_LLM_PER_MINUTE", "6000")
    monkeypatch.setenv("THROTTLE_SEARCH_PER_MINUTE", "6000")
    throttle_module.reset_all()
    yield
    throttle_module.reset_all()
