"""Test bootstrap: make the repository root importable, and keep tests offline.

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


#: Every credential that costs money if a test reaches the network with it.
#:
#: Kept as a literal as well as being derived from the provider registry, so a
#: key that is not a provider (the LLM ones) is still covered. The registry walk
#: is what stops a newly added provider from being forgotten here.
BILLED_CREDENTIALS = (
    "SERPAPI_KEY",            # per-search billing
    "TAVILY_API_KEY",         # per-page billing
    "FIRECRAWL_API_KEY",      # per-page billing
    "OPENROUTER_API_KEY",     # per-token billing
    "GEMINI_API_KEY",         # free tier, but it exhausts the quota
    "GROQ_API_KEY",
    "COHERE_API_KEY",
    "HF_TOKEN",
    "GITHUB_TOKEN",           # not billed, but it lifts rate limits
    "UNPAYWALL_EMAIL",
    "EXA_API_KEY",
    "JINA_API_KEY",
    "TINYFISH_API_KEY",
    "SCRAPINGBEE_API_KEY",
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: hits a real paid API; keep credentials visible to the test",
    )
    _preload_dotenv_consumers()


def _preload_dotenv_consumers():
    """Import litellm once, at session start, so it cannot re-load .env later.

    ``litellm/__init__.py`` calls ``load_dotenv()`` at import time. That import
    is otherwise lazy -- ``backend.research.llm`` pulls it in -- and it happens
    *inside* ``clean_env``, after the fixture has already scrubbed the API keys.
    The result is that deleting TAVILY_API_KEY or OPENROUTER_API_KEY was
    silently undone a few lines later, and offline tests went on to make real
    billed calls.

    It looked fine because whichever test imported llm first paid the cost for
    all the others, so the suite passed while quietly spending money. One day
    that test is deleted, or run alone, and every subsequent test inherits a
    live key.

    Doing the import here, before any fixture runs, means the one-time load has
    already happened by the time ``clean_env`` scrubs.
    """
    try:
        import litellm  # noqa: F401
    except Exception:
        pass


def _scrub_credentials(monkeypatch):
    """Remove every paid credential from the environment."""
    for name in BILLED_CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
    try:
        from backend.research.providers import all_providers

        for provider in all_providers():
            monkeypatch.delenv(provider.env, raising=False)
            monkeypatch.delenv(f"{provider.name.upper()}_ENABLED",
                               raising=False)
    except Exception:
        # Best effort: never let an import problem stop the suite running. The
        # literal list above has already covered the registry by hand.
        pass
    monkeypatch.delenv("SEARCH_FANOUT", raising=False)
    monkeypatch.delenv("EXTRACT_TIER_ORDER", raising=False)
    monkeypatch.delenv("EXTRACT_GRACE_S", raising=False)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, request, tmp_path):
    """Keep ambient credentials and ambient behaviour out of every test.

    Tests marked ``live`` need the real keys, so nothing is scrubbed for them.
    """
    live = request.node.get_closest_marker("live") is not None

    if not live:
        _scrub_credentials(monkeypatch)
        # Mock mode is opt-in; an ambient LLM_MOCK/SEARCH_MOCK would silently
        # turn real requests into canned data and make a run look like it
        # passed.
        for name in ("LLM_MOCK", "SEARCH_MOCK",
                     "LLM_MOCK_FAIL", "SEARCH_MOCK_FAIL"):
            monkeypatch.delenv(name, raising=False)

        # Wikipedia is a live network baseline; keep the suite offline.
        monkeypatch.setenv("WIKI_ENABLED", "0")
        # Query expansion hits real SerpApi autocomplete calls; keep the suite
        # offline and deterministic by defaulting it off here.
        monkeypatch.setenv("SEARCH_EXPAND_QUERIES", "0")

        # Pacing is a runtime policy, not a correctness one. Leaving it on would
        # add real seconds to every test that exercises the model path; the
        # tests that verify throttling itself turn it back on with a fast rate.
        from backend.research import throttle as throttle_module

        throttle_module.reset_all()
        for name in ("THROTTLE_LLM", "THROTTLE_SEARCH"):
            monkeypatch.setenv(name, "0")
        monkeypatch.setenv("THROTTLE_LLM_PER_MINUTE", "6000")
        monkeypatch.setenv("THROTTLE_SEARCH_PER_MINUTE", "6000")

    # Model health is file-backed in production; point it at a per-test file
    # so cooldowns/ceilings never leak between cases via .cache/.
    monkeypatch.setenv("MODEL_HEALTH_PATH", str(tmp_path / "model_health.json"))
    try:
        from backend.research import llm as llm_module

        llm_module.reset_health(clear_file=True)
    except Exception:
        pass

    # Re-scrub, last. Importing llm pulls in litellm, which calls load_dotenv()
    # at import time and writes every key from .env back into the environment.
    # _preload_dotenv_consumers normally makes that a no-op by taking the import
    # out of the way at session start, but re-scrubbing is what makes the
    # guarantee hold even if a late litellm import creeps back in: a leaked
    # credential then shows up as a failing test rather than a billed request.
    if not live:
        _scrub_credentials(monkeypatch)

    try:
        from backend.research import throttle as throttle_module
    except Exception:
        throttle_module = None

    if throttle_module is not None:
        throttle_module.reset_all()
    yield
    if throttle_module is not None:
        throttle_module.reset_all()