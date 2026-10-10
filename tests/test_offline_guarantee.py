"""The test suite must not reach the network.

Every credential here is real and billed. An offline test that inherits one can
spend real money and, worse, still pass -- which is how the leak this file
guards against survived for as long as it did.

The subtle failure is ordering. ``litellm/__init__.py`` calls ``load_dotenv()``
at import time, and ``backend.research.llm`` imports it lazily. If that import
happens inside the ``clean_env`` fixture *after* the keys are scrubbed, the
whole .env is written back into ``os.environ`` and the scrub is a no-op. It
looked fine because whichever test imported llm first paid the cost for all the
others, so the suite passed while quietly making requests.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Anything here reaching the network during an offline test costs real money.
BILLED = (
    "SERPAPI_KEY", "TAVILY_API_KEY", "FIRECRAWL_API_KEY", "OPENROUTER_API_KEY",
    "GEMINI_API_KEY", "GROQ_API_KEY", "COHERE_API_KEY", "HF_TOKEN",
    "EXA_API_KEY", "JINA_API_KEY", "TINYFISH_API_KEY", "SCRAPINGBEE_API_KEY",
)


@pytest.mark.parametrize("name", BILLED)
def test_no_paid_credential_survives_the_scrub(name):
    """Every paid key is gone by the time a test body runs."""
    assert os.environ.get(name) in (None, ""), \
        f"{name} is set during an offline test"


@pytest.mark.parametrize("name", BILLED)
def test_importing_llm_does_not_resurrect_credentials(name):
    """The ordering hazard itself.

    `from backend.research import llm` is what pulls in litellm, and litellm
    calls load_dotenv() at import. If that ever becomes lazy again, this is the
    test that notices -- importing llm must not put a key back.
    """
    import backend.research.llm  # noqa: F401

    assert os.environ.get(name) in (None, ""), \
        f"importing llm restored {name}; something re-loaded .env after the scrub"


def test_the_registry_is_what_gets_scrubbed():
    """Adding a provider must not need a second edit here.

    conftest walks providers.all_providers(); if that walk silently covered
    nothing, new keys would leak without any test noticing.
    """
    from backend.research.providers import all_providers

    envs = {p.env for p in all_providers()}
    assert envs, "the registry returned no providers"
    for expected in ("EXA_API_KEY", "JINA_API_KEY",
                     "TINYFISH_API_KEY", "SCRAPINGBEE_API_KEY"):
        assert expected in envs, f"{expected} is not in the registry"


def test_litellm_is_imported_before_any_fixture_scrubs():
    """The session-start preload, verified from a cold start.

    In-process this is unobservable: by the time any test runs, llm has already
    been imported by clean_env itself, so the ordering cannot be inspected from
    here. A subprocess can.

    The invariant is the one conftest relies on: once litellm has been imported
    *before* the scrub, a later ``from backend.research import llm`` must not
    write the credentials back. Run it the other way round -- scrub first, then
    import llm -- and they all come back, which is the bug this file exists to
    prevent.
    """
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(REPO)!r})
        from dotenv import load_dotenv

        from backend.research import providers
        from conftest import _preload_dotenv_consumers

        load_dotenv()                      # .env is in the environment
        assert os.environ.get("TAVILY_API_KEY"), "precondition: .env not loaded"

        # What conftest.pytest_configure does, before any fixture runs.
        _preload_dotenv_consumers()

        # What clean_env does.
        for name in providers.all_providers():
            os.environ.pop(name.env, None)
        os.environ.pop("TAVILY_API_KEY", None)

        # What clean_env also does: import llm, which imports litellm.
        from backend.research import llm  # noqa: F401

        leaked = sorted({{p.env for p in providers.all_providers()
                       if os.environ.get(p.env)}})
        print("LEAKED", leaked)
    """)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO), capture_output=True, text=True, timeout=240,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "LEAKED []" in result.stdout, (
        "importing llm after the scrub restored credentials, so the "
        "session-start preload is no longer doing its job. "
        f"stdout={result.stdout!r}")