"""The services dashboard: what a run depends on, and what is left of each.

Two properties matter more than the display:

  * it cannot break a run -- a dead or hanging provider must cost bounded time
    and show "unknown", never raise into the pipeline;
  * it reports once -- a burst of runs must not turn into a burst of provider
    lookups, or a wall of repeated text in the run log.

The balance endpoints were verified against the live APIs rather than guessed:
SerpApi ``/account`` (query param), Tavily ``/usage`` (Bearer), Firecrawl
``/v1/team/credit-usage`` (the unversioned host 404s), OpenRouter
``/api/v1/credits``.
"""
import asyncio
import time

import pytest

from backend.research import services


@pytest.fixture(autouse=True)
def _clean():
    services.reset_cache()
    yield
    services.reset_cache()


def _no_keys(monkeypatch):
    for env in ("SERPAPI_KEY", "TAVILY_API_KEY", "FIRECRAWL_API_KEY",
                "OPENROUTER_API_KEY", "UNPAYWALL_EMAIL"):
        monkeypatch.delenv(env, raising=False)


# --- never breaks a run ----------------------------------------------------

def test_a_raising_provider_is_reported_not_propagated(monkeypatch):
    monkeypatch.setattr(services, "_PROBES",
                        (("x", "X", "purpose", "SVC_TEST_KEY",
                          lambda: (_ for _ in ()).throw(RuntimeError("boom"))),))
    monkeypatch.setenv("SVC_TEST_KEY", "k")
    row = asyncio.run(services.snapshot())[0]
    assert "boom" in row.error
    assert row.line().startswith("balance unknown")


def test_hanging_providers_respect_the_ceiling(monkeypatch):
    """A 6s ceiling must cost 6s, not 6s per provider and not 60s.

    asyncio.wait_for cannot cancel a to_thread future once its thread is
    running, so awaiting one waits for the very thing it was meant to bound.
    That is why snapshot() polls futures against a deadline instead.
    """
    def hang():
        time.sleep(30)

    monkeypatch.setattr(services, "_PROBES",
                        tuple((f"p{i}", f"P{i}", "purpose", f"SVC_HANG_{i}", hang)
                              for i in range(4)))
    for i in range(4):
        monkeypatch.setenv(f"SVC_HANG_{i}", "k")
    monkeypatch.setattr(services, "PROBE_TIMEOUT_S", 1.0)

    start = time.monotonic()
    rows = asyncio.run(services.snapshot())
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"ceiling not honoured: {elapsed:.1f}s"
    assert all("timed out" in r.error for r in rows[:4])


def test_probes_run_concurrently_not_in_sequence(monkeypatch):
    """Four slow providers must cost one timeout, not four."""
    def slow():
        time.sleep(2.0)
        return {"balance": "1/1 left"}

    monkeypatch.setattr(services, "_PROBES",
                        tuple((f"p{i}", f"P{i}", "purpose", f"SVC_SLOW_{i}", slow)
                              for i in range(4)))
    for i in range(4):
        monkeypatch.setenv(f"SVC_SLOW_{i}", "k")
    monkeypatch.setattr(services, "PROBE_TIMEOUT_S", 10.0)

    start = time.monotonic()
    rows = asyncio.run(services.snapshot())
    elapsed = time.monotonic() - start

    assert elapsed < 4.0, f"probes appear to be sequential: {elapsed:.1f}s"
    assert all(r.balance == "1/1 left" for r in rows[:4])


def test_snapshot_survives_with_nothing_configured(monkeypatch):
    _no_keys(monkeypatch)
    rows = asyncio.run(services.snapshot())
    names = {r.name for r in rows}
    assert {"SerpApi", "Tavily", "Firecrawl", "LLM"} <= names
    assert all("not configured" in r.line() for r in rows[:4])


def test_render_handles_an_empty_dashboard():
    assert "unavailable" in services.render([])


# --- reports once ----------------------------------------------------------

def test_balances_are_cached_between_runs(monkeypatch):
    """A second run must not re-query the provider."""
    calls = []

    def probe():
        calls.append(1)
        return {"balance": "5/10 left"}

    monkeypatch.setattr(services, "_PROBES",
                        (("x", "X", "purpose", "SVC_CACHE_KEY", probe),))
    monkeypatch.setenv("SVC_CACHE_KEY", "k")

    asyncio.run(services.snapshot())
    asyncio.run(services.snapshot())
    assert len(calls) == 1, "provider was queried more than once inside the TTL"


def test_a_failing_probe_is_cached_too(monkeypatch):
    """Retrying a dead provider every run would hammer it for no gain."""
    calls = []

    def probe():
        calls.append(1)
        raise RuntimeError("down")

    monkeypatch.setattr(services, "_PROBES",
                        (("x", "X", "purpose", "SVC_ERR_KEY", probe),))
    monkeypatch.setenv("SVC_ERR_KEY", "k")

    asyncio.run(services.snapshot())
    asyncio.run(services.snapshot())
    assert len(calls) == 1


def test_announce_prints_once_across_rapid_runs(monkeypatch, caplog):
    _no_keys(monkeypatch)
    import logging
    caplog.set_level(logging.INFO, logger="backend.research.services")

    async def run_three():
        for q in ("one", "two", "three"):
            await services.announce(q, min_gap_s=30.0)

    asyncio.run(run_three())
    printed = [r for r in caplog.records if "SERVICES IN USE" in r.getMessage()]
    assert len(printed) == 1, f"printed {len(printed)} times, expected 1"


def test_announce_prints_again_after_the_gap(monkeypatch, caplog):
    _no_keys(monkeypatch)
    import logging
    caplog.set_level(logging.INFO, logger="backend.research.services")

    async def run():
        await services.announce("first", min_gap_s=0.0)
        await services.announce("second", min_gap_s=0.0)

    asyncio.run(run())
    printed = [r for r in caplog.records if "SERVICES IN USE" in r.getMessage()]
    assert len(printed) == 2


def test_announce_still_returns_rows_when_gated(monkeypatch):
    """Gating the print must not gate the data the SSE event carries."""
    _no_keys(monkeypatch)

    async def run():
        first = await services.announce("one", min_gap_s=30.0)
        second = await services.announce("two", min_gap_s=30.0)
        return first, second

    first, second = asyncio.run(run())
    assert first and second


# --- honesty about what is knowable ---------------------------------------

def test_no_whole_api_key_is_ever_shown(monkeypatch):
    _no_keys(monkeypatch)
    monkeypatch.setenv("SERPAPI_KEY", "SUPERSECRETVALUE1234")
    monkeypatch.setattr(services, "_PROBES",
                        (("serpapi", "SerpApi", "purpose", "SERPAPI_KEY",
                          lambda: {"balance": "1/1 left"}),))
    rendered = services.render(asyncio.run(services.snapshot()))
    assert "SUPERSECRET" not in rendered
    assert "1234" in rendered


def test_llm_row_says_no_balance_api_rather_than_inventing_one(monkeypatch):
    """Free-tier model providers publish no balance endpoint. Say so."""
    _no_keys(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "gemini/gemini-3.5-flash-lite")
    monkeypatch.setenv("LLM_FALLBACKS", "groq/llama-3.3-70b-specdec")

    llm = [r for r in asyncio.run(services.snapshot()) if r.name == "LLM"][0]
    assert llm.balance == ""
    assert "no balance API" in llm.line()
    assert any("gemini" in d for d in llm.details)


def test_llm_status_does_not_import_litellm():
    """It runs before the run; a 7s cold import of litellm is not acceptable."""
    import subprocess
    import sys
    code = ("import sys;"
            "sys.path.insert(0, '/home/xe/Documents/deep-researcher');"
            "from backend.research import services;"
            "services._model_chain();"
            "print('LITELLM' if 'litellm' in sys.modules else 'CLEAN')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=120)
    assert "CLEAN" in out.stdout, out.stdout + out.stderr


def test_exhausted_serpapi_is_called_out(monkeypatch):
    monkeypatch.setattr(services, "_PROBES",
                        (("serpapi", "SerpApi", "purpose", "SERPAPI_KEY",
                          lambda: {"balance": "0/250 searches left",
                                   "detail": "Your account has run out of searches."}),))
    monkeypatch.setenv("SERPAPI_KEY", "k")
    row = asyncio.run(services.snapshot())[0]
    assert "0/250" in row.balance
    assert "out of searches" in row.detail


def test_openrouter_free_tier_does_not_show_negative_credits(monkeypatch):
    """Subtracting usage from a zero balance reads as a bug, so it is not done."""
    class _Resp:
        status_code = 200

        def json(self):
            return {"data": {"total_credits": 0, "total_usage": 0.035021408}}

    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    out = services._probe_openrouter()
    assert "-" not in out["balance"], "must not show negative credits"
    assert "0.0350 USD used" in out["balance"]
    assert "no purchased credits" in out["detail"], (
        "a zero balance with usage must be explained, not just printed")


def test_openrouter_with_credits_reports_the_remainder(monkeypatch):
    class _Resp:
        status_code = 200

        def json(self):
            return {"data": {"total_credits": 10.0, "total_usage": 2.5}}

    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    out = services._probe_openrouter()
    assert "7.50/10.00" in out["balance"]