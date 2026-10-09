"""Tests for the provider registry.

The registry exists so that wiring a source provider is one entry rather than
three edits in three files that have to stay in step. These tests therefore
check the *properties* that make that safe -- that every consumer reads the
registry, that a missing key means absent rather than broken, and that a
provider which raises cannot take a run down -- rather than pinning any single
provider's request shape, which would make adding the next one a test rewrite.
"""
from pathlib import Path
import time

import pytest

from backend.research import providers, config

REPO = Path(__file__).resolve().parents[1]


def _provider(name, role):
    return next(p for p in providers.for_role(role) if p.name == name)


# --- registry shape ---------------------------------------------------------

def test_every_provider_declares_a_key_variable():
    for provider in providers.all_providers():
        assert provider.env, f"{provider.name} has no API key variable"
        assert provider.purpose, f"{provider.name} has no purpose for the panel"
        assert provider.meter, f"{provider.name} does not say how it is metered"


def test_search_providers_are_keyed_by_the_source_the_planner_emits():
    """A provider whose ``source`` is empty would be unreachable: the planner
    could never route to it and the result label would have nothing to key on."""
    for name, provider in providers.search_providers().items():
        assert provider.source == name
        assert provider.source_label, f"{name} would print as 'web page'"


def test_search_and_extract_labels_do_not_collide():
    """Search results and fetch methods share one label table.

    A provider with both roles and no distinct method name would make 'found
    via Jina' and 'read via Jina' the same string in the evidence legend, which
    reads as a claim about provenance that is only half true.
    """
    from backend.research.evidence import source_type_label

    search_names = set(providers.search_providers())
    for provider in providers.for_role(providers.EXTRACT):
        assert provider.method_label, \
            f"{provider.method_name} would fall back to the generic 'web page'"
        if provider.name in search_names:
            assert provider.method_name != provider.name, \
                f"{provider.name} is used for both searching and fetching; " \
                "its method name must distinguish them"
            assert (source_type_label(provider.method_name)
                    != source_type_label(provider.name)), \
                f"{provider.name}: 'found via' and 'read via' print identically"


def test_planner_offers_every_registered_search_source():
    from backend.research.planner import SOURCES

    for name in providers.search_providers():
        assert name in SOURCES, \
            f"{name} is registered but the planner cannot route to it, so its " \
            "budget would be spent returning nothing"


def test_evidence_legend_covers_every_registered_provider():
    from backend.research.evidence import _SOURCE_TYPE_LABELS

    for provider in providers.for_role(providers.SEARCH):
        assert provider.source in _SOURCE_TYPE_LABELS
    for provider in providers.for_role(providers.EXTRACT):
        assert provider.method_name in _SOURCE_TYPE_LABELS


# --- key handling -----------------------------------------------------------

def test_provider_without_a_key_is_absent_not_broken(monkeypatch):
    """A missing key must make a provider skip cleanly, never raise.

    The alternative -- treating 'no key' as an error -- turns an unconfigured
    optional provider into a failed sub-question.
    """
    for provider in providers.all_providers():
        monkeypatch.delenv(provider.env, raising=False)
        assert not provider.has_key
        assert not provider.wired
    provider = providers.search_providers()["exa"]
    assert provider.run_search("anything", 3) == ([], {})
    assert providers.wired_search_providers() == {}
    assert providers.wired_extract_providers() == []


def test_a_provider_can_be_switched_off_without_deleting_its_key(monkeypatch):
    provider = providers.search_providers()["exa"]
    monkeypatch.setenv(provider.env, "a-key")
    assert provider.wired
    monkeypatch.setenv("EXA_ENABLED", "0")
    assert not provider.wired
    for truthy in ("1", "true", "on", "yes"):
        monkeypatch.setenv("EXA_ENABLED", truthy)
        assert provider.wired, f"EXA_ENABLED={truthy} should leave it on"


def test_a_raising_provider_cannot_take_the_run_down(monkeypatch):
    """Every call site is wrapped. One bad provider must not abort a run."""
    provider = providers.search_providers()["exa"]
    monkeypatch.setenv(provider.env, "a-key")

    def boom(query, n):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(provider, "call", boom, raising=False)
    assert provider.run_search("q", 3) == ([], {})


# --- no fallback ordering ---------------------------------------------------

def test_extractors_are_not_ordered(monkeypatch):
    """There is no tier order any more, and there must not grow one back.

    A ranked chain caps recovery at the best single provider and spends the
    run's latency walking the ones that have already declined. Racing makes the
    ceiling the union of all of them.
    """
    assert not hasattr(providers, "extract_tiers")
    assert not hasattr(providers, "extract_tier_order")
    assert not hasattr(providers, "DEFAULT_EXTRACT_ORDER")
    for provider in providers.all_providers():
        assert "order" not in provider.__dict__, \
            "a provider carrying its own position is a tier order returning"


def test_every_wired_extractor_is_available_to_the_race(monkeypatch):
    """The race asks all of them, so all of them must be reachable."""
    for provider in providers.for_role(providers.EXTRACT):
        monkeypatch.setenv(provider.env, "k")
    names = {p.name for p in providers.wired_extract_providers()}
    assert names == {p.name for p in providers.for_role(providers.EXTRACT)}


def test_an_unwired_extractor_is_excluded_from_the_race(monkeypatch):
    monkeypatch.setenv("SCRAPINGBEE_API_KEY", "k")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    names = {p.name for p in providers.wired_extract_providers()}
    assert "scrapingbee" in names
    assert "tavily" not in names


def test_disabling_one_extractor_keeps_the_others(monkeypatch):
    """A single switched-off or dead key must not disable the race."""
    for provider in providers.for_role(providers.EXTRACT):
        monkeypatch.setenv(provider.env, "k")
    monkeypatch.setenv("SCRAPINGBEE_ENABLED", "0")
    providers._note_auth_failure("jina")
    providers._note_auth_failure("jina")
    names = {p.name for p in providers.wired_extract_providers()}
    assert "scrapingbee" not in names, "switched off"
    assert "jina" not in names, "quarantined"
    assert {"tavily", "tinyfish", "firecrawl"} <= names


# --- per-run caps -----------------------------------------------------------

def test_cap_comes_from_the_registry_and_the_env_overrides_it(monkeypatch):
    provider = _provider("scrapingbee", providers.EXTRACT)
    monkeypatch.delenv("SCRAPINGBEE_MAX_PER_RUN", raising=False)
    assert config.provider_max_per_run("scrapingbee") == provider.cap_default
    monkeypatch.setenv("SCRAPINGBEE_MAX_PER_RUN", "7")
    assert config.provider_max_per_run("scrapingbee") == 7


def test_legacy_cap_variables_still_work(monkeypatch):
    """TAVILY_MAX_PER_RUN and FIRECRAWL_MAX_PER_RUN predate the registry."""
    monkeypatch.setenv("TAVILY_MAX_PER_RUN", "3")
    monkeypatch.setenv("FIRECRAWL_MAX_PER_RUN", "2")
    assert config.tavily_max_per_run() == 3
    assert config.firecrawl_max_per_run() == 2


def test_zero_cap_disables_a_tier():
    for provider in providers.for_role(providers.EXTRACT):
        assert provider.cap_default > 0, \
            "a tier that is on by default must have a budget"


def test_config_falls_back_sanely_for_an_unknown_provider():
    assert config.provider_max_per_run("not-a-provider") == 10


# --- dead-key quarantine ----------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_quarantine():
    providers._auth_state.clear()
    yield
    providers._auth_state.clear()


def test_two_rejected_keys_pause_a_provider_and_then_it_is_skipped():
    """A wrong key must not be retried once per fetch for the whole run.

    Each retry spends one unit of that provider's per-run cap, so a single
    mistyped variable would quietly consume the run's budget for a provider
    that cannot answer, tier by tier, with nothing in the report to say so.
    """
    providers._note_auth_failure("exa")
    assert not providers.quarantined("exa"), "one failure is not proof"
    providers._note_auth_failure("exa")
    assert providers.quarantined("exa")
    provider = providers.search_providers()["exa"]
    assert not provider.wired
    assert "exa" not in providers.wired_search_providers()
    assert "exa" not in [p.name for p in providers.wired_extract_providers()]


def test_a_success_clears_the_quarantine(monkeypatch):
    """So a transient blip does not disable a good key for the full window."""
    providers._note_auth_failure("exa")
    providers._note_auth_failure("exa")
    assert providers.quarantined("exa")
    monkeypatch.setenv("EXA_API_KEY", "good-key")
    providers._note_auth_success("exa")
    assert not providers.quarantined("exa")
    assert providers.search_providers()["exa"].wired


def test_quarantine_is_reported_rather_than_hidden():
    """The services panel must be able to say *why* a provider is not being
    used, or a paused provider looks exactly like an unconfigured one."""
    from backend.research import services

    providers._note_auth_failure("exa")
    providers._note_auth_failure("exa")
    row = next(r for r in services._registry_rows() if r.name == "exa")
    assert row.configured is False
    assert "PAUSED" in row.detail
    assert "EXA_API_KEY" not in row.detail or row.error


def test_quarantine_expires(monkeypatch):
    """A key fixed in .env must come back without restarting the server."""
    monkeypatch.setattr(providers, "_AUTH_QUARANTINE_S", 0.0)
    providers._note_auth_failure("exa")
    providers._note_auth_failure("exa")
    assert not providers.quarantined("exa"), "the window should have lapsed"


# --- fetcher integration ----------------------------------------------------

class _Walled:
    """A response that fails every direct fetch."""
    status_code = 403
    url = "https://example.com/"
    headers = {"content-type": "text/html"}
    _body = b""
    _truncated = False

    def raise_for_status(self):
        pass

    def iter_content(self, n):
        return iter(())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _DeadSession:
    headers: dict = {}

    def get(self, url, **kw):
        return _Walled()


@pytest.fixture
def walled(monkeypatch):
    monkeypatch.setattr(config, "fetch_respect_robots", lambda: False)
    monkeypatch.setattr("backend.research.oa.resolve_pdf_url", lambda url: "")
    return _DeadSession()


def _stub_extractors(monkeypatch, **results):
    """Point every extract tier at a canned (title, text) result, or None.

    Any tier not named in ``results`` declines, which is what makes
    'the earlier tier gave up' observable.
    """
    for provider in providers.for_role(providers.EXTRACT):
        monkeypatch.setattr(provider, "call",
                            (lambda url, _r=results.get(provider.name): _r),
                            raising=False)


def test_every_wired_extractor_is_asked_not_just_the_first(walled, monkeypatch):
    """The race fires them all, so one declining provider must not stop the
    others from being tried."""
    from backend.research.fetch import Fetcher

    for name in ("JINA", "TINYFISH", "TAVILY", "SCRAPINGBEE", "FIRECRAWL"):
        monkeypatch.setenv(f"{name}_API_KEY", "k")
    _stub_extractors(monkeypatch, tavily=("T", "recovered body text " * 60))

    fetcher = Fetcher(session=walled)
    doc = fetcher.fetch("https://walled.dev/article")

    assert doc.status == "full"
    assert doc.method == "tavily"
    spent = fetcher._paid_used
    for name in ("jina", "tinyfish", "tavily", "scrapingbee", "firecrawl"):
        assert spent.get(name) == 1, (
            f"{name} was not raced; the point is that all of them are asked")


def test_the_race_answers_from_whichever_provider_comes_back_first(
        walled, monkeypatch):
    """No provider is preferred. Whichever produces usable text first wins."""
    from backend.research.fetch import Fetcher

    monkeypatch.setenv("JINA_API_KEY", "k")
    monkeypatch.setenv("SCRAPINGBEE_API_KEY", "k")
    _stub_extractors(monkeypatch,
                     jina=("J", "jina body " * 60),
                     scrapingbee=("S", "js rendered body " * 60))

    fetcher = Fetcher(session=walled)
    doc = fetcher.fetch("https://walled.dev/article")
    assert doc.status == "full"
    assert doc.method in ("jina_reader", "scrapingbee")
    assert fetcher.paid_used("jina") == 1
    assert fetcher.paid_used("scrapingbee") == 1


def test_a_thin_answer_does_not_win_over_a_richer_sibling(walled, monkeypatch):
    """The grace window exists for exactly this.

    Racing and taking the first arrival means a provider that answers fast with
    a consent wall beats one that answers a second later with the whole
    article. Every provider in the race is billed either way, so listening
    briefly costs only wall-clock.
    """
    from backend.research.fetch import Fetcher

    monkeypatch.setenv("JINA_API_KEY", "k")
    monkeypatch.setenv("TINYFISH_API_KEY", "k")
    monkeypatch.setattr(config, "extract_grace_s", lambda: 3.0)

    def fast_then_thin(url):
        return ("J", "enable javascript to continue " * 5)

    def slower_but_real(url):
        time.sleep(0.4)
        return ("T", "the full article body " * 60)

    for provider in providers.for_role(providers.EXTRACT):
        call = (slower_but_real if provider.name == "tinyfish" else fast_then_thin)
        monkeypatch.setattr(provider, "call", call, raising=False)

    fetcher = Fetcher(session=walled)
    doc = fetcher.fetch("https://walled.dev/article")
    assert doc.method == "tinyfish_fetch"
    assert len(doc.text.split()) > 100


def test_the_grace_window_is_bounded_by_config(walled, monkeypatch):
    """A slow provider must not be able to hold the run open indefinitely."""
    import time as _time

    from backend.research.fetch import Fetcher

    monkeypatch.setenv("JINA_API_KEY", "k")
    monkeypatch.setenv("TINYFISH_API_KEY", "k")
    monkeypatch.setattr(config, "extract_grace_s", lambda: 0.2)
    monkeypatch.setattr(config, "fetch_timeout_s", lambda: 10.0)

    def slow(url):
        _time.sleep(6)
        return ("S", "slow but large " * 200)

    for provider in providers.for_role(providers.EXTRACT):
        # One provider answers immediately and opens the grace window; the
        # others are still running when it closes.
        # Bind the name as a default argument: a closure over the loop variable
        # would be evaluated at call time, by which point every provider would
        # see the last iteration's name and get the same branch.
        monkeypatch.setattr(
            provider, "call",
            (lambda url, _name=provider.name:
             ("J", "short answer " * 40) if _name == "jina" else slow(url)),
            raising=False)

    fetcher = Fetcher(session=walled)
    start = _time.monotonic()
    doc = fetcher.fetch("https://walled.dev/article")
    elapsed = _time.monotonic() - start
    assert elapsed < 4, f"the race waited {elapsed:.1f}s for stragglers"
    assert doc.status == "full"
    assert doc.method == "jina_reader"


def test_a_provider_returning_nothing_does_not_beat_one_that_answers(
        walled, monkeypatch):
    from backend.research.fetch import Fetcher

    monkeypatch.setenv("JINA_API_KEY", "k")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "k")
    _stub_extractors(monkeypatch,
                     jina=None,
                     firecrawl=("F", "firecrawl body " * 60))

    fetcher = Fetcher(session=walled)
    doc = fetcher.fetch("https://walled.dev/article")
    assert doc.method == "firecrawl"
    assert "firecrawl body" in doc.text


def test_thin_text_is_not_accepted_as_a_recovery(walled, monkeypatch):
    """A provider that answers with a stub must not beat silence: a 3-word
    'enable javascript' page is worse than an honest failure."""
    from backend.research.fetch import Fetcher

    monkeypatch.setenv("JINA_API_KEY", "k")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "k")
    _stub_extractors(monkeypatch,
                     jina=("J", "please enable javascript"),
                     firecrawl=("F", "firecrawl body " * 60))

    fetcher = Fetcher(session=walled)
    doc = fetcher.fetch("https://walled.dev/article")
    assert doc.method == "firecrawl"


def test_an_extractor_hitting_its_cap_is_left_out_of_the_race(walled,
                                                              monkeypatch):
    """Caps are claimed before the call, so a spent provider costs nothing
    further and simply does not enter the race."""
    from backend.research.fetch import Fetcher

    monkeypatch.setenv("JINA_API_KEY", "k")
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setenv("JINA_MAX_PER_RUN", "1")
    _stub_extractors(monkeypatch,
                     jina=("J", "jina body " * 60),
                     tavily=("T", "tavily body " * 60))

    fetcher = Fetcher(session=walled)
    fetcher.fetch("https://walled.dev/a")
    first_spend = dict(fetcher._paid_used)
    fetcher.fetch("https://walled.dev/b")

    assert fetcher.paid_used("jina") == first_spend["jina"] == 1, \
        "a capped provider must not be charged again"
    assert fetcher.paid_used("tavily") == 2


def test_no_wired_extractor_leaves_the_failure_visible(walled, monkeypatch):
    """Recovery failing must not be laundered into a success."""
    from backend.research.fetch import Fetcher

    _stub_extractors(monkeypatch)
    fetcher = Fetcher(session=walled)
    doc = fetcher.fetch("https://walled.dev/article")
    assert doc.status == "failed"
    assert fetcher._paid_used == {}


def test_named_counters_still_reflect_the_registry_spend(walled, monkeypatch):
    """tavily_used / firecrawl_used are read by the run report."""
    from backend.research.fetch import Fetcher

    monkeypatch.setenv("TAVILY_API_KEY", "k")
    _stub_extractors(monkeypatch, tavily=("T", "body " * 60))

    fetcher = Fetcher(session=walled)
    fetcher.fetch("https://walled.dev/a")
    assert fetcher.tavily_used == 1
    assert fetcher.firecrawl_used == 0


# --- searcher integration ---------------------------------------------------

def test_a_registered_source_dispatches_to_its_provider(monkeypatch):
    from backend.research import searcher

    monkeypatch.setenv("SEARCH_CACHE_DIR", "")
    monkeypatch.setenv("EXA_API_KEY", "k")
    provider = providers.search_providers()["exa"]

    def fake(query, n):
        return [providers.make_item("T", "https://example.com/a", "s", "2026-01-01",
                                    "exa", query)], {}

    monkeypatch.setattr(provider, "call", fake, raising=False)
    rows = searcher.search("exa", "a query", 3)
    assert len(rows) == 1
    assert rows[0]["url"] == "https://example.com/a"
    assert rows[0]["type"] == "exa"


def test_a_registry_source_never_falls_through_to_google(monkeypatch):
    """If a provider is unwired, the sub-question finds nothing.

    Falling through to a SerpApi engine and returning its results labelled as
    the provider's would be worse than empty: the report would attribute links
    to an engine that never ran.
    """
    from backend.research import searcher

    monkeypatch.setenv("SEARCH_CACHE_DIR", "")
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    def explode(*args, **kwargs):
        raise AssertionError("SerpApi must not be consulted for 'exa'")

    monkeypatch.setattr("serpapi.GoogleSearch", explode, raising=False)
    assert searcher.search("exa", "a query", 3) == []


def test_provider_results_are_cached_so_a_rerun_does_not_repay(monkeypatch,
                                                               tmp_path):
    from backend.research import searcher

    monkeypatch.setenv("SEARCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("EXA_API_KEY", "k")
    provider = providers.search_providers()["exa"]
    calls = []

    def counted(query, n):
        calls.append(query)
        return [providers.make_item("T", "https://example.com/a", "s", "",
                                    "exa", query)], {}

    monkeypatch.setattr(provider, "call", counted, raising=False)
    searcher.search("exa", "a query", 3)
    searcher.search("exa", "a query", 3)
    assert len(calls) == 1, "the second identical query must be served from disk"


def test_an_unreadable_cache_falls_back_to_a_live_call(monkeypatch, tmp_path):
    """A truncated hand-edited cache file must not poison every later run."""
    from backend.research import searcher

    cache = tmp_path / "cache"
    monkeypatch.setenv("SEARCH_CACHE_DIR", str(cache))
    monkeypatch.setenv("EXA_API_KEY", "k")
    provider = providers.search_providers()["exa"]

    calls = []

    def counted(query, n):
        calls.append(query)
        return [providers.make_item("T", "https://example.com/a", "s", "",
                                    "exa", query)], {}

    monkeypatch.setattr(provider, "call", counted, raising=False)
    searcher.search("exa", "a query", 3)          # populates the cache
    for path in cache.glob("*.json"):
        path.write_text("{truncated")
    searcher.search("exa", "a query", 3)
    assert len(calls) == 2, "a corrupt cache entry must not be trusted"


def test_serpapi_sources_are_not_treated_as_registry_sources(monkeypatch):
    from backend.research import searcher

    monkeypatch.setenv("SEARCH_CACHE_DIR", "")
    monkeypatch.setenv("EXA_API_KEY", "k")
    assert searcher._registry_search("web", "q", 3) is None, \
        "'web' is a SerpApi engine and must not resolve through the registry"
    assert searcher._registry_search("arxiv", "q", 3) is None


def test_rows_from_every_source_share_one_shape(monkeypatch):
    """The collector, relevance ranker and PDF read these keys positionally."""
    from backend.research import searcher

    monkeypatch.setenv("SEARCH_CACHE_DIR", "")
    monkeypatch.setenv("EXA_API_KEY", "k")
    provider = providers.search_providers()["exa"]
    monkeypatch.setattr(
        provider, "call",
        lambda q, n: ([providers.make_item("T", "https://example.com/a", "s",
                                           "", "exa", q)], {}),
        raising=False)
    row = searcher.search("exa", "q", 3)[0]
    for key in ("title", "url", "snippet", "date", "type", "query"):
        assert key in row, f"missing {key}"


# --- services panel ---------------------------------------------------------

def test_the_panel_lists_every_registered_provider():
    from backend.research import services

    names = {row.name for row in services._registry_rows()}
    for provider in providers.all_providers():
        assert provider.name in names, \
            f"{provider.name} is registered but absent from the run-start panel"


def test_the_panel_does_not_duplicate_probed_providers():
    """Tavily and Firecrawl already appear with live balances; two rows for one
    account would read as two."""
    from backend.research import services

    probed = {name.lower() for _k, name, _p, _e, _pr in services._PROBES}
    rows = services._rows_from({})
    names = [row.name.lower() for row in rows]
    assert len(names) == len(set(names)), f"duplicate service rows: {names}"
    assert probed <= set(names)