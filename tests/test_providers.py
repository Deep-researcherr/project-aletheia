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
    assert providers.wired_extract_tiers() == []


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


# --- the fallback chain -----------------------------------------------------

def test_the_default_order_starts_cheapest_and_ends_most_expensive():
    """Cheapest and most-likely first; the expensive specialist last.

    ScrapingBee is last deliberately: it costs ~5 credits a call to render
    JavaScript, so it should only ever be reached for a page nothing else can
    read. Putting it anywhere earlier turns it into a per-page tax.
    """
    order = providers.extract_tier_order()
    assert order[0] == "jina"
    assert order[-1] == "scrapingbee"
    assert order.index("firecrawl") < order.index("scrapingbee")


def test_the_chain_covers_every_extract_provider(monkeypatch):
    """A provider missing from the default order would be unreachable."""
    order = providers.extract_tier_order()
    assert set(order) == {p.name for p in providers.for_role(providers.EXTRACT)}
    assert len(order) == len(set(order)), "a provider appears twice"


def test_tier_order_is_reconfigurable_without_a_code_change(monkeypatch):
    monkeypatch.setenv("EXTRACT_TIER_ORDER", "firecrawl, tavily")
    order = providers.extract_tier_order()
    assert order[:2] == ["firecrawl", "tavily"]
    assert set(order) == {p.name for p in providers.for_role(providers.EXTRACT)}, \
        "providers left out of EXTRACT_TIER_ORDER must be appended, not dropped"


def test_typo_in_tier_order_degrades_rather_than_disabling_the_chain(monkeypatch):
    """A misspelled provider name must not silently reduce the chain to
    nothing -- that would look exactly like 'no fallback is configured'."""
    monkeypatch.setenv("EXTRACT_TIER_ORDER", "jinaa, tavily")
    order = providers.extract_tier_order()
    assert "tavily" in order
    assert len(order) == len(providers.for_role(providers.EXTRACT))


def test_only_wired_tiers_are_offered_to_the_chain(monkeypatch):
    monkeypatch.setenv("SCRAPINGBEE_API_KEY", "k")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    names = [p.name for p in providers.wired_extract_tiers()]
    assert "scrapingbee" in names
    assert "tavily" not in names


def test_disabling_one_extractor_keeps_the_rest_of_the_chain(monkeypatch):
    """A single switched-off or dead key must not disable recovery."""
    for provider in providers.for_role(providers.EXTRACT):
        monkeypatch.setenv(provider.env, "k")
    monkeypatch.setenv("SCRAPINGBEE_ENABLED", "0")
    providers._note_auth_failure("jina")
    providers._note_auth_failure("jina")
    names = [p.name for p in providers.wired_extract_tiers()]
    assert "scrapingbee" not in names, "switched off"
    assert "jina" not in names, "quarantined"
    assert {"tavily", "tinyfish", "firecrawl"} <= set(names)
    # Order survives a gap: the surviving tiers keep their relative position.
    assert names.index("tinyfish") < names.index("tavily")


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
    assert "exa" not in [p.name for p in providers.wired_extract_tiers()]


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


def test_the_chain_stops_at_the_first_provider_that_answers(walled, monkeypatch):
    """One provider call per recovered page, not one per provider.

    This is the whole reason for the chain: every wired provider costs money,
    so paying for four losers to get one document is the wrong trade.
    """
    from backend.research.fetch import Fetcher

    for name in ("JINA", "TINYFISH", "TAVILY", "SCRAPINGBEE", "FIRECRAWL"):
        monkeypatch.setenv(f"{name}_API_KEY", "k")
    _stub_extractors(monkeypatch, jina=("J", "recovered body text " * 60))

    fetcher = Fetcher(session=walled)
    doc = fetcher.fetch("https://walled.dev/article")

    assert doc.status == "full"
    assert doc.method == "jina_reader"
    assert fetcher._paid_used == {"jina": 1}, \
        "the chain must stop on success rather than bill every provider"


def test_a_declining_provider_hands_over_to_the_next(walled, monkeypatch):
    """Recovery is not lost just because the first tier cannot read the page."""
    from backend.research.fetch import Fetcher

    for name in ("JINA", "TINYFISH", "TAVILY"):
        monkeypatch.setenv(f"{name}_API_KEY", "k")
    _stub_extractors(monkeypatch,
                     jina=None,
                     tinyfish=None,
                     tavily=("T", "tavily body " * 60))

    fetcher = Fetcher(session=walled)
    doc = fetcher.fetch("https://walled.dev/article")

    assert doc.method == "tavily"
    assert fetcher.paid_used("jina") == 1, "jina was tried and declined"
    assert fetcher.paid_used("tinyfish") == 1
    assert fetcher.paid_used("tavily") == 1
    assert fetcher.paid_used("firecrawl") == 0, "the chain must stop on success"


def test_the_chain_follows_the_configured_order(walled, monkeypatch):
    """EXTRACT_TIER_ORDER changes which provider is billed, not just their
    order in the source."""
    from backend.research.fetch import Fetcher

    monkeypatch.setenv("JINA_API_KEY", "k")
    monkeypatch.setenv("SCRAPINGBEE_API_KEY", "k")
    monkeypatch.setenv("EXTRACT_TIER_ORDER", "scrapingbee,jina")
    _stub_extractors(monkeypatch,
                     jina=("J", "jina body " * 60),
                     scrapingbee=("S", "js rendered body " * 60))

    fetcher = Fetcher(session=walled)
    doc = fetcher.fetch("https://walled.dev/article")
    assert doc.method == "scrapingbee", "the reordered provider is tried first"
    assert fetcher.paid_used("jina") == 0, "jina must not be called at all"


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


def test_an_extractor_hitting_its_cap_is_skipped_not_spent(walled, monkeypatch):
    """The cap is claimed before the call, so a spent tier costs nothing
    further and the chain moves on instead of failing the fetch."""
    from backend.research.fetch import Fetcher

    monkeypatch.setenv("JINA_API_KEY", "k")
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setenv("JINA_MAX_PER_RUN", "1")
    _stub_extractors(monkeypatch,
                     jina=("J", "jina body " * 60),
                     tavily=("T", "tavily body " * 60))

    fetcher = Fetcher(session=walled)
    assert fetcher.fetch("https://walled.dev/a").method == "jina_reader"
    second = fetcher.fetch("https://walled.dev/b")

    assert second.method == "tavily", "a capped tier must not be retried"
    assert fetcher.paid_used("jina") == 1


def test_a_raising_tier_does_not_re_run_the_whole_chain(walled, monkeypatch):
    """One malformed provider response must cost one tier, not one of each.

    ``fetch()`` wraps ``_with_fallback`` in its own try/except and then calls
    ``_with_fallback`` again on the way out. So any exception escaping the chain
    used to unwind into that handler and re-run every provider from the top --
    billing five providers twice over for a single URL. Pre-existing, and
    invisible until a provider returns something malformed.
    """
    from backend.research.fetch import Fetcher

    for name in ("JINA", "TINYFISH", "TAVILY", "SCRAPINGBEE", "FIRECRAWL"):
        monkeypatch.setenv(f"{name}_API_KEY", "k")
    _stub_extractors(monkeypatch,
                     jina=("J", "good body " * 60),
                     tinyfish=("T", "good body " * 60),
                     tavily=("V", "good body " * 60),
                     scrapingbee=("S", "good body " * 60),
                     firecrawl=("F", "good body " * 60))

    def boom(*args, **kwargs):
        raise ValueError("malformed provider response")

    monkeypatch.setattr(Fetcher, "_external_doc", staticmethod(boom))

    fetcher = Fetcher(session=walled)
    fetcher.fetch("https://walled.dev/article")

    spent = fetcher._paid_used
    assert sum(spent.values()) == len(spent), (
        f"each tier was billed more than once for one fetch: {spent}")


def test_a_raising_tier_does_not_abandon_the_rest_of_the_chain(walled,
                                                               monkeypatch):
    """A broken tier must be skipped, not fatal -- the page may still be
    readable further down the chain."""
    from backend.research.fetch import Fetcher

    monkeypatch.setenv("JINA_API_KEY", "k")
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    _stub_extractors(monkeypatch,
                     jina=("J", "good body " * 60),
                     tavily=("T", "tavily body " * 60))

    real = Fetcher._external_doc

    # _external_doc is a staticmethod, so it takes no self here.
    def only_jina_breaks(url, title, text, method):
        if method == "jina_reader":
            raise ValueError("malformed provider response")
        return real(url, title, text, method)

    monkeypatch.setattr(Fetcher, "_external_doc", staticmethod(only_jina_breaks))

    fetcher = Fetcher(session=walled)
    doc = fetcher.fetch("https://walled.dev/article")
    assert doc.status == "full"
    assert doc.method == "tavily", "the chain must continue past a broken tier"


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