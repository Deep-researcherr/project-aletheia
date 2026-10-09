"""Declarative registry of external source providers.

Two roles, and every provider declares which one it fills:

    search    (query, n) -> (rows, meta)     rows are normalized result items
    extract   (url)     -> (title, text)     a fetch-fallback tier

The point of this module is that the registry is the *only* place a provider
has to be named. ``searcher.py`` dispatches search sources through it,
``fetch.py`` walks the extract tier chain from it, and ``services.py`` builds
its dashboard from it. Before this existed, each of those three branched on the
provider name directly, so wiring one new API meant editing three call sites
and hoping all three stayed in step -- and the dashboard in particular silently
under-reported, because a provider that nobody remembered to add to the probe
list just looked like a provider that did not exist.

Adding a provider is therefore one ``Provider`` entry below.

Rules every entry obeys:

  * a missing API key makes the provider *absent*, not broken -- callers skip it
    silently rather than treating it as an error;
  * no provider call raises. A provider that times out, 401s, or returns
    nonsense yields ``None`` / ``[]`` and a warning, because one flaky upstream
    must never take down a run;
  * responses are only trusted after the row carries a usable URL and enough
    text to be worth fetching.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)

SEARCH = "search"
EXTRACT = "extract"

# Sources that mean "the open web" and are therefore interchangeable: a query
# routed to one of these can be asked of all of them at once, and the union is
# strictly more evidence than any single engine. Sources that name a corpus
# (scholar, patent, github, arxiv, news) are NOT in here -- fanning a patent
# query across a news index would throw away the thing that made it a patent
# query.
GENERAL_SOURCES = ("web",)


def _timeout(default: float = 45.0) -> float:
    """Per-call ceiling, from FETCH_TIMEOUT_S when it is set sensibly."""
    try:
        from . import config as _config
        return max(default, _config.fetch_timeout_s())
    except Exception:
        return default


def make_item(title, url, snippet, date, kind, query, extra=None) -> dict:
    """Build one normalized search row.

    Lives here rather than in searcher so the registry does not have to import
    searcher; searcher delegates its own ``_item`` to this.
    """
    row = {"title": title or "", "url": url, "snippet": snippet or "",
           "date": date, "type": kind, "query": query}
    if extra:
        row.update(extra)
    return row


# --- shared helpers ---------------------------------------------------------

def _json(resp) -> dict:
    """Decode a JSON body defensively; an HTML error page must not explode."""
    try:
        data = resp.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _clean_text(text: str) -> str:
    """Strip provider response headers and collapse whitespace.

    Jina's reader returns a preamble (``Title:``/``URL Source:``/``Warning:``)
    above the page body, and ScrapingBee can return a stub for a soft block.
    Neither should reach the chunker, so anything too thin is treated as no
    result at all.
    """
    from .fetch import _clean
    return _clean(text or "")


def _key(name: str) -> str:
    return (os.environ.get(name) or "").strip()


# --- dead-key quarantine ---------------------------------------------------
#
# A 401/403 means the key is wrong, not that the provider is briefly unhappy.
# Without this, every subsequent fetch in a long run re-asks a provider that
# cannot possibly answer, spending one unit of its per-run cap each time and
# turning one misconfigured variable into a run that quietly degrades tier by
# tier. Two consecutive auth failures quarantine the provider for a while; any
# success clears it immediately, so a transient blip does not disable a good
# key for longer than it lasts.
_AUTH_QUARANTINE_S = 600.0
_AUTH_FAILURES_BEFORE_QUARANTINE = 2
_auth_lock = threading.Lock()
_auth_state: dict[str, list] = {}      # name -> [consecutive failures, until_ts]


def _note_auth_failure(name: str) -> None:
    now = time.monotonic()
    with _auth_lock:
        entry = _auth_state.setdefault(name, [0, 0.0])
        entry[0] += 1
        if entry[0] >= _AUTH_FAILURES_BEFORE_QUARANTINE and entry[1] <= now:
            entry[1] = now + _AUTH_QUARANTINE_S
            log.error(
                "%s rejected its key %d times in a row; skipping it for %.0fs "
                "(check %s -- the run continues without it)",
                name, entry[0], _AUTH_QUARANTINE_S,
                next((p.env for p in _REGISTRY if p.name == name), "the API key"))


def _note_auth_success(name: str) -> None:
    with _auth_lock:
        _auth_state.pop(name, None)


def quarantined(name: str) -> bool:
    """True while a provider is set aside after repeated key rejections."""
    with _auth_lock:
        entry = _auth_state.get(name)
        return bool(entry) and entry[1] > time.monotonic()


def quarantine_detail(name: str) -> str:
    """Human-readable reason a provider is being skipped, or ''."""
    with _auth_lock:
        entry = _auth_state.get(name)
        if not entry or entry[1] <= time.monotonic():
            return ""
        return f"key rejected {entry[0]}x, paused {entry[1] - time.monotonic():.0f}s"


# --- search providers -------------------------------------------------------

def search_exa(query: str, n: int) -> tuple[list[dict], dict]:
    """Exa neural/semantic search -- genuinely different ranking from Google.

    Exa matches meaning rather than keywords, so it surfaces pages a
    keyword engine misses for a long natural-language sub-question. That is
    why it is a separate source and not another SerpApi engine.
    """
    key = _key("EXA_API_KEY")
    if not key or not query.strip():
        return [], {}
    import requests
    try:
        resp = requests.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json={
                "query": query,
                "numResults": max(1, n),
                # Highlights only: a short verbatim extract for the result
                # row. Full page text is deliberately not requested here --
                # content belongs to the fetch layer, which has robots rules,
                # byte caps and an open-access fallback the search layer does
                # not. Fetching it twice would pay twice and bypass both.
                "contents": {"highlights": {"numSentences": 2}},
            },
            timeout=_timeout(),
        )
        if resp.status_code in (401, 403):
            log.warning("exa: key rejected (HTTP %s)", resp.status_code)
            _note_auth_failure("exa")
            return [], {}
        if resp.status_code != 200:
            log.warning("exa: HTTP %s", resp.status_code)
            return [], {}
        data = _json(resp)
    except Exception as exc:
        log.warning("exa search failed: %s: %s", type(exc).__name__, exc)
        return [], {}

    rows = []
    for entry in (data.get("results") or [])[:n]:
        if not isinstance(entry, dict):
            continue
        url = (entry.get("url") or "").strip()
        if not url:
            continue
        highlights = entry.get("highlights") or []
        snippet = " ".join(h.strip() for h in highlights if h and h.strip())[:1200]
        published = (entry.get("publishedDate") or "")[:10]
        rows.append(make_item(
            entry.get("title"), url, snippet, published, "exa", query,
            {"exa_id": entry.get("id") or "",
             "author": entry.get("author") or ""}))
    _note_auth_success("exa")
    return rows, {}


def search_tinyfish(query: str, n: int) -> tuple[list[dict], dict]:
    """TinyFish ranked web search. Free to ~12k calls/day."""
    key = _key("TINYFISH_API_KEY")
    if not key or not query.strip():
        return [], {}
    import requests
    try:
        resp = requests.get(
            "https://api.search.tinyfish.ai",
            headers={"X-API-Key": key},
            params={"query": query, "domain_type": "web", "language": "en"},
            timeout=_timeout(),
        )
        if resp.status_code in (401, 403):
            log.warning("tinyfish: key rejected (HTTP %s)", resp.status_code)
            _note_auth_failure("tinyfish")
            return [], {}
        if resp.status_code != 200:
            log.warning("tinyfish: HTTP %s", resp.status_code)
            return [], {}
        data = _json(resp)
    except Exception as exc:
        log.warning("tinyfish search failed: %s: %s", type(exc).__name__, exc)
        return [], {}

    rows = []
    for entry in (data.get("results") or [])[:n]:
        if not isinstance(entry, dict):
            continue
        url = (entry.get("url") or "").strip()
        if not url:
            continue
        rows.append(make_item(
            entry.get("title"), url, entry.get("snippet"),
            entry.get("date"), "tinyfish", query,
            {"site_name": entry.get("site_name") or ""}))
    _note_auth_success("tinyfish")
    return rows, {}


def search_jina(query: str, n: int) -> tuple[list[dict], dict]:
    """Jina search (s.jina.ai) -- returns description plus page content.

    Rows the origin returned a non-200 for are dropped: Jina reports per-result
    fetch failures inline rather than as a request-level error, so a status
    check is the only way to know which links are actually alive.
    """
    key = _key("JINA_API_KEY")
    if not key or not query.strip():
        return [], {}
    import requests
    from urllib.parse import quote
    try:
        resp = requests.get(
            f"https://s.jina.ai/{quote(query, safe='')}",
            headers={"Authorization": f"Bearer {key}",
                     "Accept": "application/json",
                     "X-Return-Format": "json"},
            timeout=_timeout(),
        )
        if resp.status_code in (401, 403):
            log.warning("jina: key rejected (HTTP %s)", resp.status_code)
            _note_auth_failure("jina")
            return [], {}
        if resp.status_code != 200:
            log.warning("jina: HTTP %s", resp.status_code)
            return [], {}
        data = _json(resp)
    except Exception as exc:
        log.warning("jina search failed: %s: %s", type(exc).__name__, exc)
        return [], {}

    rows = []
    for entry in (data.get("data") or [])[:n]:
        if not isinstance(entry, dict):
            continue
        status = entry.get("httpStatus")
        if isinstance(status, int) and status >= 400:
            continue
        url = (entry.get("url") or "").strip()
        if not url:
            continue
        rows.append(make_item(
            entry.get("title"), url, entry.get("description"),
            entry.get("publishedTime"), "jina", query,
            {"via": "jina search"}))
    _note_auth_success("jina")
    return rows, {}


# --- extract providers ------------------------------------------------------

def extract_tavily(url: str) -> tuple[str, str] | None:
    """Re-exported from webextract so every tier has one calling convention."""
    from .webextract import tavily_extract
    return tavily_extract(url)


def extract_firecrawl(url: str) -> tuple[str, str] | None:
    from .webextract import firecrawl_scrape
    return firecrawl_scrape(url)


# Jina returns a small preamble above the page body. Everything up to and
# including this marker is metadata, not content.
_JINA_BODY = "Markdown Content:"


def extract_jina(url: str) -> tuple[str, str] | None:
    """(title, text) via Jina Reader, or None on any failure.

    Reader is markdown-first, which suits the chunker directly and costs no
    extra call to strip tags.
    """
    key = _key("JINA_API_KEY")
    if not key or not url:
        return None
    import requests
    try:
        resp = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"Authorization": f"Bearer {key}",
                     "Accept": "text/plain",
                     "X-Return-Format": "markdown",
                     # Link summaries are useful to a search engine and pure
                     # noise here; they would land in the chunk budget.
                     "X-With-Links-Summary": "false"},
            timeout=_timeout(60.0),
        )
        if resp.status_code in (401, 403):
            log.warning("jina reader: key rejected (HTTP %s)", resp.status_code)
            _note_auth_failure("jina")
            return None
        if resp.status_code != 200:
            return None
        raw = resp.text or ""
        title = ""
        body = raw
        if raw.startswith("Title:"):
            head, _, body = raw.partition(_JINA_BODY)
            for line in head.splitlines():
                if line.startswith("Title:"):
                    title = line.split(":", 1)[1].strip()
                    break
            if not _JINA_BODY in raw:
                body = raw
        _note_auth_success("jina")
        text = _clean_text(body)
        if len(text.split()) < 30:
            return None
        return (title, text)
    except Exception as exc:
        log.debug("jina reader failed for %s: %s", url, exc)
        return None


def extract_tinyfish(url: str) -> tuple[str, str] | None:
    """(title, text) via TinyFish Fetch. Free to ~1000 URLs/day."""
    key = _key("TINYFISH_API_KEY")
    if not key or not url:
        return None
    import requests
    try:
        resp = requests.post(
            "https://api.fetch.tinyfish.ai",
            headers={"X-API-Key": key, "Content-Type": "application/json"},
            json={"urls": [url], "format": "markdown"},
            timeout=_timeout(90.0),
        )
        if resp.status_code in (401, 403):
            log.warning("tinyfish fetch: key rejected (HTTP %s)", resp.status_code)
            _note_auth_failure("tinyfish")
            return None
        if resp.status_code != 200:
            return None
        results = _json(resp).get("results") or []
        if not results or not isinstance(results[0], dict):
            # Per-URL failures come back in errors[] with a 200 response, so an
            # empty results list is a failure, not a legitimately empty page.
            return None
        first = results[0]
        text = _clean_text(first.get("text") or "")
        if len(text.split()) < 30:
            return None
        _note_auth_success("tinyfish")
        return (first.get("title") or "", text)
    except Exception as exc:
        log.debug("tinyfish fetch failed for %s: %s", url, exc)
        return None


def extract_scrapingbee(url: str) -> tuple[str, str] | None:
    """(title, text) via ScrapingBee with JS rendering.

    The most expensive tier (render_js costs several credits a call), so it sits
    late in the default order. What earns that cost is that it renders in a real
    browser and so reads pages that need one.
    """
    key = _key("SCRAPINGBEE_API_KEY")
    if not key or not url:
        return None
    import requests
    try:
        resp = requests.get(
            "https://app.scrapingbee.com/api/v1",
            headers={"Authorization": f"Bearer {key}"},
            params={"url": url,
                    "render_js": "true",
                    "return_page_markdown": "true"},
            timeout=_timeout(90.0),
        )
        if resp.status_code in (401, 403):
            log.warning("scrapingbee: key rejected (HTTP %s)", resp.status_code)
            _note_auth_failure("scrapingbee")
            return None
        if resp.status_code != 200:
            return None
        _note_auth_success("scrapingbee")
        text = _clean_text(resp.text or "")
        if len(text.split()) < 30:
            return None
        return ("", text)
    except Exception as exc:
        log.debug("scrapingbee failed for %s: %s", url, exc)
        return None


# --- the registry -----------------------------------------------------------

@dataclass
class Provider:
    """One external service, in one of the two roles.

    Deliberately not frozen. The registry is built once at import and never
    reassigned, so immutability buys very little -- but it does cost the one
    thing that matters for a network client: the ability to substitute the
    transport in a test. Every provider here is a thin HTTP wrapper, and a
    wrapper you cannot stub is a wrapper you can only exercise by making real
    billed calls.
    """

    name: str
    role: str
    env: str                     # API key variable name
    purpose: str                 # shown in the services panel
    meter: str                   # how it is limited, shown in the panel
    call: Callable
    cap_default: int = 10        # per-run extract calls
    #: search sources also need a human label in the evidence legend
    source_label: str = ""
    #: source name the planner may use (search role only)
    source: str = ""
    #: how a fetch via this provider is recorded on a document. Defaults to
    #: ``name`` but is NOT always the same: search results and fetch methods
    #: share one label table in the report, so a provider with both roles needs
    #: distinct strings or "read via Jina" and "found via Jina" become the
    #: same claim in the evidence legend.
    method: str = ""
    #: legend text for the fetch method, keyed by ``method_name``
    method_label: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def method_name(self) -> str:
        return self.method or self.name

    @property
    def has_key(self) -> bool:
        return bool(_key(self.env))

    @property
    def wired(self) -> bool:
        """Usable right now: has a key, is not switched off, is not quarantined.

        Quarantine is checked here rather than inside each call so every
        consumer -- the fallback chain, the search dispatcher and the dashboard
        -- agrees on whether the provider is available. Three separate
        opinions about "is this provider usable" is how a run ends up paying
        for a dead key.
        """
        return (self.has_key and not self.is_disabled
                and not quarantined(self.name))

    @property
    def is_disabled(self) -> bool:
        raw = (os.environ.get(f"{self.name.upper()}_ENABLED") or "").strip().lower()
        return raw in ("0", "false", "no", "off")

    def run_search(self, query: str, n: int) -> tuple[list[dict], dict]:
        if not self.wired:
            return [], {}
        try:
            rows, meta = self.call(query, n)
        except Exception as exc:                     # belt and braces
            log.warning("%s: search raised %s: %s", self.name, type(exc).__name__, exc)
            return [], {}
        return rows or [], meta or {}

    def run_extract(self, url: str) -> tuple[str, str] | None:
        if not self.wired:
            return None
        try:
            return self.call(url)
        except Exception as exc:
            log.warning("%s: extract raised %s: %s", self.name, type(exc).__name__, exc)
            return None


_REGISTRY: tuple[Provider, ...] = (
    Provider(
        name="exa", role=SEARCH, env="EXA_API_KEY",
        purpose="neural/semantic web search",
        meter="billed per search call",
        call=search_exa, cap_default=60,
        source="exa", source_label="Exa neural search result",
        meta={"why": "ranks by meaning, so it finds pages keyword search misses"},
    ),
    Provider(
        name="tinyfish", role=SEARCH, env="TINYFISH_API_KEY",
        purpose="ranked web search",
        meter="free to ~12k calls/day",
        call=search_tinyfish, cap_default=60,
        source="tinyfish", source_label="TinyFish ranked web result",
        meta={"why": "independent index; a second opinion on the same query"},
    ),
    Provider(
        name="jina", role=SEARCH, env="JINA_API_KEY",
        purpose="search with page content",
        meter="free tier, then billed",
        call=search_jina, cap_default=60,
        source="jina", source_label="Jina search result",
        meta={"why": "returns descriptions alongside links"},
    ),
    Provider(
        name="jina", role=EXTRACT, env="JINA_API_KEY",
        purpose="fetch fallback: page to markdown",
        meter="free tier, then billed",
        call=extract_jina, cap_default=20, method="jina_reader",
        method_label="page extracted via Jina Reader",
    ),
    Provider(
        name="tinyfish", role=EXTRACT, env="TINYFISH_API_KEY",
        purpose="fetch fallback: page to markdown",
        meter="free to 1000 URLs/day",
        call=extract_tinyfish, cap_default=20, method="tinyfish_fetch",
        method_label="page extracted via TinyFish Fetch",
    ),
    Provider(
        name="tavily", role=EXTRACT, env="TAVILY_API_KEY",
        purpose="paid fetch fallback",
        meter="billed per page",
        call=extract_tavily, cap_default=10,
        method_label="page extracted via Tavily",
    ),
    Provider(
        name="scrapingbee", role=EXTRACT, env="SCRAPINGBEE_API_KEY",
        purpose="fetch fallback: JS rendering, stealth proxy",
        meter="~5 credits per rendered page",
        call=extract_scrapingbee, cap_default=4,
        method_label="page extracted via ScrapingBee (JS rendered)",
    ),
    Provider(
        name="firecrawl", role=EXTRACT, env="FIRECRAWL_API_KEY",
        purpose="paid fetch fallback",
        meter="billed per page",
        call=extract_firecrawl, cap_default=4,
        method_label="page extracted via Firecrawl",
    ),
)


def all_providers() -> tuple[Provider, ...]:
    return _REGISTRY


def for_role(role: str) -> list[Provider]:
    return [p for p in _REGISTRY if p.role == role]


def search_providers() -> dict[str, Provider]:
    """Search providers keyed by the source name the planner emits."""
    return {p.source: p for p in _REGISTRY
            if p.role == SEARCH and p.source}


def general_sources() -> tuple[str, ...]:
    """Every open-web search source, SerpApi first.

    SerpApi leads because its rows carry the most (dates, related questions,
    answer boxes), and when two engines return the same URL the collector keeps
    whichever arrives first.
    """
    return GENERAL_SOURCES + tuple(sorted(search_providers()))


def wired_general_sources() -> list[str]:
    """The general sources that could actually answer a query right now."""
    out = []
    for name in general_sources():
        if name == "web":
            if (os.environ.get("SERPAPI_KEY") or "").strip():
                out.append(name)
        elif search_providers().get(name, None) and search_providers()[name].wired:
            out.append(name)
    return out


def wired_extract_providers() -> list[Provider]:
    """Every extract provider that could run, unordered.

    There is deliberately no fallback *order*. A tiered chain spends the run's
    time walking providers that have already declined, and its hit rate is
    bounded by the best single provider rather than by the union of all of
    them. The caller races them instead; see Fetcher._race_extractors.

    Kept cheap -- no I/O, just key presence -- because it runs per failed
    fetch.
    """
    return [p for p in _REGISTRY if p.role == EXTRACT and p.wired]


def wired_search_providers() -> dict[str, Provider]:
    return {name: p for name, p in search_providers().items() if p.wired}


def source_names() -> tuple[str, ...]:
    """Search source names the planner may choose from, wired or not.

    Unwired ones are still offered: the planner prompt is static text, and a
    provider whose key is missing simply returns nothing at call time rather
    than wasting the sub-question.
    """
    return tuple(sorted(search_providers()))