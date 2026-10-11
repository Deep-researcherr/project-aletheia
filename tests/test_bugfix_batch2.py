"""Regression tests for audit batch 7-12 (silent-corruption / security).

7.  Retrieval limitation reasons reach evidence (not "partial (partial)").
8.  Dedupe keeps distinct claims that share a long common prefix.
9.  Search cache: corrupt files fall back to live, writes are atomic.
10. Fetch budget and paid caps are race-free under fetch_many threads.
11. Scraped URLs are scheme-checked before href (frontend static test).
12. Export registry is bounded and evicts oldest entries.
"""
import json
import threading

import pytest

from backend.research import config


# --- 9. search cache -----------------------------------------------------------

def test_corrupt_cache_falls_back_to_live_call(monkeypatch, tmp_path):
    from backend.research import searcher

    monkeypatch.setenv("SEARCH_CACHE_DIR", str(tmp_path / "cache"))
    path = None
    calls = {"n": 0}

    def fake_run(params):
        calls["n"] += 1
        return {"organic_results": [{"title": "T", "link": "https://a.dev"}]}

    monkeypatch.setattr(searcher, "_run", fake_run)

    # Prime the cache, then corrupt it.
    searcher._cached_run("web", "q", 6, {"engine": "google", "q": "q"})
    path = searcher._cache_path("web", "q", 6)
    path.write_text("{not valid json")

    result = searcher._cached_run("web", "q", 6, {"engine": "google", "q": "q"})

    assert result["organic_results"][0]["title"] == "T"
    assert calls["n"] == 2, "a corrupt cache must trigger exactly one live call"


def test_cache_write_is_atomic(monkeypatch, tmp_path):
    """Concurrent writers must never produce a torn file: the write goes to
    a temp path and is renamed, so readers only ever see whole JSON."""
    from backend.research import searcher

    monkeypatch.setenv("SEARCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(searcher, "_run",
                        lambda params: {"organic_results": [], "n": 1})

    errors = []

    def worker(i):
        try:
            searcher._cached_run("web", f"q{i}", 6, {"engine": "google"})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    cache_dir = tmp_path / "cache"
    for file in cache_dir.iterdir():
        assert not file.name.endswith(".tmp"), "temp files must be renamed away"
        json.loads(file.read_text())  # every file is whole JSON


# --- 10. fetch budget races ----------------------------------------------------

def test_paid_cap_is_race_free(monkeypatch):
    """Check-then-increment without a lock let two workers both pass a cap
    of 1 and make two billed calls."""
    from backend.research.fetch import Fetcher

    class _StubSession:
        headers = {}

    monkeypatch.setenv("TAVILY_MAX_PER_RUN", "1")
    monkeypatch.setenv("FIRECRAWL_MAX_PER_RUN", "0")
    fetcher = Fetcher(session=_StubSession())

    results = []
    lock = threading.Lock()

    def claim():
        got = fetcher._claim_paid("tavily")
        with lock:
            results.append(got)

    threads = [threading.Thread(target=claim) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(1 for r in results if r) == 1, \
        "exactly one worker may pass a per-run cap of one"


def test_fetch_budget_is_race_free(monkeypatch):
    from backend.research.fetch import Fetcher

    class _StubSession:
        headers = {}

    monkeypatch.setenv("RESEARCH_FETCH_BUDGET", "5")
    fetcher = Fetcher(session=_StubSession())

    granted = []
    lock = threading.Lock()

    def claim():
        got = fetcher._claim_fetch()
        with lock:
            granted.append(got)

    threads = [threading.Thread(target=claim) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(1 for g in granted if g) == 5, \
        "a budget of five must grant exactly five fetches"


# --- 12. export registry -------------------------------------------------------

def test_export_registry_is_bounded(monkeypatch, tmp_path):
    from backend.research import export

    monkeypatch.setenv("EXPORT_DIR", str(tmp_path / "exports"))
    monkeypatch.setattr(config, "export_enabled", lambda: True)

    for i in range(export._KEEP_EXPORTS + 6):
        export.save(f"question number {i}", f"# report {i}")

    assert len(export._last) == export._KEEP_EXPORTS, \
        "the export registry must evict oldest entries"
    # The most recent question is still served; the oldest is gone.
    assert export.last_export(f"question number {export._KEEP_EXPORTS + 5}")
    assert export.last_export("question number 0") == {}


# --- 7. retrieval limitations reach evidence -----------------------------------

def test_chunk_carries_retrieval_limitation():
    from backend.research.chunker import chunk_document

    chunks = chunk_document(
        "## S\n\n" + ("body text. " * 40),
        source_id="s1", source_url="https://a.dev", source_title="A",
        retrieval_status="partial",
        retrieval_limitation="paywall detected; text may be truncated")

    assert chunks and chunks[0].retrieval_limitation.startswith("paywall")


def test_dedupe_keeps_distinct_claims_with_a_shared_prefix():
    """Two findings sharing a long lead-in but different trailing figures
    are different claims; a truncated key used to merge them."""
    from backend.research.evidence import Evidence, dedupe

    prefix = ("the benchmark measured throughput on the reference cluster "
              "for the standard workload configuration over a full day of ")
    a = Evidence(claim=prefix + "steady state and reported 1,800 tokens/s",
                 source_url="https://a.dev", quality="strong")
    b = Evidence(claim=prefix + "burst load and reported 940 tokens/s",
                 source_url="https://b.dev", quality="strong")

    kept = dedupe([a, b])

    assert len(kept) == 2, "distinct claims must not be merged by prefix"


def test_harvest_uses_the_real_limitation():
    from backend.research.chunker import Chunk
    from backend.research.extract import _harvest

    chunk = Chunk(chunk_id="S1#0", source_id="S1",
                  source_url="https://a.dev", source_title="A",
                  content="Measured 1.8 seconds to first frame in release mode.",
                  retrieval_status="partial",
                  retrieval_limitation="download hit the byte cap")
    reply = {"evidence": [{"chunk": "S1#0",
                           "claim": "first frame in 1.8s in release mode",
                           "quote": "1.8 seconds to first frame",
                           "quality": "strong", "confidence": 0.9}]}

    accepted, _, _status = _harvest(reply, [], {"S1#0": chunk})

    assert accepted[0].retrieval_limitation == "download hit the byte cap", \
        "the human-readable reason must survive into evidence"
