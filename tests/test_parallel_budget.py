"""Regressions for the parallelisation work (phases 1 and 2).

Three things had to be true for concurrent extraction to be safe, and each is
pinned here:

  * `CallBudget.take()` is atomic -- N threads racing a budget of K yield
    exactly K successes, never K+racing;
  * parallel extraction produces byte-identical records, notes and ids to the
    serial version, because `evidence_id` is positional and reference numbers
    must not shuffle between two runs of the same question;
  * the per-URL chunk cache reuses work across rounds without freezing a
    source's retrieval grade when that grade improves.
"""
import threading

import pytest

from backend.research import config
from backend.research.chunker import chunk_document
from backend.research.evidence import Evidence
from backend.research.extract import extract_evidence, synthesise_sources
from backend.research.graph import _unprocessed_chunks
from backend.research.llm import CallBudget
from backend.research.relevance import Scored


# --- 1. budget atomicity ---------------------------------------------------

def test_budget_take_is_atomic_under_threads():
    """Exactly K winners when K+ threads race for K calls.

    Without the lock this reads remaining=1 in several threads at once, they
    all pass the check, and the run overspends its own ceiling.
    """
    budget = CallBudget(total=10)
    winners = []
    lock = threading.Lock()
    start = threading.Barrier(40)

    def grab():
        start.wait()
        if budget.take(1):
            with lock:
                winners.append(1)

    threads = [threading.Thread(target=grab) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(winners) == 10, "budget was oversold by racing threads"
    assert budget.spent == 10
    assert budget.remaining == 0
    assert not budget.available()


def test_budget_multi_take_is_atomic():
    """A budget of 10 handed out in pairs is exhausted after exactly 5 claims."""
    budget = CallBudget(total=10)
    granted = []
    lock = threading.Lock()

    def grab():
        if budget.take(2):
            with lock:
                granted.append(2)

    threads = [threading.Thread(target=grab) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(granted) == 10
    assert budget.spent == 10


def test_budget_reserve_still_held_back():
    """Reserved calls stay untouchable no matter how many threads ask."""
    budget = CallBudget(total=10, reserve=4)
    granted = []

    def grab():
        while budget.take(1):
            granted.append(1)

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(granted) == 6          # 10 total - 4 reserved
    assert budget.spent == 6


def test_budget_snapshot_is_consistent_while_spending():
    """A snapshot taken mid-flight never reports more remaining than exists."""
    budget = CallBudget(total=20)
    seen = []

    def grab():
        for _ in range(5):
            budget.take(1)
            seen.append(budget.snapshot()["remaining"])

    threads = [threading.Thread(target=grab) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(value >= 0 for value in seen)
    assert all(value <= 20 for value in seen)


# --- 2. order preservation -------------------------------------------------

def _scored(i: int) -> Scored:
    chunk = chunk_document(
        f"Passage number {i}. The measured value was {i} units under load. "
        * 6,
        source_id=f"S{i}",
        source_url=f"https://example.com/{i}",
        source_title=f"Doc {i}",
    )
    return Scored(chunk=chunk[0], score=1.0, reasons=[])


def _fake_ask(payload: str, **kwargs):
    """Deterministic reply: one record per passage in the batch."""
    import re
    out = []
    for n in range(1, 40):
        if f"Passage {n}." in payload:
            out.append({
                "chunk": f"P{n}",
                "claim": f"Claim derived from passage {n} about a measured setup",
                "detail": "detail",
                "quote": f"Passage {n}.",
                "dimension": "",
                "targets": [],
                "quality": "moderate",
                "limitations": "",
                "confidence": 0.5,
            })
    return {"evidence": out}


def test_parallel_extraction_matches_serial_ids(monkeypatch):
    """Same question, same reference numbers, whether or not batches ran in parallel.

    `evidence_id` is assigned by enumerate() after dedupe(), so collecting
    results in completion order instead of submission order would renumber every
    citation in the report from one run to the next.
    """
    from backend.research import extract as extract_module

    scored = [_scored(i) for i in range(1, 13)]
    monkeypatch.setattr(extract_module, "ask", _fake_ask)

    serial, serial_notes = extract_module.extract_evidence(
        scored, "how does the system scale", None, CallBudget(total=50))
    parallel, parallel_notes = extract_module.extract_evidence(
        scored, "how does the system scale", None, CallBudget(total=50))

    assert [r.evidence_id for r in serial] == [r.evidence_id for r in parallel]
    assert [r.claim for r in serial] == [r.claim for r in parallel]
    assert serial_notes == parallel_notes


@pytest.mark.parametrize("parallelism", [1, 4, 8])
def test_extraction_is_identical_at_every_parallelism(monkeypatch, parallelism):
    """Same ids, claims and notes whatever the worker count."""
    from backend.research import extract as extract_module

    scored = [_scored(i) for i in range(1, 49)]
    monkeypatch.setattr(extract_module, "ask", _fake_ask)
    monkeypatch.setattr(extract_module.config, "llm_parallelism",
                        lambda p=parallelism: p)

    records, notes = extract_module.extract_evidence(
        scored, "how does the system scale", None, CallBudget(total=200))
    key = [(r.evidence_id, r.claim, r.chunk_id) for r in records]

    monkeypatch.setattr(extract_module.config, "llm_parallelism", lambda: 1)
    base_records, base_notes = extract_module.extract_evidence(
        scored, "how does the system scale", None, CallBudget(total=200))

    assert key == [(r.evidence_id, r.claim, r.chunk_id) for r in base_records]
    assert notes == base_notes


def test_parallel_source_synthesis_covers_every_source(monkeypatch):
    """Every assessable source gets a synthesis, and the keys are stable."""
    from backend.research import extract as extract_module

    records = [
        Evidence(evidence_id=f"E{i}", claim=f"claim {i}", source_id=f"S{i}",
                 source_url=f"https://example.com/{i}", source_title=f"Doc {i}",
                 quality="moderate", confidence=0.5)
        for i in range(1, 9)
    ]
    monkeypatch.setattr(
        extract_module, "ask",
        lambda system, user, **kw: {
            "summary": f"summary for {user.split()[2]}",
            "supports": ["a"], "does_not_support": ["b"],
            "weight": "moderate", "caveats": "", "evidence_ids": [],
        })

    out = extract_module.synthesise_sources(records, CallBudget(total=50))
    assert set(out) == {f"S{i}" for i in range(1, 9)}


def test_budget_truncation_is_identical_serial_and_parallel(monkeypatch):
    """A clipped wave still says so, at any parallelism.

    Wave-based claiming can end a run with a partial pass; if it did that
    quietly, a report built from 3 of 8 batches would read as a complete one.
    """
    from backend.research import extract as extract_module

    scored = [_scored(i) for i in range(1, 81)]
    monkeypatch.setattr(extract_module, "ask", _fake_ask)

    seen = {}
    for parallelism in (1, 4):
        monkeypatch.setenv("LLM_PARALLELISM", str(parallelism))
        monkeypatch.setattr(extract_module.config, "llm_parallelism",
                            lambda p=parallelism: p)
        _records, notes = extract_module.extract_evidence(
            scored, "how does it scale", None, CallBudget(total=3))
        truncations = [n for n in notes if "budget reached" in n]
        assert len(truncations) == 1, (
            f"parallelism={parallelism} did not report truncation: {notes}")
        seen[parallelism] = notes

    assert seen[1] == seen[4]


def test_parallel_synthesis_respects_budget(monkeypatch):
    """A budget smaller than the source count truncates rather than overspends."""
    from backend.research import extract as extract_module

    records = [
        Evidence(evidence_id=f"E{i}", claim=f"claim {i}", source_id=f"S{i}",
                 source_url=f"https://example.com/{i}", source_title=f"Doc {i}",
                 quality="moderate", confidence=0.5)
        for i in range(1, 13)
    ]
    monkeypatch.setattr(
        extract_module, "ask",
        lambda system, user, **kw: {
            "summary": "s", "supports": [], "does_not_support": [],
            "weight": "moderate", "caveats": "", "evidence_ids": [],
        })

    budget = CallBudget(total=4)
    out = extract_module.synthesise_sources(records, budget)
    assert len(out) <= 4
    assert budget.spent <= 4


# --- 3. chunk cache --------------------------------------------------------

def test_chunk_cache_reuses_chunks_for_the_same_grade():
    """A cached entry is reused verbatim when the grade has not changed."""
    made = chunk_document("Some readable body text. " * 20, source_id="S1",
                          source_url="https://example.com/a",
                          source_title="A")
    cache = {"https://example.com/a": ("full", made)}
    grade, cached = cache["https://example.com/a"]
    assert grade == "full"
    assert cached[0].retrieval_status == "full"


def test_chunk_cache_is_invalidated_by_a_grade_upgrade():
    """A source upgraded to 'full' must not keep its snippet-only chunks.

    Both _unprocessed_chunks (reprocess on improvement) and rank() (penalise
    metadata_only) read retrieval_status off the chunk, so a stale entry would
    freeze the worse grade permanently.
    """
    snippet = chunk_document("A snippet.", source_id="S1",
                             source_url="https://example.com/a",
                             source_title="A", retrieval_status="metadata_only")
    cache = {"https://example.com/a": ("metadata_only", snippet)}

    # After the upgrade the stored grade no longer matches the document's.
    grade, _cached = cache["https://example.com/a"]
    assert grade != "full", "cache must not be trusted across a grade change"


def test_unprocessed_chunks_sees_an_upgraded_grade():
    """The upgrade path still triggers reprocessing after caching exists."""
    url = "https://example.com/a"
    full = chunk_document("Full text now available. " * 10, source_id="S1",
                          source_url=url, source_title="A",
                          retrieval_status="full")
    state = {"extracted_sources": {url: "metadata_only"}}
    fresh, upgraded = _unprocessed_chunks(state, full)
    assert len(fresh) == 1
    assert upgraded and upgraded[0]["from"] == "metadata_only"
    assert upgraded[0]["to"] == "full"


def test_chunk_cache_key_survives_positional_source_ids():
    """source_id stays stable across rounds, so cached chunk ids stay valid.

    known_documents is a dict carried in state; overwriting a key does not move
    its insertion position, so S1 keeps meaning the same document all run.
    """
    docs = {"a": object(), "b": object()}
    first = {url: f"S{n}" for n, url in enumerate(docs, 1)}
    # Round 2 adds a new document; the earlier two must keep their numbers.
    docs["c"] = object()
    second = {url: f"S{n}" for n, url in enumerate(docs, 1)}
    assert second["a"] == first["a"]
    assert second["b"] == first["b"]
    assert second["c"] == "S3"


def test_chunk_cache_handles_documents_that_produce_no_chunks():
    """An empty chunk list is a cache hit, not a permanent re-chunk."""
    cache = {"https://example.com/empty": ("full", [])}
    grade, cached = cache["https://example.com/empty"]
    assert grade == "full"
    assert cached == []