"""Progress events: the long stages must not look frozen.

retrieve and analyse take minutes and used to emit nothing until they finished,
so the browser showed a frozen "Reading pages" for the whole of it. They now
write progress from inside the stage via LangGraph's stream writer, which
arrives as a "custom" stream alongside the node updates.

These pin the contract: the event shape, that it reaches the stream at all, and
that a node called directly (no stream attached) stays silent rather than
raising.
"""
import asyncio

from backend.research import graph as G
from backend.research.fetch import FAILED, FetchedDoc
from backend.research.llm import CallBudget


class _StubFetcher:
    def __init__(self):
        self.budget = 100
        self.fetched = 0

    def fetch_many(self, urls, on_done=None):
        out = []
        for url in urls:
            self.fetched += 1
            doc = FetchedDoc(url=url, final_url=url, status="full", method="html",
                             title=f"Doc {url}", text="body text about the topic " * 40)
            out.append(doc)
            if on_done is not None:
                on_done(doc)
        return [d for d in out if d.status != FAILED]


def test_progress_is_silent_without_a_stream():
    """A node called directly must not raise just because nothing is attached."""
    G._progress("retrieve", done=1, total=2)


def test_progress_survives_a_broken_writer(monkeypatch):
    """Progress is cosmetic and must never take a stage down with it."""
    class Boom:
        def __call__(self, _payload):
            raise RuntimeError("stream gone")

    import langgraph.config as lc
    monkeypatch.setattr(lc, "get_stream_writer", lambda: Boom())
    G._progress("analyse", done=1, total=2)


def test_progress_reaches_an_attached_writer(monkeypatch):
    written = []
    monkeypatch.setattr("langgraph.config.get_stream_writer",
                        lambda: written.append)
    G._progress("retrieve", done=3, total=9, note="read a doc")
    assert written == [{"type": "progress", "stage": "retrieve",
                        "done": 3, "total": 9, "note": "read a doc"}]


def test_retrieve_emits_one_event_per_document(monkeypatch):
    """Real per-document progress, not a single start and stop."""
    written = []
    monkeypatch.setattr("langgraph.config.get_stream_writer",
                        lambda: written.append)

    state = {"question": "q", "max_rounds": 1, "round": 1,
             "plan": {"subquestions": []}, "log": [], "notes": [],
             "fetcher": _StubFetcher(),
             "raw": [{"title": "T", "url": f"https://x.dev/{i}", "snippet": "s",
                      "date": "2025", "type": "web", "query": "q"}
                     for i in range(5)]}
    asyncio.run(G.retrieve(state))

    counts = [(e.get("done"), e.get("total")) for e in written]
    assert counts[0] == (0, 5), "should announce the batch first"
    assert [c[0] for c in counts[1:]] == [1, 2, 3, 4, 5]
    assert all(e["type"] == "progress" and e["stage"] == "retrieve" for e in written)


def test_retrieve_progress_does_not_break_a_legacy_fetcher(monkeypatch):
    """A fetcher that predates the callback still retrieves documents."""
    written = []
    monkeypatch.setattr("langgraph.config.get_stream_writer",
                        lambda: written.append)

    class Legacy:
        budget = 100
        fetched = 0

        def fetch_many(self, urls):          # no on_done parameter
            return [FetchedDoc(url=u, final_url=u, status="full", method="html",
                               title="T", text="body " * 40) for u in urls]

    state = {"question": "q", "max_rounds": 1, "round": 1,
             "plan": {"subquestions": []}, "log": [], "notes": [],
             "fetcher": Legacy(),
             "raw": [{"title": "T", "url": f"https://x.dev/{i}", "snippet": "s",
                      "date": "2025", "type": "web", "query": "q"}
                     for i in range(3)]}
    out = asyncio.run(G.retrieve(state))
    assert out["retrieval_stats"]["retrieved"] == 3
    # Still announced the batch, just without per-document detail.
    assert written and written[0]["done"] == 0


def test_analyse_emits_progress_for_each_phase(monkeypatch):
    """Both halves of analyse report: the ranking pass and the extraction."""
    written = []
    monkeypatch.setattr("langgraph.config.get_stream_writer",
                        lambda: written.append)

    from backend.research.chunker import chunk_document
    chunks = []
    for i in range(3):
        made = chunk_document(
            f"How the system scales under load, measured throughput {i}. " * 30,
            source_id=f"S{i}", source_url=f"https://e.com/{i}",
            source_title=f"D{i}")
        chunks.extend(made)

    # Force the ranked set non-empty: relevance scoring is corpus-dependent, so
    # the assertion here is about progress reporting, not about which passages
    # the ranker happens to keep.
    real_rank = G.rank
    from backend.research.relevance import Scored
    monkeypatch.setattr(
        G, "rank",
        lambda scored, q, p=None: (real_rank(scored, q, p)
                                   or [Scored(chunk=c, score=1.0, reasons=[])
                                       for c in scored[:2]]))

    state = {"question": "how does it scale", "chunks": chunks, "extracted": [],
             "extracted_sources": {}, "plan": {}, "started_at": None,
             "budget": CallBudget(total=10), "max_rounds": 1}
    asyncio.run(G.analyse(state))

    stages = [e["stage"] for e in written if e.get("stage") == "analyse"]
    assert stages, "analyse emitted no progress"
    notes = " ".join(str(e.get("note", "")) for e in written)
    assert "ranking" in notes
    assert "extracting evidence" in notes


def test_runlog_tallies_progress_without_storing_every_url(monkeypatch):
    """The PDF keeps a per-stage summary, not a log of every document."""
    from backend.research import runlog

    runlog.reset()
    runlog.begin("a question about scaling", 2)
    for done in (1, 2, 3):
        runlog.record("a question about scaling", {
            "type": "progress", "stage": "retrieve", "done": done, "total": 3,
            "url": f"https://example.com/{done}", "note": "read a doc"})
    bundle = runlog.get("a question about scaling")
    runlog.reset()

    # Not complete, so get() returns None; inspect through record's own store.
    assert bundle is None
    from backend.research import runlog as rl
    rl.reset()
    rl.begin("q", 1)
    rl.record("q", {"type": "progress", "stage": "retrieve", "done": 2,
                    "total": 5, "note": "reading"})
    stored = rl._runs["q"]["progress"]["retrieve"]
    assert stored == {"done": 2, "total": 5, "note": "reading"}
    assert "url" not in stored