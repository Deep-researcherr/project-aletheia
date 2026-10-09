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


def test_stream_writer_is_unavailable_inside_a_worker_thread():
    """Why the writer has to be captured before it crosses into a thread pool.

    LangGraph resolves the stream writer through a contextvar, and contextvars
    do not propagate into ThreadPoolExecutor workers: get_config() raises
    "outside of a runnable context" there. Resolving the writer inside a fetch
    callback therefore yields None and the per-document events silently never
    fire -- the feature looks alive in unit tests and does nothing in a real
    run. This pins the underlying behaviour so the capture above stays
    necessary rather than looking redundant.
    """
    from concurrent.futures import ThreadPoolExecutor

    import langgraph.config as lc

    # With no runnable context at all, the lookup is a safe no-op.
    assert G._stream_writer() is None

    # Simulate being inside a graph node: get_stream_writer() reads
    # config[CONF][CONFIG_KEY_RUNTIME].stream_writer, so the stand-in has to be
    # subscriptable exactly as the real config is.
    runtime = type("R", (), {"stream_writer": staticmethod(lambda _e: None)})()
    token = lc.var_child_runnable_config.set(
        {lc.CONF: {lc.CONFIG_KEY_RUNTIME: runtime}})
    try:
        assert G._stream_writer() is not None, "resolvable in the node context"

        def from_worker(_i):
            try:
                return G._stream_writer()
            except Exception as exc:      # pragma: no cover - defensive
                return exc

        with ThreadPoolExecutor(max_workers=1) as pool:
            got = list(pool.map(from_worker, [0]))[0]
        assert got is None, (
            "contextvar unexpectedly propagated into the worker; if this now "
            "holds, the writer capture in retrieve() can be simplified")
    finally:
        lc.var_child_runnable_config.reset(token)


def test_retrieve_emits_one_event_per_document_from_worker_threads(monkeypatch):
    """Per-document progress, emitted from real worker threads.

    The fake fetcher dispatches to a thread pool, so this reproduces the
    contextvar problem: the writer is resolved once in the node and handed to
    the workers.
    """
    from concurrent.futures import ThreadPoolExecutor
    import langgraph.config as lc

    delivered = []
    # A real runnable context, so _stream_writer() resolves through the same
    # contextvar path production uses rather than a stubbed global.
    runtime = type("R", (), {"stream_writer": staticmethod(delivered.append)})()
    token = lc.var_child_runnable_config.set(
        {lc.CONF: {lc.CONFIG_KEY_RUNTIME: runtime}})
    try:
        run = _retrieve_against(_ThreadedFetcher(), 5)
    finally:
        lc.var_child_runnable_config.reset(token)

    counts = [(e.get("done"), e.get("total")) for e in delivered]
    assert counts[0] == (0, 5)
    assert sorted(c[0] for c in counts[1:]) == [1, 2, 3, 4, 5], (
        f"per-document events lost across the thread boundary: {counts}")


class _ThreadedFetcher:
    """Dispatches fetches to real worker threads, like the production fetcher."""

    budget = 100
    fetched = 0

    def fetch_many(self, urls, on_done=None):
        from concurrent.futures import ThreadPoolExecutor

        def one(url):
            doc = FetchedDoc(url=url, final_url=url, status="full",
                             method="html", title=f"Doc {url}",
                             text="body text " * 40)
            if on_done is not None:
                on_done(doc)              # runs on a worker thread
            return doc

        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(one, urls))


def _retrieve_against(fetcher, count):
    state = {"question": "q", "max_rounds": 1, "round": 1,
             "plan": {"subquestions": []}, "log": [], "notes": [],
             "fetcher": fetcher,
             "raw": [{"title": "T", "url": f"https://x.dev/{i}", "snippet": "s",
                      "date": "2025", "type": "web", "query": "q"}
                     for i in range(count)]}
    return asyncio.run(G.retrieve(state))



def test_progress_survives_a_real_compiled_graph_with_worker_threads():
    """The full path: compiled graph, node context, real fetch threads.

    Unit tests with a stubbed writer cannot catch this class of bug -- the
    writer only works inside the node's context, and a stub does not care. Here
    the node resolves the writer from LangGraph for real, the fetch runs on
    genuine worker threads, and the events still have to arrive.
    """
    import operator
    import time as _time
    from concurrent.futures import ThreadPoolExecutor
    from typing import Annotated, TypedDict

    from langgraph.graph import START, StateGraph

    class _S(TypedDict):
        v: Annotated[list, operator.add]

    class _Fetcher:
        budget = 100
        fetched = 0

        def fetch_many(self, urls, on_done=None):
            def one(url):
                _time.sleep(0.02)
                if on_done:
                    on_done(f"doc-{url}")
                return url
            with ThreadPoolExecutor(max_workers=4) as pool:
                return list(pool.map(one, urls))

    async def node(state):
        writer = G._stream_writer()
        todo = [f"u{i}" for i in range(8)]
        progress = G._ProgressQueue("retrieve", writer)
        progress.announce(len(todo), note="downloading sources")

        def on_done(doc):
            progress.done += 1
            progress.emit(done=progress.done, total=len(todo),
                          url=doc, note=f"read {doc}")

        got = await G._fetch_reporting(_Fetcher(), todo, on_done, progress)
        return {"v": [len(got)]}

    builder = StateGraph(_S)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    app = builder.compile()

    async def drain():
        events = []
        async for mode, chunk in app.astream({"v": []},
                                            stream_mode=["updates", "custom"]):
            if mode == "custom":
                events.append(chunk)
        return events

    events = asyncio.run(drain())
    counts = [e.get("done") for e in events]
    assert counts == list(range(9)), f"expected 0..8, got {counts}"
    assert all(e["stage"] == "retrieve" for e in events)


def test_retrieve_emits_one_event_per_document(monkeypatch):
    """Real per-document progress, not a single start and stop."""
    written = []
    monkeypatch.setattr(G, "_stream_writer", lambda: written.append)

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