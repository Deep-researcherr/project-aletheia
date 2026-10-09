"""LangGraph orchestration.

    START -> planner -> [search_worker x N in parallel] -> collect
          -> retrieve -> analyse -> gap_check --(gaps)--> search_worker ...
                                              `--(done)--> contradiction_check
                                                          -> report -> END

``retrieve`` and ``analyse`` are the new stages. Retrieval, extraction and
reference discovery all run after the first search round and are reused by every
later round, so the gap-check loop refines the same evidence base instead of
re-downloading the corpus.

Only search, gap_check, contradiction_check and report use the LLM. Everything
between them -- downloading, parsing, chunking, relevance scoring, reference
numbering, the reference table -- is deterministic.
"""
import asyncio
import logging
import operator
import re
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import config
from .chunker import chunk_document
from .collector import Collector
from .evidence import statistics
from .export import save as save_export
from .extract import (extract_evidence, find_conflicts,
                      findings_from_contradictions, synthesise_sources)
from .fetch import FAILED, Fetcher, FULL, METADATA, PARTIAL, snippet_only_doc
from .llm import CallBudget, ask
from .markdown_html import render_markdown_html
from .planner import SOURCES, make_plan
from .references import candidates, describe, forward_candidates
from .relevance import dimensions as question_dimensions
from .relevance import diversify, rank, source_context, uncovered_questions
from .report import evidence_holes, generate_report, status_headline, write_report
from .searcher import cited_by_search, search_full
from .wiki import fetch_wikipedia

log = logging.getLogger(__name__)

# Strength order for a retrieval grade. A source only ever improves as more
# rounds succeed, so the index keeps the best result seen for each URL.
_STATUS_RANK = {FAILED: 0, METADATA: 1, PARTIAL: 2, FULL: 3}

GAP_SYS = """You are a research lead. Given the question and evidence so far, decide if more searching
is needed. Only request searches that fill a SPECIFIC gap: no primary source, one-sided evidence,
only stale/undated results, missing recent news, or an unanswered sub-question. Return JSON only:
{"sufficient":true|false,"missing":"what is missing","follow_ups":[{"source":"web|news|scholar|github","query":"..."}]}
At most 3 follow_ups. Prefer queries that would reach primary sources, benchmarks or official
documentation over general commentary."""


class State(TypedDict, total=False):
    question: str
    max_rounds: int
    round: int
    plan: dict
    pending: list                         # searches to run next
    seen: list                            # "source|query" already run
    raw: Annotated[list, operator.add]    # parallel workers append here
    log: Annotated[list, operator.add]
    paa: Annotated[list, operator.add]   # related questions/searches bank
    ai_leads: Annotated[list, operator.add]  # AI-overview reference links
    evidence: list
    gap: dict
    contradictions: list
    report: str

    # --- deep-synthesis state ---
    documents: list                       # FetchedDoc (kept out of streamed output)
    chunks: list                          # Chunk objects, provenance intact
    extracted: list                       # Evidence records
    source_synth: dict
    retrieval_stats: dict
    notes: Annotated[list, operator.add]
    budget: object                        # CallBudget, shared across stages
    retrieval_index: dict                 # url -> what was retrieved, whole run
    documents_by_url: dict                # url -> FetchedDoc, best seen
    chunks_by_url: dict                   # url -> [Chunk], so re-chunking is done once
    status: dict                          # research completion state
    fetcher: object                      # Fetcher, so the fetch budget is per run
    evidence_stats: dict
    extracted_sources: dict                    # url -> grade already extracted
    started_at: float                          # time.monotonic() at run start
    export: dict


def _col(state) -> Collector:
    col = Collector()
    col.add([dict(r) for r in state.get("raw", [])])
    return col


def _progress(stage: str, **fields) -> None:
    """Write a progress event from inside a stage, if a stream is attached.

    retrieve and analyse take minutes and used to report nothing until they
    finished, so the browser showed a frozen step for the whole of it. The
    writer is absent when a node is called directly (tests, or a graph run with
    no custom stream), so this stays silent rather than raising.

    Only usable from inside the node's own context -- see :func:`_progress_queue`.
    """
    writer = _stream_writer()
    if writer is None:
        return
    _write_progress(writer, stage, **fields)


def _stream_writer():
    """Resolve the stream writer, or None when no stream is attached.

    Must be called from inside the node. LangGraph resolves the writer through
    a contextvar, and the writer it hands back is itself a closure that reads
    that same contextvar on every call. So neither looking it up nor holding on
    to it helps once you cross a thread boundary: from a fetch worker both raise
    "Called get_config outside of a runnable context", and _write_progress
    swallows that, so the events vanish silently. That looks exactly like a
    feature working in tests and doing nothing in a real run.
    """
    try:
        from langgraph.config import get_stream_writer
        return get_stream_writer()
    except Exception:
        return None


def _write_progress(writer, stage: str, **fields) -> None:
    if writer is None:
        return
    try:
        writer({"type": "progress", "stage": stage, **fields})
    except Exception:               # progress must never break a stage
        log.debug("progress write failed for stage %s", stage, exc_info=True)


class _ProgressQueue:
    """Collects progress from worker threads and emits it from the node.

    A fetcher runs in a thread pool, and LangGraph's stream writer only works
    inside the node's own context (see :func:`_stream_writer`). So worker
    threads append to a queue here, and the node drains it into the real writer
    as the batch progresses -- which is also where the events get batched, so a
    hundred documents do not become a hundred separate stream frames.

    Draining happens from the node's context: ``asyncio.to_thread`` returns
    control to the event loop before the next drain, so the writer is valid
    again by then.
    """

    def __init__(self, stage: str, writer):
        self.stage = stage
        self.writer = writer
        self.done = 0
        self.total = 0
        self._pending: list[dict] = []

    def emit(self, note: str = "", **fields) -> None:
        """Safe to call from any thread, including a fetch worker."""
        self._pending.append({"note": note, **fields})

    def announce(self, total: int, note: str = "") -> None:
        self.total = total
        self.emit(note=note, done=0, total=total)

    def drain(self, limit: int = 20) -> None:
        """Emit up to `limit` queued events. Call from the node's context."""
        if self.writer is None or not self._pending:
            return
        batch, self._pending = self._pending[:limit], self._pending[limit:]
        for item in batch:
            # The producer may have set an explicit counter; only advance the
            # running total when it did not, so drain never double-counts.
            if "done" in item:
                self.done = int(item["done"])
            else:
                self.done += 1
            _write_progress(self.writer, self.stage, **item)


def _out_of_time(state) -> bool:
    """True once the run has exceeded its soft wall-clock ceiling.

    RESEARCH_TIME_BUDGET_S used to be defined in config and read by nothing,
    which is worse than not having it: a knob that silently does nothing reads
    as a guarantee. Checked at stage boundaries rather than mid-call, so a run
    always stops between units of work and can say why it stopped.
    """
    limit = config.wall_clock_budget_s()
    if limit <= 0:
        return False
    started = state.get("started_at")
    if started is None:
        return False
    return (time.monotonic() - started) > limit


def _budget(state) -> CallBudget:
    budget = state.get("budget")
    if budget is None:
        rounds = state.get("max_rounds")
        budget = CallBudget(total=config.total_llm_budget(rounds))
    return budget


def planner(state):
    plan = make_plan(state["question"])
    pending = [{"source": s["source"], "query": s["query"]}
               for sq in plan.get("subquestions", []) for s in sq["searches"]]
    rounds = state["max_rounds"]

    budget = _budget(state)
    needed = config.budget_estimate(rounds)
    notes: list[str] = []

    if budget.total < needed:
        # Deeper runs cost more per round, so a budget sized for fewer rounds
        # starves synthesis and the report comes back partial rather than deeper.
        log.warning(
            "research depth %d round(s) expects about %d model calls but the "
            "budget is %d; synthesis may be cut short. Raise "
            "RESEARCH_LLM_BUDGET or lower the round count.",
            rounds, needed, budget.total,
        )
        notes.append(
            f"model-call budget ({budget.total}) is below the estimated need "
            f"({needed}) for {rounds} round(s); the report may be partial")
    elif not config.budget_is_explicit():
        log.info("model-call budget auto-sized to %d for %d round(s)",
                 budget.total, rounds)

    update = {"plan": plan, "pending": pending, "round": 1,
              "seen": [f'{p["source"]}|{p["query"]}' for p in pending],
              "budget": budget, "started_at": time.monotonic()}
    if notes:
        update["notes"] = notes
    return update


def route(state):
    """After the planner: fan out over searches, or start retrieval."""
    if state.get("pending"):
        return [Send("search_worker", p) for p in state["pending"]]
    return "retrieve"


def route_after_gap(state):
    """After the gap check: either chase the gaps or finish and synthesise.

    This has to be a separate router from :func:`route`. Sharing one made an
    empty ``pending`` list send the graph back to ``retrieve`` forever instead of
    terminating, because ``retrieve`` and ``analyse`` are re-entered on every
    pass.
    """
    if state.get("pending"):
        return [Send("search_worker", p) for p in state["pending"]]
    return "contradiction_check"


async def search_worker(task):
    try:
        (res, meta), err = await asyncio.to_thread(
            search_full, task["source"], task["query"]), None
    except Exception as e:
        (res, meta), err = ([], {}), str(e)
    bank = list(meta.get("related_questions", []) or [])
    bank += list(meta.get("related_searches", []) or [])
    leads = []
    for ref in (meta.get("ai_overview_refs") or [])[:3]:
        if isinstance(ref, dict) and ref.get("link"):
            leads.append({"title": ref.get("title", ""),
                          "url": ref["link"]})
    return {"raw": res, "log": [{**task, "count": len(res), "error": err}],
            "paa": bank, "ai_leads": leads}


def collect(state):
    return {"evidence": _col(state).list()}


async def _fetch_reporting(fetcher, todo: list, on_doc_done, progress) -> list:
    """Fetch in a worker thread while draining progress from the node's context.

    The fetch itself blocks a thread, so the node awaits it as a task and drains
    the queue on an interval. Draining has to happen here, in the node's own
    context, because LangGraph's stream writer only works there -- a worker
    thread writing to it raises, and the raise would be swallowed.
    """
    task = asyncio.ensure_future(
        asyncio.to_thread(fetcher.fetch_many, todo, on_doc_done))
    while not task.done():
        progress.drain()
        await asyncio.sleep(0.4)
    progress.drain(limit=10_000)          # flush whatever is left
    return task.result()


async def retrieve(state):
    """Download and chunk the real documents behind the search results.

    Reference discovery runs here too: outbound links from documents we actually
    read are better leads than more blind queries, and the ceiling stops it
    becoming crawling.
    """
    question = state["question"]
    plan = state.get("plan") or {}
    notes: list[str] = []

    if not config.fetch_enabled():
        return {"documents": [], "chunks": [], "retrieval_stats":
                {"enabled": False, "note": "full-text retrieval is disabled"}, "notes": ["retrieval disabled"]}

    col = _col(state)
    sources = col.list()
    urls = [s["url"] for s in sources if s.get("url")]

    if _out_of_time(state):
        # Stop between stages, keeping everything already retrieved this run.
        # Returning the empty chunk list here (rather than fetching nothing and
        # reporting zero sources) would erase documents an earlier round paid
        # for, so the accumulated state is passed straight through.
        notes.append(
            f"run exceeded its {config.wall_clock_budget_s():.0f}s wall-clock "
            f"budget; no further sources were fetched this round")
        log.warning("run exceeded its wall-clock budget; retrieval stopped early")
        # Chunks already built in earlier rounds are carried over rather than
        # returned empty: clearing them would discard documents the run already
        # paid to fetch and make the report thinner for no reason.
        carried: list = []
        for _grade, made in (state.get("chunks_by_url") or {}).values():
            carried.extend(made)
        return {"documents": list((state.get("documents_by_url") or {}).values()),
                "chunks": carried, "fetcher": state.get("fetcher"),
                "retrieval_index": dict(state.get("retrieval_index") or {}),
                "documents_by_url": dict(state.get("documents_by_url") or {}),
                "chunks_by_url": dict(state.get("chunks_by_url") or {}),
                "retrieval_stats": {"enabled": False,
                                    "note": "run exceeded its wall-clock budget; "
                                            "no further documents were fetched"},
                "notes": notes}

    # One fetcher per run, held in state, so the fetch budget is per request
    # rather than shared across the whole process.
    fetcher = state.get("fetcher") or Fetcher()

    # Retrieval accumulates across gap-check rounds. The index records what was
    # actually obtained for each URL, so a later round neither re-downloads a
    # document it already has nor re-reports the run as if it had read nothing.
    index: dict = dict(state.get("retrieval_index") or {})
    known_documents: dict = dict(state.get("documents_by_url") or {})

    # Skip anything a previous round already read. Without this, round 2 spends
    # the remaining fetch budget re-requesting URLs it already has and then
    # exhausts it, leaving the later rounds unable to retrieve anything.
    already_read = {url for url, meta in index.items()
                    if meta.get("status") in ("full", "partial")}
    todo = [u for u in urls if u not in already_read]

    # Wikipedia baseline first: zero-cost encyclopedic grounding fetched
    # before any LLM call. Runs once (round 1) and is folded into the same
    # index, so later rounds treat it as already-read.
    if not index and config.wiki_enabled():
        try:
            subs = [sq.get("question", "") for sq in plan.get("subquestions", [])
                    if isinstance(sq, dict) and sq.get("question")]
            wiki_docs = await asyncio.to_thread(
                fetch_wikipedia, question, 3, 6000, subs) if question else []
        except Exception as exc:
            log.info("wikipedia baseline failed: %s", exc)
            wiki_docs = []
        for doc in wiki_docs or []:
            if doc.url not in index and doc.usable:
                index[doc.url] = {
                    "status": doc.status,
                    "method": doc.method,
                    "words": doc.word_count,
                    "title": doc.title or doc.url,
                    "url": doc.url,
                    "limitation": doc.limitation,
                }
                known_documents[doc.url] = doc
        if wiki_docs:
            notes.append(f"wikipedia baseline: {len(wiki_docs)} article(s) pre-loaded")

    # Worker threads cannot write to the stream directly, so they queue here and
    # the node drains the queue while the batch runs.
    progress = _ProgressQueue("retrieve", _stream_writer())
    progress.announce(len(todo), note="downloading sources")

    def on_doc_done(doc) -> None:
        """Called from the fetch worker threads as each document completes."""
        progress.done += 1
        progress.emit(done=progress.done, total=len(todo), url=doc.url,
                      status=doc.status, note=f"read {doc.title or doc.url}")

    try:
        if not todo:
            fetched = []
        else:
            # The progress callback is optional: a fetcher that predates it (or
            # a stub) must not turn a cosmetic feature into a failed stage.
            try:
                fetched = await _fetch_reporting(
                    fetcher, todo, on_doc_done, progress)
            except TypeError:
                log.debug("fetcher does not accept a progress callback")
                fetched = await asyncio.to_thread(fetcher.fetch_many, todo)
    except Exception as exc:
        log.warning("retrieval failed: %s", exc)
        return {"documents": [], "chunks": [], "fetcher": fetcher,
                "retrieval_stats": {"enabled": True, "error": str(exc)[:200]},
                "notes": [f"retrieval failed: {type(exc).__name__}"]}

    fetched = fetched[: config.max_sources()]
    fetched_urls = {d.url for d in fetched} | {d.final_url for d in fetched}

    # Sources we could not read are kept as explicitly-marked metadata-only
    # evidence rather than dropped, so the report can still reference them while
    # stating that only the snippet was available.
    docs = list(fetched)
    unreadable: list[dict] = []
    for result in sources:
        url = result.get("url")
        if not url or url in fetched_urls or url in already_read:
            continue
        unreadable.append(result)
        docs.append(snippet_only_doc(result))

    # Reference discovery: mine outbound links from what we retrieved, bounded.
    references_used: list[str] = []
    if config.references_enabled() and config.references_max() > 0:
        known = set(index) | {s["url"] for s in sources}
        leads = [u for u in candidates(docs, question, known) if u not in index]
        # Forward chase: citing documents are usually newer than the cited
        # work, so highly-cited papers lead to the state of the art.
        for url in forward_candidates(sources, question, known, index):
            if url not in leads:
                leads.append(url)
        # AI-overview reference links: discovered by Google's overview, kept
        # as plain URLs for discovery. Never prose -- the generated text
        # stays out of the evidence path entirely.
        for entry in state.get("ai_leads") or []:
            url = (entry.get("url") or "") if isinstance(entry, dict) else ""
            if url and url not in index and url not in known \
                    and url not in leads:
                leads.append(url)
        leads = leads[:config.references_max()]
        if leads:
            try:
                extra = await asyncio.to_thread(fetcher.fetch_many, leads)
            except Exception:
                extra = []
            extra = [d for d in extra if d.usable]
            if extra:
                docs.extend(extra)
                references_used = [d.url for d in extra]
        notes.append(describe(references_used, max(0, len(leads) - len(references_used))))

    # Fold this round into the per-run index. A status only ever improves, so a
    # source first seen as a snippet and later fetched properly is upgraded
    # rather than being stuck reporting the worse grade.
    for doc in docs:
        previous = index.get(doc.url, {})
        if _STATUS_RANK.get(previous.get("status"), -1) <= _STATUS_RANK.get(doc.status, -1):
            index[doc.url] = {
                "status": doc.status,
                "method": doc.method,
                "words": doc.word_count,
                "title": doc.title or doc.url,
                "url": doc.url,
                "limitation": doc.limitation,
            }
        if doc.usable:
            known_documents[doc.url] = doc

    chunks: list = []
    by_url = {s["url"]: s for s in sources if s.get("url")}
    # Chunks are cached per URL: without this every gap-check round re-ran
    # chunk_document over the whole corpus to rebuild chunks that are
    # byte-identical to last round (~1.4s of CPU per 100 large docs, per round).
    #
    # Keying on URL is safe even though source_id is positional
    # ("S{n}" from enumerate below): known_documents is a dict carried in
    # state, and overwriting an existing key does not move its insertion
    # position, so a document keeps the same n -- and the same S1#2 style chunk
    # ids -- for the whole run. A cached chunk cannot go stale that way.
    #
    # Caveat: cited_by comes from search metadata that can change between
    # rounds, so a cached chunk keeps round 1's citation counts. Left alone
    # deliberately; re-chunking to chase a citation count costs more than the
    # number is worth.
    # The grade is stored with the chunks so an upgrade invalidates the entry
    # and an empty chunk list is still a cache hit rather than a re-chunk.
    chunk_cache: dict = dict(state.get("chunks_by_url") or {})
    for source_id, doc in enumerate(known_documents.values(), 1):
        cached = chunk_cache.get(doc.url)
        # Invalidate on upgrade. Chunks carry retrieval_status, and both
        # _unprocessed_chunks (reprocesses a source whose grade improved) and
        # rank() (penalises metadata_only) read it off the chunk. A cached
        # chunk from before an upgrade would freeze the worse grade, so the
        # source would never be re-read and would stay ranked as a snippet.
        if cached is not None and cached[0] == doc.status:
            chunks.extend(cached[1])
            continue
        meta = by_url.get(doc.url, {})
        made = chunk_document(
            doc.text,
            source_id=f"S{source_id}",
            source_url=doc.url,
            source_title=doc.title,
            source_type=doc.method,
            publication_date=doc.date,
            retrieval_status=doc.status,
            retrieval_limitation=doc.limitation,
            source_publisher=doc.publisher,
            authors=doc.authors,
            doi=doc.doi,
            pages=doc.pages,
            entity=meta.get("entity", ""),
            entity_type=meta.get("entity_type", ""),
            cited_by=int(meta.get("cited_by", 0) or 0),
        )
        chunk_cache[doc.url] = (doc.status, made)
        chunks.extend(made)

    stats = _aggregate_retrieval(index, chunks, references_used)
    if unreadable:
        notes.append(f"{len(unreadable)} source(s) could not be retrieved "
                     f"(robots.txt, paywall, bot protection, rate limit or fetch "
                     f"budget); they are cited as metadata-only")
    log.info("retrieval: %d/%d sources readable (%d full), %d words, %d chunks",
             stats["retrieved"], stats["attempted"], stats["full_text"],
             stats["total_words"], len(chunks))

    return {"documents": list(known_documents.values()), "chunks": chunks,
            "fetcher": fetcher, "retrieval_index": index,
            "documents_by_url": known_documents,
            "chunks_by_url": chunk_cache,
            "retrieval_stats": stats, "notes": notes}


def _aggregate_retrieval(index: dict, chunks: list,
                         references_used: list[str]) -> dict:
    """Statistics for the whole run, derived from the accumulated index.

    Every counter here describes the same set of sources: the index is the single
    record of what was attempted and what was obtained, so the frontend cannot be
    shown round N's numbers alongside round 1's evidence.
    """
    statuses = [meta.get("status") for meta in index.values()]
    return {
        "enabled": True,
        "attempted": len(index),
        "retrieved": sum(1 for s in statuses if s in ("full", "partial")),
        "full_text": statuses.count("full"),
        "partial": statuses.count("partial"),
        "metadata_only": statuses.count("metadata_only"),
        "failed": statuses.count("failed"),
        "total_words": sum(m.get("words", 0) for m in index.values()),
        "chunks": len(chunks),
        "references_followed": len(references_used),
        "sources": [
            {"status": meta.get("status"), "method": meta.get("method"),
             "words": meta.get("words", 0), "title": meta.get("title"),
             "url": meta.get("url"), "limitation": meta.get("limitation")}
            for meta in index.values()
        ],
    }


def _grade_rank(status: str) -> int:
    return {"failed": 0, "metadata_only": 1, "partial": 2, "full": 3}.get(status, 0)


_WORD = re.compile(r"[a-z0-9]+")


def _query_tokens(query: str) -> set[str]:
    return {w for w in _WORD.findall(str(query or "").lower()) if len(w) > 2}


def is_duplicate_query(source: str, query: str, seen: list[str],
                       threshold: float | None = None) -> bool:
    """True when this follow-up repeats a query that already ran.

    Exact ``source|query`` matches are duplicates. Paraphrases are caught with
    a lightweight token-overlap (Jaccard) filter on the same source, so
    "latest LLM benchmarks 2026" does not re-run "2026 latest LLM benchmark".
    """
    key = f"{source}|{query}"
    if key in seen:
        return True
    limit = threshold if threshold is not None else config.gap_dup_threshold()
    mine = _query_tokens(query)
    if not mine:
        return False
    for entry in seen:
        try:
            seen_source, seen_query = entry.split("|", 1)
        except ValueError:
            continue
        if seen_source != source:
            continue
        other = _query_tokens(seen_query)
        if not other:
            continue
        union = mine | other
        jaccard = len(mine & other) / len(union) if union else 0.0
        overlap_min = len(mine & other) / min(len(mine), len(other))
        if jaccard >= limit or overlap_min >= 0.85:
            return True
    return False


def _unprocessed_chunks(state, chunks: list) -> tuple[list, list[dict]]:
    """Chunks whose source still needs evidence extraction.

    Later rounds re-chunk the whole corpus, so without this every extra round
    re-pays for documents that were already processed -- roughly 11 of the 13
    calls a round costs were redundant. A source is re-processed only when its
    retrieval grade has *improved*, which happens when a later fetch finally
    succeeds on something that was previously snippet-only or blocked.
    """
    seen: dict = dict(state.get("extracted_sources") or {})
    if not config.analyse_reuse_sources():
        return chunks, []

    fresh: list = []
    upgraded: list[dict] = []
    for chunk in chunks:
        url = chunk.source_url
        grade = chunk.retrieval_status
        if url not in seen:
            fresh.append(chunk)
            continue
        if _grade_rank(grade) > _grade_rank(seen[url]):
            fresh.append(chunk)
            upgraded.append({"url": url, "from": seen[url], "to": grade})
    return fresh, upgraded


def _attribute_round(notes: list[str], round_number: int) -> list[str]:
    """Name the round an evidence-integrity failure happened in.

    Without this the note read "6 of 7 passage batches" on a run where round 1
    had already extracted 83 records from 40 batches. That is technically true
    of the round it came from and badly misleading about the report as a whole:
    a reader concludes the document is thin when it is not. Naming the round
    keeps the warning without overstating it.
    """
    from .extract import INTEGRITY_PREFIX

    out = []
    for note in notes or []:
        text = str(note)
        if text.startswith(INTEGRITY_PREFIX):
            out.append(f"{INTEGRITY_PREFIX}in round {round_number}, "
                       f"{text[len(INTEGRITY_PREFIX):]}")
        else:
            out.append(text)
    return out


async def analyse(state):
    """Rank passages deterministically, then extract evidence in batched LLM calls.

    Findings from earlier rounds are always carried forward. Every early exit
    still merges what already exists, otherwise a round that finds nothing would
    silently discard evidence an earlier round had already paid for.
    """
    question = state["question"]
    chunks = state.get("chunks") or []
    notes: list[str] = []
    previous = state.get("extracted") or []

    def merge(records: list) -> list:
        if not previous:
            return records
        merged: dict = {}
        for record in previous:
            merged[f"{record.claim[:80].lower()}|{record.source_url}"] = record
        for record in records:
            merged.setdefault(f"{record.claim[:80].lower()}|{record.source_url}",
                              record)
        combined = sorted(merged.values(), key=lambda r: (r.rank, -r.confidence))
        cap = config.max_evidence_records()
        if len(combined) > cap:
            notes.append(f"evidence set trimmed from {len(combined)} to {cap} records")
            combined = combined[:cap]
        for index, record in enumerate(combined, 1):
            record.evidence_id = f"E{index}"
        return combined

    if not chunks or not config.analysis_enabled():
        notes.append("no retrieved passages available for evidence extraction")
        return {"extracted": merge([]), "source_synth": {}, "notes": notes}

    candidates, upgraded = _unprocessed_chunks(state, chunks)
    if upgraded:
        notes.append(f"re-processing {len(upgraded)} source(s) whose retrieval "
                     f"improved since they were last read")
    skipped = len(chunks) - len(candidates)
    if skipped and not candidates:
        notes.append("no new or improved sources this round; reusing existing "
                     "evidence rather than re-extracting the same documents")
        return {"extracted": merge([]), "source_synth": {}, "notes": notes}
    if skipped:
        notes.append(f"skipped {skipped} passage(s) from sources already processed")

    budget = _budget(state)
    if _out_of_time(state):
        notes.append(
            f"run exceeded its {config.wall_clock_budget_s():.0f}s wall-clock "
            f"budget; {len(candidates)} candidate passage(s) were left "
            f"unanalysed rather than extending the run")
        log.warning("run exceeded its wall-clock budget; analysis stopped early")
        return {"extracted": merge([]), "source_synth": {}, "notes": notes}

    _progress("analyse", done=0, total=len(candidates),
              note=f"ranking {len(candidates)} passages")
    scored = diversify(rank(candidates, question, state.get("plan")))
    # Belt-and-braces: the ranked set must contain only newly discovered
    # sources. Candidates are already fresh, but rank/diversify operate on
    # ids that could theoretically reintroduce an old URL, so filter the
    # ranked output down to fresh URLs before any extraction call is paid for.
    fresh_urls = {c.source_url for c in candidates}
    before_rank = len(scored)
    scored = [s for s in scored if s.chunk.source_url in fresh_urls]
    if len(scored) < before_rank:
        notes.append(f"filtered {before_rank - len(scored)} ranked passage(s) "
                     f"from already-processed sources before extraction")
    if not scored:
        notes.append("retrieved passages were not relevant enough to extract from")
        return {"extracted": merge([]), "source_synth": {}, "notes": notes}

    _progress("analyse", done=0, total=len(scored),
              note=f"extracting evidence from {len(scored)} passages")
    records, extraction_notes = await asyncio.to_thread(
        extract_evidence, scored, question, state.get("plan"), budget)
    notes.extend(_attribute_round(extraction_notes, state.get("round") or 1))
    _progress("analyse", done=len(scored), total=len(scored),
              note=f"{len(records)} verified record(s) extracted")

    # Remember what was read, and how well, so the next round can skip it.
    tracked = dict(state.get("extracted_sources") or {})
    for chunk in candidates:
        tracked.setdefault(chunk.source_url, chunk.retrieval_status)
    for record in records:
        tracked[record.source_url] = record.retrieval_status

    records = merge(records)

    synth: dict = {}
    if records and budget.available():
        # Only assess sources that have not been assessed yet.
        done: dict = dict(state.get("source_synth") or {})
        todo = [r for r in records if r.source_id and r.source_id not in done]
        if todo:
            sources_todo = len({r.source_id for r in todo})
            _progress("analyse", done=0, total=sources_todo,
                      note=f"assessing {sources_todo} source(s)")
            fresh_synth = await asyncio.to_thread(synthesise_sources, todo, budget)
            done.update(fresh_synth)
            synth = done
            _progress("analyse", done=sources_todo, total=sources_todo,
                      note=f"{len(fresh_synth)} source(s) assessed")

    log.info("analysis: %d/%d passages processed (%d already seen) -> %d new "
             "records, %d total", len(scored), len(chunks), skipped,
             len(records) - len(previous), len(records))
    return {"extracted": records, "source_synth": synth,
            "extracted_sources": tracked, "notes": notes}


async def gap_check(state):
    rnd, seen = state["round"], list(state["seen"])
    if rnd >= state["max_rounds"]:
        return {"gap": {"sufficient": False, "missing": "round limit reached"}, "pending": []}

    records = state.get("extracted") or []
    context = _col(state).context()
    if records:
        # Prefer real extracted evidence over snippets when judging sufficiency.
        lines = [f"[{r.evidence_id}] {r.claim} ({r.source_title}, {r.source_url})"
                 for r in records[:60]]
        context = "\n".join(lines)
    prompt = (source_context(state["question"], state.get("plan"))
              + f"\n\nEVIDENCE SO FAR:\n{context}")

    # Under-evidenced dimensions are named explicitly so recovery targets
    # them instead of accepting "missing evidence" and moving on.
    plan = state.get("plan") or {}
    gap_dims = question_dimensions(state["question"])
    if not gap_dims and plan:
        gap_dims = [sq.get("question", "") for sq in plan.get("subquestions", [])
                    if sq.get("question")]
    holes = evidence_holes(records, gap_dims)
    if holes:
        prompt += ("\n\nUNDER-EVIDENCED DIMENSIONS (prioritize follow-up searches "
                   "that fill exactly these):\n" + "\n".join(f"- {h}" for h in holes))
    asked = uncovered_questions(state.get("paa") or [], records)
    if asked:
        prompt += ("\n\nASKED ELSEWHERE BUT UNANSWERED HERE (real follow-up "
                   "questions from related searches; prefer these over "
                   "inventing queries):\n" + "\n".join(f"- {q}" for q in asked))

    try:
        gap = await asyncio.to_thread(ask, GAP_SYS, prompt, True)
    except Exception as exc:
        log.warning("gap_check failed: %s", exc)
        return {"gap": {"sufficient": False, "missing": f"gap check unavailable: {exc}"},
                "pending": [], "round": rnd + 1}

    if not isinstance(gap, dict):
        gap = {"sufficient": False, "missing": "malformed gap-check response", "follow_ups": []}

    new = []
    dropped = 0
    if not gap.get("sufficient"):
        for f in (gap.get("follow_ups") or [])[:3]:
            if not isinstance(f, dict):
                continue
            if f.get("source") not in SOURCES or not f.get("query"):
                continue
            if is_duplicate_query(f["source"], f["query"], seen):
                dropped += 1
                continue
            new.append({"source": f["source"], "query": f["query"]})
            seen.append(f'{f.get("source")}|{f.get("query")}')
    if dropped:
        gap = dict(gap, dropped_duplicates=dropped)
    return {"gap": gap, "pending": new, "seen": seen, "round": rnd + 1}


async def contradiction_check(state):
    """Prefer disagreements between extracted findings; fall back to snippets."""
    records = state.get("extracted") or []
    if len(records) >= 2:
        conflicts = await asyncio.to_thread(find_conflicts, records, _budget(state))
        return {"contradictions": conflicts or []}
    return {"contradictions": _col(state).find_contradictions()}


async def synthesizer(state):
    records = list(state.get("extracted") or [])
    records += findings_from_contradictions(state.get("contradictions") or [], records)

    notes = list(state.get("notes") or [])
    question, col = state["question"], _col(state)
    contradictions = state.get("contradictions") or []
    result = await asyncio.to_thread(
        generate_report, question, col, contradictions,
        records, state.get("source_synth") or {}, notes, state.get("plan"), _budget(state))
    report = result.markdown
    try:
        report_html = render_markdown_html(report) if report else ""
    except Exception as exc:
        log.warning("server-side HTML compile failed: %s", exc)
        report_html = ""

    stats = statistics(records) if records else {}
    export = {}
    if report:
        try:
            export = save_export(question, report, stats,
                                 state.get("retrieval_stats") or {})
        except Exception as exc:
            log.warning("markdown export failed: %s", exc)

    # The completion state is part of the result, not a detail: a report that
    # lost sections is a partial research run and must be labelled as one.
    completion = result.as_dict()
    completion["headline"] = status_headline(result.status, result.missing_sections)
    if result.status == "partial":
        log.warning("research PARTIAL: %d section(s) missing (%s)",
                    len(result.missing_sections),
                    "; ".join(sorted(set(result.missing_sections))[:4]))
    elif result.status == "failed":
        log.error("research FAILED: no usable report could be generated")

    return {"report": report, "report_html": report_html,
              "sections": [(h, m) for h, m in (result.sections or [])],
              "evidence_stats": stats, "export": export,
              "status": completion}


def build_graph():
    g = StateGraph(State)
    for name, fn in [("planner", planner), ("search_worker", search_worker),
                     ("collect", collect), ("retrieve", retrieve),
                     ("analyse", analyse), ("gap_check", gap_check),
                     ("contradiction_check", contradiction_check),
                     ("synthesizer", synthesizer)]:
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", route, ["search_worker", "retrieve"])
    g.add_edge("search_worker", "collect")
    g.add_edge("collect", "retrieve")
    g.add_conditional_edges("gap_check", route_after_gap,
                            ["search_worker", "contradiction_check"])
    g.add_edge("retrieve", "analyse")
    g.add_edge("analyse", "gap_check")
    g.add_edge("contradiction_check", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


graph = build_graph()