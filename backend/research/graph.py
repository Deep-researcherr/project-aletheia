"""LangGraph orchestration.

    START -> planner -> initial_search
          -> [SerpAPI searches + Wikipedia in parallel]
          -> collect -> retrieve -> analyse -> gap_check
                                      --(gaps)--> search_worker ...
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
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import config
from .chunker import chunk_document
from .collector import Collector
from .evidence import statistics
from .export import save as save_export
from .extract import (
    extract_evidence,
    find_conflicts,
    findings_from_contradictions,
    synthesise_sources,
)
from .fetch import FAILED, Fetcher, FULL, METADATA, PARTIAL, snippet_only_doc
from .llm import CallBudget, ask
from .markdown_html import render_markdown_html
from .planner import SOURCES, make_plan
from .references import candidates, describe
from .relevance import diversify, rank, source_context
from .report import generate_report, status_headline, write_report
from .searcher import search
from .wiki import fetch_wikipedia

log = logging.getLogger(__name__)

# Strength order for a retrieval grade. A source only ever improves as more
# rounds succeed, so the index keeps the best result seen for each URL.
_STATUS_RANK = {
    FAILED: 0,
    METADATA: 1,
    PARTIAL: 2,
    FULL: 3,
}

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
    pending: list
    seen: list
    raw: Annotated[list, operator.add]
    log: Annotated[list, operator.add]
    wiki_documents: Annotated[list, operator.add]
    evidence: list
    gap: dict
    contradictions: list
    report: str

    # --- deep-synthesis state ---
    documents: list
    chunks: list
    extracted: list
    source_synth: dict
    retrieval_stats: dict
    notes: Annotated[list, operator.add]
    budget: object
    retrieval_index: dict
    documents_by_url: dict
    status: dict
    fetcher: object
    evidence_stats: dict
    extracted_sources: dict
    export: dict


def _col(state) -> Collector:
    col = Collector()
    col.add([dict(r) for r in state.get("raw", [])])
    return col


def _budget(state) -> CallBudget:
    budget = state.get("budget")
    if budget is None:
        rounds = state.get("max_rounds")
        budget = CallBudget(total=config.total_llm_budget(rounds))
    return budget


def planner(state):
    plan = make_plan(state["question"])

    pending = [
        {
            "source": s["source"],
            "query": s["query"],
        }
        for sq in plan.get("subquestions", [])
        for s in sq["searches"]
    ]

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
            rounds,
            needed,
            budget.total,
        )

        notes.append(
            f"model-call budget ({budget.total}) is below the estimated need "
            f"({needed}) for {rounds} round(s); the report may be partial"
        )

    elif not config.budget_is_explicit():
        log.info(
            "model-call budget auto-sized to %d for %d round(s)",
            budget.total,
            rounds,
        )

    update = {
        "plan": plan,
        "pending": pending,
        "round": 1,
        "seen": [
            f'{p["source"]}|{p["query"]}'
            for p in pending
        ],
        "budget": budget,
    }

    if notes:
        update["notes"] = notes

    return update


def route(state):
    """After planner, run the initial search round."""

    if state.get("pending"):
        return "initial_search"

    return "retrieve"


async def _run_initial_serp_search(task):
    """Run one initial SerpAPI search without blocking the event loop."""

    try:
        results = await asyncio.to_thread(
            search,
            task["source"],
            task["query"],
        )

        return {
            "results": results or [],
            "log": {
                **task,
                "count": len(results or []),
                "error": None,
            },
        }

    except Exception as exc:
        return {
            "results": [],
            "log": {
                **task,
                "count": 0,
                "error": str(exc),
            },
        }


async def initial_search(state):
    """Run the entire first search round concurrently.

    All planned SerpAPI searches and the Wikipedia baseline are started
    together. Wikipedia is fetched once and carried into retrieval as a
    full-text document, so it does not need to be downloaded again later.
    """

    pending = state.get("pending") or []
    question = state.get("question", "")

    tasks = [
        _run_initial_serp_search(task)
        for task in pending
    ]

    # Start Wikipedia at the same time as the SerpAPI tasks.
    if config.wiki_enabled() and question:
        wiki_task = asyncio.to_thread(
            fetch_wikipedia,
            question,
        )
    else:
        wiki_task = None

    if wiki_task is not None:
        search_results, wiki_docs = await asyncio.gather(
            asyncio.gather(*tasks),
            wiki_task,
        )
    else:
        search_results = await asyncio.gather(*tasks)
        wiki_docs = []

    raw: list = []
    logs: list = []

    for item in search_results:
        raw.extend(item["results"])
        logs.append(item["log"])

    notes: list[str] = []

    if wiki_docs:
        notes.append(
            f"wikipedia baseline: {len(wiki_docs)} article(s) pre-loaded"
        )

    return {
        "raw": raw,
        "log": logs,
        "wiki_documents": wiki_docs or [],
        "notes": notes,
    }


def route_after_gap(state):
    """After the gap check: either chase the gaps or finish and synthesise.

    This has to be a separate router from :func:`route`. Sharing one made an
    empty ``pending`` list send the graph back to ``retrieve`` forever instead
    of terminating, because ``retrieve`` and ``analyse`` are re-entered on every
    pass.
    """

    if state.get("pending"):
        return [
            Send("search_worker", p)
            for p in state["pending"]
        ]

    return "contradiction_check"


async def search_worker(task):
    try:
        res, err = (
            await asyncio.to_thread(
                search,
                task["source"],
                task["query"],
            ),
            None,
        )

    except Exception as e:
        res, err = [], str(e)

    return {
        "raw": res,
        "log": [
            {
                **task,
                "count": len(res),
                "error": err,
            }
        ],
    }


def collect(state):
    return {
        "evidence": _col(state).list()
    }


async def retrieve(state):
    """Download and chunk the real documents behind the search results.

    Reference discovery runs here too: outbound links from documents we actually
    read are better leads than more blind queries, and the ceiling stops it
    becoming crawling.
    """

    question = state["question"]
    notes: list[str] = []

    if not config.fetch_enabled():
        return {
            "documents": [],
            "chunks": [],
            "retrieval_stats": {
                "enabled": False,
                "note": "full-text retrieval is disabled",
            },
            "notes": ["retrieval disabled"],
        }

    col = _col(state)
    sources = col.list()
    urls = [
        s["url"]
        for s in sources
        if s.get("url")
    ]

    # One fetcher per run, held in state, so the fetch budget is per request
    # rather than shared across the whole process.
    fetcher = state.get("fetcher") or Fetcher()

    # Retrieval accumulates across gap-check rounds. The index records what was
    # actually obtained for each URL, so a later round neither re-downloads a
    # document it already has nor re-reports the run as if it had read nothing.
    index: dict = dict(
        state.get("retrieval_index") or {}
    )

    known_documents: dict = dict(
        state.get("documents_by_url") or {}
    )

    # Wikipedia was fetched concurrently with the initial SerpAPI searches.
    # It is already full-text, so it does not need to pass through Fetcher.
    wiki_docs = state.get("wiki_documents") or []

    for doc in wiki_docs:
        if not doc.usable:
            continue

        previous = index.get(doc.url, {})

        if _STATUS_RANK.get(
            previous.get("status"),
            -1,
        ) <= _STATUS_RANK.get(
            doc.status,
            -1,
        ):
            index[doc.url] = {
                "status": doc.status,
                "method": doc.method,
                "words": doc.word_count,
                "title": doc.title or doc.url,
                "url": doc.url,
                "limitation": doc.limitation,
            }

        known_documents[doc.url] = doc

    # Skip anything a previous round already read. Without this, round 2 spends
    # the remaining fetch budget re-requesting URLs it already has and then
    # exhausts it, leaving the later rounds unable to retrieve anything.
    already_read = {
        url
        for url, meta in index.items()
        if meta.get("status") in ("full", "partial")
    }

    todo = [
        u
        for u in urls
        if u not in already_read
    ]

    try:
        fetched = (
            await asyncio.to_thread(
                fetcher.fetch_many,
                todo,
            )
            if todo
            else []
        )

    except Exception as exc:
        log.warning("retrieval failed: %s", exc)

        return {
            "documents": [],
            "chunks": [],
            "fetcher": fetcher,
            "retrieval_stats": {
                "enabled": True,
                "error": str(exc)[:200],
            },
            "notes": [
                f"retrieval failed: {type(exc).__name__}"
            ],
        }

    fetched = fetched[: config.max_sources()]

    fetched_urls = {
        d.url
        for d in fetched
    } | {
        d.final_url
        for d in fetched
    }

    # Sources we could not read are kept as explicitly-marked metadata-only
    # evidence rather than dropped, so the report can still reference them while
    # stating that only the snippet was available.
    docs = list(fetched)
    unreadable: list[dict] = []

    for result in sources:
        url = result.get("url")

        if (
            not url
            or url in fetched_urls
            or url in already_read
        ):
            continue

        unreadable.append(result)
        docs.append(
            snippet_only_doc(result)
        )

    # Reference discovery: mine outbound links from what we retrieved, bounded.
    references_used: list[str] = []

    if (
        config.references_enabled()
        and config.references_max() > 0
    ):
        known = set(index) | {
            s["url"]
            for s in sources
        }

        leads = [
            u
            for u in candidates(
                docs,
                question,
                known,
            )
            if u not in index
        ]

        if leads:
            try:
                extra = await asyncio.to_thread(
                    fetcher.fetch_many,
                    leads,
                )
            except Exception:
                extra = []

            extra = [
                d
                for d in extra
                if d.usable
            ]

            if extra:
                docs.extend(extra)
                references_used = [
                    d.url
                    for d in extra
                ]

        notes.append(
            describe(
                references_used,
                max(
                    0,
                    len(leads) - len(references_used),
                ),
            )
        )

    # Fold this round into the per-run index. A status only ever improves, so a
    # source first seen as a snippet and later fetched properly is upgraded
    # rather than being stuck reporting the worse grade.
    for doc in docs:
        previous = index.get(
            doc.url,
            {},
        )

        if _STATUS_RANK.get(
            previous.get("status"),
            -1,
        ) <= _STATUS_RANK.get(
            doc.status,
            -1,
        ):
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

    for source_id, doc in enumerate(
        known_documents.values(),
        1,
    ):
        chunks.extend(
            chunk_document(
                doc.text,
                source_id=f"S{source_id}",
                source_url=doc.url,
                source_title=doc.title,
                source_type=doc.method,
                publication_date=doc.date,
                retrieval_status=doc.status,
                source_publisher=doc.publisher,
                authors=doc.authors,
                doi=doc.doi,
                pages=doc.pages,
            )
        )

    stats = _aggregate_retrieval(
        index,
        chunks,
        references_used,
    )

    if unreadable:
        notes.append(
            f"{len(unreadable)} source(s) could not be retrieved "
            f"(robots.txt, paywall, bot protection, rate limit or fetch "
            f"budget); they are cited as metadata-only"
        )

    log.info(
        "retrieval: %d/%d sources readable (%d full), %d words, %d chunks",
        stats["retrieved"],
        stats["attempted"],
        stats["full_text"],
        stats["total_words"],
        len(chunks),
    )

    return {
        "documents": list(
            known_documents.values()
        ),
        "chunks": chunks,
        "fetcher": fetcher,
        "retrieval_index": index,
        "documents_by_url": known_documents,
        "retrieval_stats": stats,
        "notes": notes,
    }


def _aggregate_retrieval(
    index: dict,
    chunks: list,
    references_used: list[str],
) -> dict:
    """Statistics for the whole run, derived from the accumulated index.

    Every counter here describes the same set of sources: the index is the single
    record of what was attempted and what was obtained, so the frontend cannot be
    shown round N's numbers alongside round 1's evidence.
    """

    statuses = [
        meta.get("status")
        for meta in index.values()
    ]

    return {
        "enabled": True,
        "attempted": len(index),
        "retrieved": sum(
            1
            for s in statuses
            if s in ("full", "partial")
        ),
        "full_text": statuses.count("full"),
        "partial": statuses.count("partial"),
        "metadata_only": statuses.count("metadata_only"),
        "failed": statuses.count("failed"),
        "total_words": sum(
            m.get("words", 0)
            for m in index.values()
        ),
        "chunks": len(chunks),
        "references_followed": len(references_used),
        "sources": [
            {
                "status": meta.get("status"),
                "method": meta.get("method"),
                "words": meta.get("words", 0),
                "title": meta.get("title"),
                "url": meta.get("url"),
                "limitation": meta.get("limitation"),
            }
            for meta in index.values()
        ],
    }


def _grade_rank(status: str) -> int:
    return {
        "failed": 0,
        "metadata_only": 1,
        "partial": 2,
        "full": 3,
    }.get(status, 0)


_WORD = re.compile(r"[a-z0-9]+")


def _query_tokens(query: str) -> set[str]:
    return {
        w
        for w in _WORD.findall(
            str(query or "").lower()
        )
        if len(w) > 2
    }


def is_duplicate_query(
    source: str,
    query: str,
    seen: list[str],
    threshold: float | None = None,
) -> bool:
    """True when this follow-up repeats a query that already ran.

    Exact ``source|query`` matches are duplicates. Paraphrases are caught with
    a lightweight token-overlap (Jaccard) filter on the same source, so
    "latest LLM benchmarks 2026" does not re-run "2026 latest LLM benchmark".
    """

    key = f"{source}|{query}"

    if key in seen:
        return True

    limit = (
        threshold
        if threshold is not None
        else config.gap_dup_threshold()
    )

    mine = _query_tokens(query)

    if not mine:
        return False

    for entry in seen:
        try:
            seen_source, seen_query = entry.split(
                "|",
                1,
            )
        except ValueError:
            continue

        if seen_source != source:
            continue

        other = _query_tokens(seen_query)

        if not other:
            continue

        union = mine | other

        jaccard = (
            len(mine & other) / len(union)
            if union
            else 0.0
        )

        overlap_min = (
            len(mine & other) / min(
                len(mine),
                len(other),
            )
        )

        if (
            jaccard >= limit
            or overlap_min >= 0.85
        ):
            return True

    return False


def _unprocessed_chunks(
    state,
    chunks: list,
) -> tuple[list, list[dict]]:
    """Chunks whose source still needs evidence extraction.

    Later rounds re-chunk the whole corpus, so without this every extra round
    re-pays for documents that were already processed -- roughly 11 of the 13
    calls a round costs were redundant. A source is re-processed only when its
    retrieval grade has *improved*, which happens when a later fetch finally
    succeeds on something that was previously snippet-only or blocked.
    """

    seen: dict = dict(
        state.get("extracted_sources") or {}
    )

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
            upgraded.append(
                {
                    "url": url,
                    "from": seen[url],
                    "to": grade,
                }
            )

    return fresh, upgraded


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
            merged[
                f"{record.claim[:80].lower()}|{record.source_url}"
            ] = record

        for record in records:
            merged.setdefault(
                f"{record.claim[:80].lower()}|{record.source_url}",
                record,
            )

        combined = sorted(
            merged.values(),
            key=lambda r: (
                r.rank,
                -r.confidence,
            ),
        )

        cap = config.max_evidence_records()

        if len(combined) > cap:
            notes.append(
                f"evidence set trimmed from {len(combined)} to {cap} records"
            )
            combined = combined[:cap]

        for index, record in enumerate(
            combined,
            1,
        ):
            record.evidence_id = f"E{index}"

        return combined

    if not chunks or not config.analysis_enabled():
        notes.append(
            "no retrieved passages available for evidence extraction"
        )

        return {
            "extracted": merge([]),
            "source_synth": {},
            "notes": notes,
        }

    candidates, upgraded = _unprocessed_chunks(
        state,
        chunks,
    )

    if upgraded:
        notes.append(
            f"re-processing {len(upgraded)} source(s) whose retrieval "
            f"improved since they were last read"
        )

    skipped = len(chunks) - len(candidates)

    if skipped and not candidates:
        notes.append(
            "no new or improved sources this round; reusing existing "
            "evidence rather than re-extracting the same documents"
        )

        return {
            "extracted": merge([]),
            "source_synth": {},
            "notes": notes,
        }

    if skipped:
        notes.append(
            f"skipped {skipped} passage(s) from sources already processed"
        )

    budget = _budget(state)

    scored = diversify(
        rank(
            candidates,
            question,
            state.get("plan"),
        )
    )

    # Belt-and-braces: the ranked set must contain only newly discovered
    # sources. Candidates are already fresh, but rank/diversify operate on
    # ids that could theoretically reintroduce an old URL, so filter the
    # ranked output down to fresh URLs before any extraction call is paid for.
    fresh_urls = {
        c.source_url
        for c in candidates
    }

    before_rank = len(scored)

    scored = [
        s
        for s in scored
        if s.chunk.source_url in fresh_urls
    ]

    if len(scored) < before_rank:
        notes.append(
            f"filtered {before_rank - len(scored)} ranked passage(s) "
            f"from already-processed sources before extraction"
        )

    if not scored:
        notes.append(
            "retrieved passages were not relevant enough to extract from"
        )

        return {
            "extracted": merge([]),
            "source_synth": {},
            "notes": notes,
        }

    records, extraction_notes = await asyncio.to_thread(
        extract_evidence,
        scored,
        question,
        state.get("plan"),
        budget,
    )

    notes.extend(extraction_notes)

    # Remember what was read, and how well, so the next round can skip it.
    tracked = dict(
        state.get("extracted_sources") or {}
    )

    for chunk in candidates:
        tracked.setdefault(
            chunk.source_url,
            chunk.retrieval_status,
        )

    for record in records:
        tracked[
            record.source_url
        ] = record.retrieval_status

    records = merge(records)

    synth: dict = {}

    if records and budget.available():
        # Only assess sources that have not been assessed yet.
        done: dict = dict(
            state.get("source_synth") or {}
        )

        todo = [
            r
            for r in records
            if r.source_id
            and r.source_id not in done
        ]

        if todo:
            fresh_synth = await asyncio.to_thread(
                synthesise_sources,
                todo,
                budget,
            )

            done.update(fresh_synth)
            synth = done

    log.info(
        "analysis: %d/%d passages processed (%d already seen) -> %d new "
        "records, %d total",
        len(scored),
        len(chunks),
        skipped,
        len(records) - len(previous),
        len(records),
    )

    return {
        "extracted": records,
        "source_synth": synth,
        "extracted_sources": tracked,
        "notes": notes,
    }


async def gap_check(state):
    rnd, seen = (
        state["round"],
        list(state["seen"]),
    )

    if rnd >= state["max_rounds"]:
        return {
            "gap": {
                "sufficient": False,
                "missing": "round limit reached",
            },
            "pending": [],
        }

    records = state.get("extracted") or []

    context = _col(state).context()

    if records:
        # Prefer real extracted evidence over snippets when judging sufficiency.
        lines = [
            f"[{r.evidence_id}] {r.claim} "
            f"({r.source_title}, {r.source_url})"
            for r in records[:60]
        ]

        context = "\n".join(lines)

    prompt = (
        source_context(
            state["question"],
            state.get("plan"),
        )
        + f"\n\nEVIDENCE SO FAR:\n{context}"
    )

    try:
        gap = await asyncio.to_thread(
            ask,
            GAP_SYS,
            prompt,
            True,
        )

    except Exception as exc:
        log.warning(
            "gap_check failed: %s",
            exc,
        )

        return {
            "gap": {
                "sufficient": False,
                "missing": f"gap check unavailable: {exc}",
            },
            "pending": [],
            "round": rnd + 1,
        }

    if not isinstance(gap, dict):
        gap = {
            "sufficient": False,
            "missing": "malformed gap-check response",
            "follow_ups": [],
        }

    new = []
    dropped = 0

    if not gap.get("sufficient"):
        for f in (
            gap.get("follow_ups") or []
        )[:3]:
            if not isinstance(f, dict):
                continue

            if (
                f.get("source") not in SOURCES
                or not f.get("query")
            ):
                continue

            if is_duplicate_query(
                f["source"],
                f["query"],
                seen,
            ):
                dropped += 1
                continue

            new.append(
                {
                    "source": f["source"],
                    "query": f["query"],
                }
            )

            seen.append(
                f'{f.get("source")}|{f.get("query")}'
            )

    if dropped:
        gap = dict(
            gap,
            dropped_duplicates=dropped,
        )

    return {
        "gap": gap,
        "pending": new,
        "seen": seen,
        "round": rnd + 1,
    }


async def contradiction_check(state):
    """Prefer disagreements between extracted findings; fall back to snippets."""

    records = state.get("extracted") or []

    if len(records) >= 2:
        conflicts = await asyncio.to_thread(
            find_conflicts,
            records,
            _budget(state),
        )

        return {
            "contradictions": conflicts or []
        }

    return {
        "contradictions": _col(state).find_contradictions()
    }


async def synthesizer(state):
    records = list(
        state.get("extracted") or []
    )

    records += findings_from_contradictions(
        state.get("contradictions") or [],
        records,
    )

    notes = list(
        state.get("notes") or []
    )

    question = state["question"]
    col = _col(state)

    contradictions = state.get("contradictions") or []

    result = await asyncio.to_thread(
        generate_report,
        question,
        col,
        contradictions,
        records,
        state.get("source_synth") or {},
        notes,
        state.get("plan"),
        _budget(state),
    )

    report = result.markdown

    try:
        report_html = (
            render_markdown_html(report)
            if report
            else ""
        )

    except Exception as exc:
        log.warning(
            "server-side HTML compile failed: %s",
            exc,
        )
        report_html = ""

    stats = (
        statistics(records)
        if records
        else {}
    )

    export = {}

    if report:
        try:
            export = save_export(
                question,
                report,
                stats,
                state.get("retrieval_stats") or {},
            )

        except Exception as exc:
            log.warning(
                "markdown export failed: %s",
                exc,
            )

    # The completion state is part of the result, not a detail: a report that
    # lost sections is a partial research run and must be labelled as one.
    completion = result.as_dict()

    completion["headline"] = status_headline(
        result.status,
        result.missing_sections,
    )

    if result.status == "partial":
        log.warning(
            "research PARTIAL: %d section(s) missing (%s)",
            len(result.missing_sections),
            "; ".join(
                sorted(
                    set(result.missing_sections)
                )[:4]
            ),
        )

    elif result.status == "failed":
        log.error(
            "research FAILED: no usable report could be generated"
        )

    return {
        "report": report,
        "report_html": report_html,
        "evidence_stats": stats,
        "export": export,
        "status": completion,
    }


def build_graph():
    g = StateGraph(State)

    for name, fn in [
        ("planner", planner),
        ("initial_search", initial_search),
        ("search_worker", search_worker),
        ("collect", collect),
        ("retrieve", retrieve),
        ("analyse", analyse),
        ("gap_check", gap_check),
        ("contradiction_check", contradiction_check),
        ("synthesizer", synthesizer),
    ]:
        g.add_node(name, fn)

    g.add_edge(
        START,
        "planner",
    )

    g.add_conditional_edges(
        "planner",
        route,
        [
            "initial_search",
            "retrieve",
        ],
    )

    # Initial SerpAPI + Wikipedia round is joined here.
    # initial_search only completes after ALL initial searches and Wikipedia
    # have completed.
    g.add_edge(
        "initial_search",
        "collect",
    )

    g.add_edge(
        "collect",
        "retrieve",
    )

    # Later gap-check rounds continue using the existing search_worker fan-out.
    g.add_conditional_edges(
        "gap_check",
        route_after_gap,
        [
            "search_worker",
            "contradiction_check",
        ],
    )

    g.add_edge(
        "search_worker",
        "collect",
    )

    g.add_edge(
        "retrieve",
        "analyse",
    )

    g.add_edge(
        "analyse",
        "gap_check",
    )

    g.add_edge(
        "contradiction_check",
        "synthesizer",
    )

    g.add_edge(
        "synthesizer",
        END,
    )

    return g.compile()


graph = build_graph()