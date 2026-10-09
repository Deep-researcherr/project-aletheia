"""Streaming bridge from the graph to the SSE contract.

The event types the original pipeline emitted -- plan, results, evidence, gap,
contradictions, report -- are all still emitted, with the same payloads. The
retrieval and analysis stages add new events around them so the browser can show
what the pipeline is actually doing instead of sitting on a spinner for minutes.

"evidence" still reports the number of collected sources, as before; the deeper
structured findings are reported under "analysis" so nothing existing shifts
meaning underneath a client that already works.
"""
from .graph import graph
from . import config
from .runlog import begin as _begin_run, record as _record_event

import logging

log = logging.getLogger(__name__)


def _section_html(markdown: str) -> str:
    from .markdown_html import render_markdown_html
    return render_markdown_html(markdown) if markdown else ""


async def deep_research(question: str, max_rounds: int = 3):
    evidence = []
    stats: dict = {}
    _begin_run(question, max_rounds)

    # Say what the run depends on, and what is left of each budget, before it
    # spends anything. Announced once per run rather than per stage, so the log
    # stays readable, and never allowed to raise: a status line must not be able
    # to take down a research run.
    services = []
    try:
        from . import services as services_module
        services = await services_module.announce(question)
    except Exception as exc:
        log.debug("service announcement unavailable: %s", exc)

    init = {
        "question": question,
        "max_rounds": max_rounds,
        "raw": [],
        "log": [],
        "notes": [],
        "paa": [],
        "ai_leads": [],
    }

    def emit(event: dict):
        """Record for the PDF export, then stream to the browser."""
        _record_event(question, event)
        return event

    if services:
        yield emit({"type": "services",
                    "data": [row.as_dict() for row in services]})

    # "custom" carries progress events written from inside a stage via
    # get_stream_writer(). retrieve and analyse are the two longest stages and
    # used to emit nothing until they finished, so the browser sat on a frozen
    # "Reading pages" for minutes with no idea anything was happening.
    async for mode, chunk in graph.astream(init, stream_mode=["updates", "custom"]):
        if mode == "custom":
            # Already in the run-log event shape; recorded like any other.
            yield emit(chunk)
            continue

        for node, upd in chunk.items():
            if node == "planner":
                yield emit({"type": "plan", "data": upd["plan"]})

            elif node == "search_worker":
                yield emit({"type": "results", **upd["log"][0]})

            elif node == "collect":
                evidence = upd["evidence"]
                yield emit({"type": "evidence", "count": len(evidence)})

            elif node == "retrieve":
                stats = upd.get("retrieval_stats") or {}
                if stats.get("enabled"):
                    yield emit({
                        "type": "retrieval",
                        "data": {
                            "attempted": stats.get("attempted", 0),
                            "retrieved": stats.get("retrieved", 0),
                            "full_text": stats.get("full_text", 0),
                            "partial": stats.get("partial", 0),
                            "metadata_only": stats.get("metadata_only", 0),
                            "failed": stats.get("failed", 0),
                            "words": stats.get("total_words", 0),
                            "chunks": stats.get("chunks", 0),
                            "references_followed": stats.get("references_followed", 0),
                            "sources": stats.get("sources", [])[:40],
                        },
                    })
                else:
                    yield emit({"type": "retrieval", "data": {
                        "enabled": False,
                        "note": stats.get("note", "full-text retrieval unavailable")}})

            elif node == "analyse":
                records = upd.get("extracted") or []
                yield emit({
                    "type": "analysis",
                    "data": {
                        "records": len(records),
                        "dimensions": sorted({r.dimension for r in records if r.dimension}),
                        "sources_summarised": len(upd.get("source_synth") or {}),
                        "notes": list(upd.get("notes") or [])[:8],
                    },
                })

            elif node == "gap_check":
                yield emit({"type": "gap", "data": upd["gap"]})

            elif node == "contradiction_check":
                yield emit({"type": "contradictions", "data": upd["contradictions"]})

            elif node == "synthesizer":
                export = upd.get("export") or {}
                if not export:
                    export = _export_meta(question)
                completion = upd.get("status") or {"status": "completed",
                                                   "missing_sections": [],
                                                   "headline": "COMPLETE REPORT"}
                # Emitted as its own event so a client can render the state
                # before it has the (potentially very large) report body.
                yield emit({"type": "status", "data": completion})
                # Section progress first: if the orchestrator model failed
                # mid-report and a fallback resumed from the output pool, the
                # stream still flows in document order instead of stalling
                # until the full body is ready -- or breaking entirely.
                for heading, markdown in upd.get("sections") or []:
                    try:
                        section_html = _section_html(markdown)
                    except Exception:
                        section_html = ""
                    yield emit({"type": "report_section",
                                      "heading": heading,
                                      "data": markdown,
                                      "html": section_html})
                report_event = emit({
                    "type": "report",
                    "data": upd["report"],
                    "html": upd.get("report_html") or "",
                    "sources": evidence,
                    "stats": upd.get("evidence_stats") or {},
                    "retrieval": stats,
                    "export": export,
                    "status": completion,
                })
                # Total wall clock for the run. Read after emit(), because
                # emit() is what marks the bundle complete and stamps the
                # duration; asking before that always returned None.
                report_event["elapsed_s"] = _elapsed_for(question)
                yield report_event


def _export_meta(question: str) -> dict:
    """Where the markdown export was written, so the UI can offer a download."""
    from .export import last_export
    return last_export(question)


def _elapsed_for(question: str):
    """Wall clock for this run, read back from the recorded bundle."""
    try:
        from .runlog import get
        bundle = get(question) or {}
        return bundle.get("elapsed_s")
    except Exception:
        return None


def format_duration(seconds) -> str:
    """Human duration for the report cover: 4m 12s, 1h 03m 09s."""
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return "unknown"
    if total < 0:
        return "unknown"
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"