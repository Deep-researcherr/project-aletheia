"""Per-run event recorder for the PDF export.

`researcher.deep_research` streams events to the browser; the "Save PDF"
button needs the same data afterwards. This module accumulates each run's
events into one bundle keyed by question, so `/api/report/pdf` can rebuild
the full report (overview, sections, sources, references) without rerunning
anything. Bounded to the most recent runs; never raises.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

_KEEP_RUNS = 8

_lock = threading.Lock()
_runs: "OrderedDict[str, dict]" = OrderedDict()


def _key(question: str) -> str:
    from .export import slugify
    return slugify(question or "")


def begin(question: str, max_rounds: int) -> None:
    """Start (or restart) the bundle for a question."""
    with _lock:
        _runs[_key(question)] = {
            "question": question,
            "max_rounds": max_rounds,
            "plan": {},
            "searches": [],
            "evidence_count": 0,
            "retrieval": {},
            "analysis": {},
            "gaps": [],
            "contradictions": [],
            "progress": {},
            "sections": [],
            "report": "",
            "report_html": "",
            "sources": [],
            "stats": {},
            "export": {},
            "status": {},
            "complete": False,
        }
        _runs.move_to_end(_key(question))
        while len(_runs) > _KEEP_RUNS:
            _runs.popitem(last=False)


def record(question: str, event: dict) -> None:
    """Fold one streamed event into the question's bundle."""
    try:
        key = _key(question)
        with _lock:
            bundle = _runs.get(key)
            if bundle is None:
                return
            kind = event.get("type")
            if kind == "progress":
                # Mid-stage progress (per-document fetch, per-batch extraction).
                # Kept as a running tally so the PDF can say what the run was
                # doing, without storing every URL it ever touched.
                stage = str(event.get("stage") or "")
                if stage:
                    seen = bundle["progress"].setdefault(
                        stage, {"done": 0, "total": 0, "note": ""})
                    try:
                        seen["done"] = int(event.get("done") or 0)
                        seen["total"] = int(event.get("total") or 0)
                    except (TypeError, ValueError):
                        pass
                    note = str(event.get("note") or "")
                    if note:
                        seen["note"] = note[:160]
            elif kind == "plan":
                bundle["plan"] = event.get("data") or {}
            elif kind == "results":
                bundle["searches"].append({
                    "query": event.get("query", ""),
                    "source": event.get("source", ""),
                    "count": event.get("count", 0),
                    "error": event.get("error"),
                })
            elif kind == "evidence":
                bundle["evidence_count"] = event.get("count", 0)
            elif kind == "retrieval":
                bundle["retrieval"] = event.get("data") or {}
            elif kind == "analysis":
                bundle["analysis"] = event.get("data") or {}
            elif kind == "gap":
                bundle["gaps"].append(event.get("data") or {})
            elif kind == "contradictions":
                bundle["contradictions"] = event.get("data") or []
            elif kind == "report_section":
                bundle["sections"].append(event.get("heading", ""))
            elif kind == "status":
                bundle["status"] = event.get("data") or {}
            elif kind == "report":
                bundle["report"] = event.get("data") or ""
                bundle["report_html"] = event.get("html") or ""
                bundle["sources"] = event.get("sources") or []
                bundle["stats"] = event.get("stats") or {}
                bundle["export"] = event.get("export") or {}
                bundle["status"] = event.get("status") or bundle["status"]
                bundle["complete"] = True
            _runs.move_to_end(key)
    except Exception:
        pass


def get(question: str) -> dict | None:
    """The completed bundle for a question, or None."""
    with _lock:
        bundle = _runs.get(_key(question))
        if not bundle or not bundle.get("complete") or not bundle.get("report"):
            return None
        return dict(bundle)


def reset() -> None:
    """Forget every recorded run. Intended for tests."""
    with _lock:
        _runs.clear()
