"""The wall-clock budget is a real ceiling, not a decorative knob.

RESEARCH_TIME_BUDGET_S was defined in config.py and read by nothing, which
reads like a guarantee and behaves like nothing. These pin the enforcement:
the run stops between stages, says why, and keeps the work it already did.
"""
import asyncio
import time

from backend.research import config, graph as G
from backend.research.chunker import chunk_document


def _chunks(text: str = "Body text about the system. " * 40):
    return chunk_document(text, source_id="S1",
                          source_url="https://example.com/a",
                          source_title="A", retrieval_status="full")


def test_deadline_detects_an_overrun(monkeypatch):
    monkeypatch.setattr(config, "wall_clock_budget_s", lambda: 5.0)
    assert G._out_of_time({"started_at": time.monotonic() - 10})


def test_deadline_allows_a_run_inside_its_budget(monkeypatch):
    monkeypatch.setattr(config, "wall_clock_budget_s", lambda: 600.0)
    assert not G._out_of_time({"started_at": time.monotonic()})


def test_deadline_is_inert_without_a_start_time():
    """A missing timestamp must not stop everything."""
    assert not G._out_of_time({})


def test_zero_disables_the_ceiling(monkeypatch):
    monkeypatch.setattr(config, "wall_clock_budget_s", lambda: 0.0)
    assert not G._out_of_time({"started_at": time.monotonic() - 10_000})


def test_analyse_stops_at_the_deadline_and_explains(monkeypatch):
    monkeypatch.setattr(config, "wall_clock_budget_s", lambda: 5.0)
    state = {
        "question": "how does it scale", "chunks": _chunks(), "extracted": [],
        "extracted_sources": {}, "plan": {}, "budget": None, "max_rounds": 2,
        "started_at": time.monotonic() - 999,
    }
    out = asyncio.run(G.analyse(state))
    assert any("wall-clock budget" in n for n in out["notes"])
    # Evidence from earlier rounds is merged forward, never discarded.
    assert "extracted" in out and "source_synth" in out


def test_retrieve_stops_at_the_deadline_but_keeps_prior_chunks(monkeypatch):
    """Stopping must not make the report thinner for no reason.

    Returning an empty chunk list here would discard documents earlier rounds
    already paid to fetch.
    """
    monkeypatch.setattr(config, "wall_clock_budget_s", lambda: 5.0)
    made = _chunks()
    state = {
        "question": "q", "plan": {}, "started_at": time.monotonic() - 999,
        "documents_by_url": {"https://example.com/a": object()},
        "chunks_by_url": {"https://example.com/a": ("full", made)},
        "retrieval_index": {}, "ai_leads": [], "raw": [],
    }
    out = asyncio.run(G.retrieve(state))
    assert len(out["chunks"]) == len(made)
    assert any("wall-clock budget" in n for n in out["notes"])
    assert out["retrieval_stats"]["enabled"] is False


def test_planner_stamps_the_start_time():
    """Without started_at there is no deadline to enforce."""
    from backend.research import planner as planner_module

    planner_module.make_plan = lambda q: {"subquestions": [
        {"question": "s", "searches": [{"source": "web", "query": "q"}]}]}
    out = G.planner({"question": "q", "max_rounds": 1})
    assert isinstance(out.get("started_at"), float)