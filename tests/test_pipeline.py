import asyncio
import datetime
import os

import pytest
from dotenv import load_dotenv

import backend.research.collector as C
import backend.research.graph as G
import backend.research.planner as P
from backend.research.collector import Collector, year_of
from backend.research.report import ReportResult
from backend.research.researcher import deep_research

load_dotenv()


def test_year_of():
    assert year_of("Mar 5, 2025") == 2025
    assert year_of(None) is None
    assert year_of("3 days ago") == datetime.date.today().year


def test_collector_dedupes_and_flags_stale():
    col = Collector()
    new = col.add([
        {"title": "A", "url": "https://www.x.com/a/", "snippet": "", "date": "2019", "type": "web", "query": "q"},
        {"title": "A dup", "url": "https://x.com/a", "snippet": "", "date": "2025", "type": "web", "query": "q"},
    ])
    assert len(new) == 1
    assert new[0]["stale"] is True


def test_planner_drops_unknown_sources(monkeypatch):
    monkeypatch.setattr(P, "ask", lambda *a, **k: {"subquestions": [
        {"question": "q", "searches": [
            {"source": "web", "query": "a"},
            {"source": "tiktok", "query": "b"},
        ]}]})
    plan = P.make_plan("anything")
    assert plan["subquestions"][0]["searches"] == [{"source": "web", "query": "a"}]


def test_graph_runs_end_to_end_with_mocks(monkeypatch):
    monkeypatch.setattr(G, "make_plan", lambda q: {"subquestions": [{"question": "a", "searches": [
        {"source": "web", "query": "rust"}, {"source": "news", "query": "cpp"}]}]})

    # Retrieval is a network stage; this test is about graph topology only.
    monkeypatch.setattr(G.config, "fetch_enabled", lambda: False)

    def fake_search(source, query, n=6):
        return ([{"title": f"{source}{query}", "url": f"https://x.com/{source}/{query}",
                 "snippet": "s", "date": "2025", "type": source, "query": query}], {})
    monkeypatch.setattr(G, "search_full", fake_search)

    calls = {"n": 0}

    def fake_ask(system, user, json_mode=False, role="default"):
        calls["n"] += 1
        return {"sufficient": calls["n"] > 1, "missing": "x",
                "follow_ups": [{"source": "scholar", "query": "bench"}]}
    monkeypatch.setattr(G, "ask", fake_ask)
    monkeypatch.setattr(C, "ask", lambda *a, **k: {"contradictions": []})
    # The graph asks for a ReportResult now, so it can report how complete the
    # run was rather than only returning markdown.
    def fake_generate(question, collector, contradictions, *args, **kwargs):
        return ReportResult(f"REPORT {len(collector.list())}", "completed")
    monkeypatch.setattr(G, "generate_report", fake_generate)

    async def run():
        return [ev async for ev in deep_research("q", 3)]

    events = asyncio.run(run())
    types = [e["type"] for e in events]
    # The service/credit banner is deliberately the first thing a run emits:
    # you should see what the run depends on before it spends anything.
    assert "services" in types
    assert types.index("services") < types.index("plan")
    assert types[-1] == "report"
    assert "gap" in types
    assert events[-1]["data"] == "REPORT 3"  # 2 first-round + 1 follow-up
    assert events[-1]["status"]["status"] == "completed"


def test_gap_check_terminates_when_no_follow_ups(monkeypatch):
    """An empty follow-up list must end the loop, not send it back to retrieve."""
    monkeypatch.setattr(G, "make_plan", lambda q: {"subquestions": [{"question": "a", "searches": [
        {"source": "web", "query": "rust"}]}]})
    monkeypatch.setattr(G.config, "fetch_enabled", lambda: False)
    monkeypatch.setattr(G, "search_full", lambda source, query, n=6: ([], {}))
    monkeypatch.setattr(G, "ask", lambda *a, **k: {"sufficient": False, "missing": "x",
                                                   "follow_ups": []})
    monkeypatch.setattr(G, "generate_report",
                        lambda q, col, c, *a, **k: ReportResult("REPORT", "completed"))

    async def run():
        return [ev async for ev in deep_research("q", 2)]

    events = asyncio.run(run())
    types = [e["type"] for e in events]
    assert types.count("retrieval") == 1, "retrieval must not repeat without new searches"
    assert "contradictions" in types
    assert types[-1] == "report"


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("SERPAPI_KEY"), reason="no SERPAPI_KEY")
def test_serpapi_live():
    """Uses 1 SerpApi credit. Confirms your key and the engine setup work."""
    from backend.research.searcher import search
    results = search("web", "rust game engine", n=3)
    assert len(results) > 0
    assert results[0]["url"]