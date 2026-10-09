"""End-to-end tests of the real graph running entirely on mock mode.

No real model or SerpApi call happens in this file.

Note on layers: deep_research() yields only graph events. The terminal "done"
marker and the conversion of a failure into an "error" event both live in
backend/main.py, so any assertion about "done" or "error" goes through the
HTTP endpoint rather than through deep_research() directly.
"""

import asyncio
import re

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.research.researcher import deep_research


@pytest.fixture(autouse=True)
def clean_mocks(monkeypatch):
    """Default every mock switch off so nothing leaks between tests."""
    for name in (
        "LLM_MOCK",
        "LLM_MOCK_FAIL",
        "LLM_MOCK_DELAY_MS",
        "SEARCH_MOCK",
        "SEARCH_MOCK_FAIL",
        "SEARCH_MOCK_DELAY_MS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_MOCK", "1")
    monkeypatch.setenv("SEARCH_MOCK", "1")
    monkeypatch.setenv("LLM_MOCK_DELAY_MS", "0")
    monkeypatch.setenv("SEARCH_MOCK_DELAY_MS", "0")


def run_graph(question, rounds):
    """Graph-level events only (no terminal marker)."""

    async def drive():
        return [event async for event in deep_research(question, rounds)]

    return asyncio.run(drive())


def run_sse(question, rounds=1):
    """Full SSE stream through the endpoint, including the terminal marker."""
    client = TestClient(main.app)
    response = client.get(f"/api/research?q={question}&rounds={rounds}")
    events = []
    for frame in response.text.split("\n\n"):
        if not frame.strip():
            continue
        line = next(ln for ln in frame.split("\n") if ln.startswith("data: "))
        events.append(__import__("json").loads(line[6:]))
    return events


def types_of(events):
    return [event["type"] for event in events]


def cited_numbers(text):
    """Every evidence number the report cites, in either citation style.

    The real model writes combined groups like "[14, 16, 18, 19]"; the mock also
    keeps the legacy separate style "[1], [2]" in one place so both are covered.
    """
    numbers = []
    for group in re.findall(r"\[([\d,\s]+)\]", text):
        numbers.extend(int(part) for part in group.split(",") if part.strip())
    return numbers


def test_rounds_one_produces_the_full_sequence():
    order = types_of(run_sse("What is FastAPI?", 1))

    assert order.index("services") < order.index("plan")
    for expected in ("results", "evidence", "gap", "contradictions", "report"):
        assert expected in order, f"missing {expected} in {order}"
    assert order[-1] == "done"
    assert order[-2] == "report"
    assert order.count("done") == 1


def test_report_has_short_answer_and_a_valid_citation():
    events = run_graph("What is FastAPI?", 1)

    report = next(e["data"] for e in events if e["type"] == "report")
    evidence_count = next(
        e["count"] for e in events if e["type"] == "evidence"
    )
    sources = next(e["sources"] for e in events if e["type"] == "report")

    assert "## Short answer" in report
    assert sources, "report should carry its sources"

    cited = cited_numbers(report)
    assert cited, "report should cite evidence"
    assert all(1 <= n <= evidence_count for n in cited), (
        f"citation outside evidence range (count={evidence_count}): {cited}"
    )


def test_report_exercises_both_citation_styles():
    events = run_graph("What is FastAPI?", 1)
    report = next(e["data"] for e in events if e["type"] == "report")

    assert re.search(r"\[\d+,\s*\d+", report), (
        "expected at least one combined citation like [1, 2, 3]"
    )
    assert re.search(r"\[\d+\](?:,|\.)\s*\[\d+\]", report), (
        "expected at least one separate citation like [1], [2]"
    )


def test_collector_dedupes_skips_and_keeps_junk_and_stale():
    """Assert the collector's real behaviour, not an imagined one."""
    events = run_graph("What is FastAPI?", 1)

    results = [e for e in events if e["type"] == "results"]
    evidence = next(e["count"] for e in events if e["type"] == "evidence")

    rows_returned = sum(e["count"] for e in results)
    # one empty-URL row and one duplicate per search are dropped
    assert evidence == rows_returned - (2 * len(results)), (
        f"expected one skip and one merge per search; "
        f"rows={rows_returned} evidence={evidence} searches={len(results)}"
    )

    sources = next(e["sources"] for e in events if e["type"] == "report")
    urls = [r["url"] for r in sources]

    assert all(u for u in urls), "no evidence should carry an empty URL"
    assert len({r["id"] for r in sources}) == len(sources)
    # there is no blocked-domain list today, so junk survives by design
    assert any("social-chatter.example" in u for u in urls), (
        "junk domain is expected to survive: no blocked list exists"
    )
    # the duplicate is merged, so its www+slash variant is absent
    assert not any(u.endswith("/") for u in urls)


def test_stale_evidence_is_flagged():
    events = run_graph("What is FastAPI?", 1)
    sources = next(e["sources"] for e in events if e["type"] == "report")

    stale = [r for r in sources if r.get("stale")]
    fresh = [r for r in sources if not r.get("stale")]

    assert stale, "the archived 2015 mock result should be flagged stale"
    assert fresh, "the recent mock results should not be flagged"


def test_rounds_two_stops_at_the_round_limit():
    """At rounds=2 the final gap is the graph's own limit branch, not the mock's."""
    events = run_graph("What is FastAPI?", 2)
    gaps = [e["data"] for e in events if e["type"] == "gap"]

    assert gaps[0]["sufficient"] is False
    assert gaps[0]["follow_ups"], "first gap should ask for follow-ups"
    assert gaps[-1]["missing"] == "round limit reached"


def test_rounds_three_runs_the_follow_up_branch_and_finishes_sufficient():
    order = types_of(run_sse("What is FastAPI?", 3))

    first_gap = order.index("gap")
    second_gap = order.index("gap", first_gap + 1)
    extra_results = order[first_gap + 1:second_gap].count("results")
    assert extra_results >= 1, f"follow-up branch did not run: {order}"

    events = run_graph("What is FastAPI?", 3)
    gaps = [e["data"] for e in events if e["type"] == "gap"]

    assert gaps[0]["sufficient"] is False
    assert gaps[-1]["sufficient"] is True, f"final gap not sufficient: {gaps[-1]}"

    assert order[-1] == "done"
    assert order[-2] == "report"


def test_llm_mock_fail_planner_falls_back_and_completes(monkeypatch):
    """A dead planner degrades to a generic plan instead of killing the run."""
    monkeypatch.setenv("LLM_MOCK_FAIL", "planner")

    events = run_sse("What is FastAPI?", 1)
    order = types_of(events)

    assert order.index("services") < order.index("plan")
    assert order[-1] == "done"
    assert "report" in order, f"the fallback plan must still produce a report: {order}"
    assert "error" not in order


def test_llm_mock_fail_synth_keeps_results_then_errors(monkeypatch):
    monkeypatch.setenv("LLM_MOCK_FAIL", "synth")

    events = run_sse("What is FastAPI?", 1)
    order = types_of(events)

    assert "results" in order
    assert "evidence" in order
    assert order[-2:] == ["error", "done"]
    assert "report" not in order
    assert "synth" in events[-2]["message"]


def test_search_mock_fail_marks_that_source_and_the_run_continues(monkeypatch):
    monkeypatch.setenv("SEARCH_MOCK_FAIL", "web")

    events = run_sse("What is FastAPI?", 1)
    order = types_of(events)

    results = [e for e in events if e["type"] == "results"]
    failed = [e for e in results if e["error"]]
    ok = [e for e in results if not e["error"]]

    assert failed, "the failing source should report an error"
    assert all(e["source"] == "web" for e in failed)
    assert ok, "other sources should still succeed"
    assert order[-2:] == ["report", "done"]


def test_mock_mode_bypasses_the_search_cache(monkeypatch, tmp_path):
    """A cache directory must not shadow mock results."""
    monkeypatch.setenv("SEARCH_CACHE_DIR", str(tmp_path))

    from backend.research import searcher

    results = searcher.search("web", "anything", n=6)

    assert results
    assert not list(tmp_path.iterdir()), "mock mode should write nothing to the cache"


def test_no_mock_env_means_the_real_llm_path_runs(monkeypatch):
    """Guard: with the switches off, ask() must reach litellm."""
    import litellm

    from backend.research import llm

    monkeypatch.delenv("LLM_MOCK", raising=False)
    monkeypatch.setenv("LLM_MODEL", "gemini/gemini-3.8-flash")
    monkeypatch.setenv("LLM_FALLBACKS", "")

    seen = []

    class Msg:
        content = "from the real path"

    class Resp:
        choices = [type("C", (), {"message": Msg()})()]

    def fake_completion(**kwargs):
        seen.append(kwargs["model"])
        return Resp()

    monkeypatch.setattr(litellm, "completion", fake_completion)

    assert llm.ask("sys", "user") == "from the real path"
    assert seen == ["gemini/gemini-3.8-flash"]


def test_no_mock_env_means_the_real_search_path_runs(monkeypatch):
    """Guard: with SEARCH_MOCK off, search() must call SerpApi."""
    monkeypatch.delenv("SEARCH_MOCK", raising=False)
    monkeypatch.delenv("SEARCH_CACHE_DIR", raising=False)
    monkeypatch.setenv("SERPAPI_KEY", "test-key")

    import serpapi

    calls = []

    class FakeSearch:
        def __init__(self, params):
            calls.append(params)

        def get_dict(self):
            return {"organic_results": [
                {"title": "T", "link": "https://example.com/x", "snippet": "S"}
            ]}

    monkeypatch.setattr(serpapi, "GoogleSearch", FakeSearch)

    from backend.research import searcher

    results = searcher.search("web", "anything", n=3)

    assert len(calls) == 1, "the real path should have called SerpApi"
    assert results[0]["url"] == "https://example.com/x"
    assert results[0]["type"] == "web"