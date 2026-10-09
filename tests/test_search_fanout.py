"""Tests for the LangGraph search fan-out and its ordering guarantee.

Two separate concerns, and the second is the one that matters:

  * every general query reaches every wired open-web engine, concurrently;
  * the results are reassembled in SUBMISSION order, not completion order.

The second is not a nicety. LangGraph merges a superstep's writes in the order
tasks finish, the collector numbers sources from insertion order, and those
numbers are what the report's citations point at. If completion order leaked
through, the same research run would number its sources differently every time
and the PDF's citation numbers would stop agreeing with the source list.
"""
import asyncio

import pytest

from backend.research import config, providers
from backend.research import graph as graph_module


# --- fan-out shape ----------------------------------------------------------

def _wired(monkeypatch, *sources):
    """Wire the given general sources; everything else is left unwired."""
    for provider in providers.all_providers():
        monkeypatch.delenv(provider.env, raising=False)
    monkeypatch.setenv("SERPAPI_KEY", "k")
    for source in sources:
        monkeypatch.setenv(f"{source.upper()}_API_KEY", "k")


def test_a_general_query_is_expanded_to_every_wired_engine(monkeypatch):
    _wired(monkeypatch, "exa", "jina", "tinyfish")
    tasks = graph_module._fan_out([{"source": "web", "query": "q"}], 1)
    assert [t["source"] for t in tasks] == ["web", "exa", "jina", "tinyfish"]
    assert all(t["query"] == "q" for t in tasks)


def test_an_unwired_engine_is_not_expanded_into(monkeypatch):
    _wired(monkeypatch, "exa")
    tasks = graph_module._fan_out([{"source": "web", "query": "q"}], 1)
    assert [t["source"] for t in tasks] == ["web", "exa"]


def test_corpus_specific_sources_are_never_expanded(monkeypatch):
    """A patent query asked of a news index does not add evidence; it throws
    away the constraint that made it a patent query."""
    _wired(monkeypatch, "exa", "jina", "tinyfish")
    for source in ("scholar", "patent", "github", "arxiv", "news"):
        tasks = graph_module._fan_out([{"source": source, "query": "q"}], 1)
        assert len(tasks) == 1, f"{source} was fanned out"
        assert tasks[0]["source"] == source


def test_several_queries_expand_independently(monkeypatch):
    _wired(monkeypatch, "exa")
    tasks = graph_module._fan_out(
        [{"source": "web", "query": "a"},
         {"source": "scholar", "query": "b"},
         {"source": "web", "query": "c"}], 1)
    assert [t["source"] for t in tasks] == ["web", "exa", "scholar", "web", "exa"]
    assert [t["query"] for t in tasks] == ["a", "a", "b", "c", "c"]


def test_fanout_can_be_switched_off(monkeypatch):
    """One engine per query is a third of the search spend and a thinner
    report; it stays available as a setting, not as the default."""
    _wired(monkeypatch, "exa", "jina")
    monkeypatch.setattr(config, "search_fanout", lambda: False)
    tasks = graph_module._fan_out([{"source": "web", "query": "q"}], 1)
    assert [t["source"] for t in tasks] == ["web"]


def test_fanout_ceiling_is_honoured(monkeypatch):
    _wired(monkeypatch, "exa", "jina", "tinyfish")
    monkeypatch.setattr(config, "search_fanout", lambda: True)
    monkeypatch.setattr(config, "search_fanout_max_engines", lambda: 2)
    tasks = graph_module._fan_out([{"source": "web", "query": "q"}], 1)
    assert len(tasks) == 2


def test_serpapi_leads_the_expansion(monkeypatch):
    """Its rows carry the most metadata, and when two engines return the same
    URL the collector keeps whichever arrived first."""
    _wired(monkeypatch, "exa", "jina")
    assert graph_module._fan_out([{"source": "web", "query": "q"}], 1)[0]["source"] \
        == "web"


def test_an_empty_plan_expands_to_nothing(monkeypatch):
    _wired(monkeypatch, "exa")
    assert graph_module._fan_out([], 1) == []


# --- the routers actually route the expansion -------------------------------

def test_route_sends_one_task_per_engine(monkeypatch):
    _wired(monkeypatch, "exa", "jina")
    sends = graph_module.route({"pending": [{"source": "web", "query": "q"}],
                                "round": 1})
    assert len(sends) == 3
    assert {s.arg["source"] for s in sends} == {"web", "exa", "jina"}


def test_route_after_gap_also_fans_out(monkeypatch):
    """The gap router is a separate node; leaving it unexpanded would mean
    round 2 quietly searches fewer engines than round 1, which is exactly the
    kind of thing that looks like 'nothing new was found'."""
    _wired(monkeypatch, "exa", "jina")
    sends = graph_module.route_after_gap(
        {"pending": [{"source": "web", "query": "q"}], "round": 2})
    assert len(sends) == 3
    assert all(s.arg["seq"].startswith("2:") for s in sends)


def test_an_empty_pending_list_still_terminates(monkeypatch):
    """Regression guard: these two routers used to be one, and a shared empty
    pending sent the graph back to retrieve forever."""
    _wired(monkeypatch, "exa")
    assert graph_module.route({"pending": [], "round": 1}) == "retrieve"
    assert graph_module.route_after_gap({"pending": [], "round": 1}) == \
        "contradiction_check"


# --- ordering ---------------------------------------------------------------

def test_seq_is_unique_and_sortable(monkeypatch):
    _wired(monkeypatch, "exa", "jina")
    pending = [{"source": "web", "query": "a"}, {"source": "web", "query": "b"}]
    tasks = graph_module._fan_out(pending, 1)
    seqs = [t["seq"] for t in tasks]
    assert len(seqs) == len(set(seqs))
    assert seqs == sorted(seqs)


def test_rounds_do_not_collide_in_seq(monkeypatch):
    """Round 2 must not reuse round 1's keys, or the reducer would silently
    overwrite round 1's results."""
    _wired(monkeypatch, "exa")
    one = graph_module._fan_out([{"source": "web", "query": "q"}], 1)
    two = graph_module._fan_out([{"source": "web", "query": "q"}], 2)
    assert {t["seq"] for t in one} & {t["seq"] for t in two} == set()


def test_results_are_read_in_submission_order_not_completion_order():
    """The core guarantee. `raw` is written by a reducer that concatenates in
    arrival order, so it must never be read directly."""
    state = {
        "raw": [{"title": "arrived last", "url": "https://c.example", "snippet": "",
                 "date": "", "type": "web", "query": "q"}],
        "raw_by_task": {
            "1:000:00": [{"title": "first", "url": "https://a.example",
                          "snippet": "", "date": "", "type": "web", "query": "q"}],
            "1:001:00": [{"title": "second", "url": "https://b.example",
                          "snippet": "", "date": "", "type": "web", "query": "q"}],
        },
    }
    assert [r["title"] for r in graph_module._ordered_raw(state)] == \
        ["first", "second"]


def test_a_late_arriving_task_does_not_move_its_neighbours():
    """Out-of-order completion must not renumber anything already placed."""
    rows = {"1:000:00": [{"title": "a", "url": "https://a.example", "snippet": "",
                          "date": "", "type": "web", "query": "q"}],
            "1:000:01": [{"title": "b", "url": "https://b.example", "snippet": "",
                          "date": "", "type": "web", "query": "q"}],
            "1:000:02": [{"title": "c", "url": "https://c.example", "snippet": "",
                          "date": "", "type": "web", "query": "q"}]}

    def titles(mapping):
        return [r["title"] for r in graph_module._ordered_raw(
            {"raw_by_task": mapping})]

    before = titles(rows)
    # b arrives after c, as it would with a slower provider.
    partial = {k: v for k, v in rows.items() if k != "1:000:01"}
    later = titles({**partial, "1:000:01": rows["1:000:01"]})
    assert before == ["a", "b", "c"]
    assert later[0] == "a", "an earlier task must not be displaced"
    assert later[-1] == "c", "a later task must not be displaced"


def test_source_ids_follow_submission_order(monkeypatch):
    """The end the guarantee exists for: the numbers the report cites."""
    monkeypatch.setenv("SEARCH_CACHE_DIR", "")
    state = {
        "raw_by_task": {
            "1:000:00": [{"title": "google", "url": "https://a.example",
                          "snippet": "s", "date": "", "type": "web", "query": "q"}],
            "1:000:01": [{"title": "exa", "url": "https://b.example",
                          "snippet": "s", "date": "", "type": "web", "query": "q"}],
        },
    }
    evidence = graph_module.collect(state)["evidence"]
    assert [s["title"] for s in evidence] == ["google", "exa"]
    assert [s["id"] for s in evidence] == [1, 2]


def test_duplicate_urls_across_engines_collapse_to_one_source(monkeypatch):
    """Four engines answering the same query overlap heavily; the collector must
    union them rather than listing the same page four times."""
    monkeypatch.setenv("SEARCH_CACHE_DIR", "")
    duplicate = "https://example.com/paper"
    state = {
        "raw_by_task": {
            f"1:000:0{i}": [{"title": f"engine{i}", "url": duplicate,
                             "snippet": "s", "date": "", "type": "web",
                             "query": "q"}]
            for i in range(4)
        },
    }
    assert len(graph_module.collect(state)["evidence"]) == 1


def test_fanout_widens_the_source_pool(monkeypatch):
    """The stated reason for the fan-out: the union beats any single engine."""
    monkeypatch.setenv("SEARCH_CACHE_DIR", "")
    engines = {
        "1:000:00": ("https://shared.example", "https://only-google.example"),
        "1:000:01": ("https://shared.example", "https://only-exa.example"),
        "1:000:02": ("https://shared.example", "https://only-jina.example"),
    }
    state = {"raw_by_task": {
        seq: [{"title": url, "url": url, "snippet": "s", "date": "",
               "type": "web", "query": "q"} for url in urls]
        for seq, urls in engines.items()}}
    evidence = graph_module.collect(state)["evidence"]
    assert {s["url"] for s in evidence} == {
        "https://shared.example", "https://only-google.example",
        "https://only-exa.example", "https://only-jina.example"}


def test_a_failing_engine_does_not_remove_the_others(monkeypatch):
    """search_worker catches per-task; one provider being down is a thinner
    round, not a failed one."""
    state = {"raw_by_task": {"1:000:00": [], "1:000:01": [
        {"title": "b", "url": "https://b.example", "snippet": "s", "date": "",
         "type": "exa", "query": "q"}]}}
    evidence = graph_module.collect(state)["evidence"]
    assert [s["title"] for s in evidence] == ["b"]
    assert evidence[0]["id"] == 1, "the empty engine must not consume id 1"


# --- search_worker ----------------------------------------------------------

def test_search_worker_records_under_its_seq(monkeypatch):
    monkeypatch.setattr(graph_module, "search_full",
                        lambda source, query: (
                            [{"title": source, "url": f"https://{source}.example",
                              "snippet": "", "date": "", "type": source,
                              "query": query}], {}))
    update = asyncio.run(graph_module.search_worker(
        {"source": "exa", "query": "q", "seq": "1:000:03"}))
    assert update["raw_by_task"] == {"1:000:03": update["raw_by_task"]["1:000:03"]}
    assert update["raw_by_task"]["1:000:03"][0]["url"] == "https://exa.example"


def test_search_worker_survives_a_failing_engine(monkeypatch):
    def boom(source, query):
        raise RuntimeError("provider down")

    monkeypatch.setattr(graph_module, "search_full", boom)
    update = asyncio.run(graph_module.search_worker(
        {"source": "exa", "query": "q", "seq": "1:000:00"}))
    assert update["raw_by_task"] == {"1:000:00": []}
    assert update["log"][0]["error"]


def test_seq_is_not_logged_as_a_search_field(monkeypatch):
    """`seq` is routing bookkeeping; it is not a source and must not appear in
    the run log where it would read as one."""
    monkeypatch.setattr(graph_module, "search_full",
                        lambda source, query: ([], {}))
    update = asyncio.run(graph_module.search_worker(
        {"source": "web", "query": "q", "seq": "1:000:00"}))
    assert "seq" not in update["log"][0]


def test_the_dict_reducer_merges_rather_than_replaces():
    """Two workers writing in the same superstep must both survive."""
    merged = graph_module._merge_dicts({"a": [1]}, {"b": [2]})
    assert merged == {"a": [1], "b": [2]}
    assert graph_module._merge_dicts(None, {"b": [2]}) == {"b": [2]}


def test_state_declares_raw_by_task_with_a_merging_reducer():
    """An Annotated field without a reducer would have LangGraph replace the
    dict on every write, so only the last engine's results would survive."""
    hints = graph_module.State.__annotations__
    assert "raw_by_task" in hints
    reducer = getattr(hints["raw_by_task"], "__metadata__", [None])[0]
    assert reducer is graph_module._merge_dicts, \
        "raw_by_task needs the merging reducer; list concatenation would " \
        "concatenate a dict's keys instead of merging it"