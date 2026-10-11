"""A degraded run must never be reported as a complete one.

Caught live: a run where 3 of 7 extraction batches returned nothing at all was
reported as `completed` with headline `COMPLETE REPORT` and no missing
sections. Every section was written and every citation resolved, so nothing
downstream objected -- the loss was invisible.

Three things had to change together, and the bug needed all three:

  1. a batch that yields no records and discards nothing used to produce no
     note at all, so there was nothing downstream to notice;
  2. notes were written into an HTML comment, invisible in the rendered report;
  3. completion status only considered failed sections and evidence holes, so
     an intact-looking report could be built on a gutted evidence base.
"""
import os
import re

from backend.research import extract as E
from backend.research.chunker import chunk_document
from backend.research.evidence import Evidence
from backend.research.relevance import Scored
from backend.research.report import generate_report, partial_banner


def _scored(count: int, size: int = 20):
    out = []
    for i in range(1, count + 1):
        for c in chunk_document(
                f"Passage about edge computing systems {i}. " * size,
                source_id=f"S{i}", source_url=f"https://e.com/{i}",
                source_title=f"D{i}"):
            out.append(Scored(chunk=c, score=1.0, reasons=[]))
    return out


def _answering_ask(good_sources: int):
    """A provider that answers for the first N sources, then returns junk.

    The failure shape is a reply with no readable evidence in it -- not an
    empty one. An empty reply means the model looked at the passages and found
    nothing extractable, which is correct behaviour and must not be counted as
    a lost batch; see ``test_declining_is_not_degradation``.
    """
    def ask(system, user, **kwargs):
        sources = [int(m) for m in re.findall(r"id=S(\d+)#", user)]
        if not sources or min(sources) <= good_sources:
            return {"evidence": [
                {"chunk": f"P{i + 1}",
                 "claim": f"Measured claim {i} about edge throughput",
                 "detail": "", "quote": "", "dimension": "", "targets": [],
                 "quality": "moderate", "limitations": "", "confidence": 0.6}
                for i in range(len(sources))]}
        return {"unexpected": "shape"}
    return ask


# --- 1. the signal exists at all -------------------------------------------

def test_an_unreadable_batch_is_reported(monkeypatch):
    """No records, nothing discarded, and a reply we cannot read: a failure."""
    monkeypatch.setattr(E, "ask", lambda *a, **k: {"unexpected": "shape"})
    monkeypatch.setenv("EXTRACT_BATCH_CHARS", "700")

    _records, notes = E.extract_evidence(
        _scored(3), "edge versus cloud tradeoffs", None, None)

    empties = [n for n in notes if "unreadable reply" in n]
    assert empties, "a batch whose reply could not be read left no trace"


def test_declining_is_not_degradation(monkeypatch):
    """The model finding nothing in a passage is correct, not a lost batch.

    Verified live: asked about a reference list, acknowledgements and a figure
    caption, the model replied with a bare ``[]``. The old code read any
    non-dict as a dead reply, so that healthy outcome was logged as "the
    model's reply was unusable" 30 times on one run and downgraded the report
    to PARTIAL.
    """
    monkeypatch.setattr(E, "ask", lambda *a, **k: [])
    monkeypatch.setenv("EXTRACT_BATCH_CHARS", "700")

    records, notes = E.extract_evidence(
        _scored(3), "edge versus cloud tradeoffs", None, None)

    assert E.integrity_notes(notes) == [], (
        "declining to invent claims must not degrade the run")
    assert records == []
    assert any("no extractable claims" in n for n in notes), \
        "but it should still be visible as an observation"


def test_empty_evidence_object_is_also_a_decline(monkeypatch):
    """Both empty shapes mean the same thing."""
    monkeypatch.setattr(E, "ask", lambda *a, **k: {"evidence": []})
    monkeypatch.setenv("EXTRACT_BATCH_CHARS", "700")

    records, notes = E.extract_evidence(
        _scored(3), "edge versus cloud tradeoffs", None, None)

    assert records == []
    assert E.integrity_notes(notes) == []


def test_degradation_is_weighed_by_passages_not_batches(monkeypatch):
    """A batch-count threshold over-reports, because failures skew small.

    Batches are capped at 12 passages, so losing many tiny batches costs far
    less evidence than losing a few full ones. The threshold must count
    passages. Batch shapes are pinned here so the ratio is deterministic.
    """
    import backend.research.extract as _E

    one = _scored(1)[0]                       # a single Scored passage
    twelve = [_scored(1)[0]] * 12

    def scored(reps):
        return [one] * reps

    # Full 12-passage batches answer; anything smaller returns junk.
    def ask(system, user, **kwargs):
        n = len(re.findall(r"id=S(\d+)#", user))
        if n == 12:
            return {"evidence": [
                {"chunk": f"P{i + 1}",
                 "claim": f"Measured claim {i} about edge throughput",
                 "detail": "", "quote": "", "dimension": "", "targets": [],
                 "quality": "moderate", "limitations": "", "confidence": 0.6}
                for i in range(n)]}
        return {"unexpected": "shape"}

    monkeypatch.setattr(_E, "ask", ask)
    monkeypatch.setenv("EXTRACT_MAX_BATCHES", "40")

    # 3 of 5 batches dead, but each dead batch holds ONE passage out of 27.
    monkeypatch.setattr(_E, "plan_batches",
                        lambda scored_in: [twelve, twelve, scored(1),
                                           scored(1), scored(1)])
    records, notes = _E.extract_evidence(
        scored(27), "edge versus cloud tradeoffs", None, None)

    assert records, "the full batches should have produced evidence"
    assert E.integrity_notes(notes) == [], (
        "losing 3 of 27 passages is not a degraded evidence base even though "
        "3 of 5 batches died")

    # The converse: the same number of dead batches, but now they hold five
    # passages each -- 15 of 27 lost, over half the evidence base.
    monkeypatch.setattr(_E, "plan_batches",
                        lambda scored_in: [twelve, scored(5), scored(5),
                                           scored(5)])
    _r2, notes2 = _E.extract_evidence(
        scored(27), "edge versus cloud tradeoffs", None, None)
    assert E.integrity_notes(notes2), (
        "losing 15 of 27 passages is a real loss and must be flagged")


def test_a_healthy_run_raises_no_integrity_note(monkeypatch):
    """The warning must not fire on a run that actually worked."""
    monkeypatch.setattr(E, "ask", _answering_ask(good_sources=99))
    monkeypatch.setenv("EXTRACT_BATCH_CHARS", "700")

    records, notes = E.extract_evidence(
        _scored(3), "edge versus cloud tradeoffs", None, None)

    assert records
    assert E.integrity_notes(notes) == []


def test_partially_degraded_run_is_flagged(monkeypatch):
    """The live scenario: some batches answer, most do not."""
    monkeypatch.setattr(E, "ask", _answering_ask(good_sources=4))
    monkeypatch.setenv("EXTRACT_BATCH_CHARS", "700")

    _records, notes = E.extract_evidence(
        _scored(12), "edge versus cloud tradeoffs", None, None)

    flagged = E.integrity_notes(notes)
    assert len(flagged) == 1
    assert "passage batches" in flagged[0]


def test_totally_degraded_run_is_flagged(monkeypatch):
    monkeypatch.setattr(E, "ask", lambda *a, **k: {"unexpected": "shape"})
    monkeypatch.setenv("EXTRACT_BATCH_CHARS", "700")

    records, notes = E.extract_evidence(
        _scored(6), "edge versus cloud tradeoffs", None, None)

    assert records == []
    assert E.integrity_notes(notes)


def test_rejected_records_are_not_counted_as_silent(monkeypatch):
    """Discarding unverified quotes is the guard working, not a failure."""
    def ask(system, user, **kwargs):
        sources = [int(m) for m in re.findall(r"id=S(\d+)#", user)]
        return {"evidence": [
            {"chunk": f"P{i + 1}", "claim": "A claim carrying a number 42 here",
             "detail": "", "quote": "a quote that is nowhere in the passage",
             "dimension": "", "targets": [], "quality": "moderate",
             "limitations": "", "confidence": 0.6}
            for i in range(len(sources))]}

    monkeypatch.setattr(E, "ask", ask)
    monkeypatch.setenv("EXTRACT_BATCH_CHARS", "700")

    records, notes = E.extract_evidence(
        _scored(4), "edge versus cloud tradeoffs", None, None)

    assert any("discarded" in n for n in notes)
    assert E.integrity_notes(notes) == [], (
        "discarding unverified quotes is the guard doing its job")


# --- 2. and 3. it reaches the reader and the status ------------------------

def _report_with(notes):
    records = [Evidence(
        evidence_id="E1", claim="Edge offloading cuts measured latency",
        source_id="S1", source_url="https://e.com/1", source_title="D1",
        quality="strong", confidence=0.8, retrieval_status="full")]
    empty = type("C", (), {"list": lambda self: []})()
    return generate_report(
        "edge versus cloud tradeoffs", empty, [], records=records,
        source_synth={}, notes=notes, plan={"subquestions": []}, budget=None)


def test_degraded_run_is_partial_not_complete():
    notes = [E.INTEGRITY_PREFIX
             + "evidence extraction returned nothing usable for 8 of 12 batches"]
    result = _report_with(notes)
    assert result.status == "partial"
    assert any("integrity" in m for m in result.missing_sections)


def test_integrity_failure_is_visible_in_the_markdown():
    """It used to live in an HTML comment, which nobody reads."""
    notes = [E.INTEGRITY_PREFIX
             + "evidence extraction returned nothing usable for 8 of 12 batches"]
    markdown = _report_with(notes).markdown
    assert "How this run went" in markdown
    assert "PARTIAL REPORT" in markdown
    assert "degraded" in markdown


def test_banner_does_not_claim_sections_are_absent():
    """Every section was written; saying otherwise would be a different lie."""
    notes = [E.INTEGRITY_PREFIX + "evidence extraction returned nothing usable"]
    banner = partial_banner("partial", notes)
    assert "could not be written" not in banner
    assert "degraded" in banner


def test_banner_still_reports_genuinely_missing_sections():
    banner = partial_banner(
        "partial", ["Cost (no evidence was selected for this section)"])
    assert "could not be written" in banner
    assert "absent from this report" in banner


def test_integrity_and_missing_sections_are_both_reported():
    banner = partial_banner("partial", [
        "Cost (no evidence was selected for this section)",
        E.INTEGRITY_PREFIX + "evidence extraction returned nothing usable",
    ])
    assert "could not be written" in banner
    assert "degraded" in banner


def test_integrity_note_names_the_round_it_came_from():
    """Round 1 extracted 83 records; round 2 extracted 1.

    An unqualified "6 of 7 passage batches" on that run reads as a statement
    about the whole report, and a reader would conclude the document is thin
    when most of its evidence was fine.
    """
    from backend.research.graph import _attribute_round

    raw = [E.INTEGRITY_PREFIX
           + "evidence extraction returned nothing usable for 6 of 7 passage batches"]
    out = _attribute_round(raw, 2)
    assert out[0].startswith(E.INTEGRITY_PREFIX)
    assert "in round 2" in out[0]
    assert "6 of 7" in out[0]


def test_round_attribution_leaves_ordinary_notes_alone():
    from backend.research.graph import _attribute_round

    plain = ["skipped 12 passage(s) from sources already processed"]
    assert _attribute_round(plain, 3) == plain


def test_attributed_note_still_downgrades_and_stays_visible():
    from backend.research.graph import _attribute_round

    attributed = _attribute_round(
        [E.INTEGRITY_PREFIX + "extraction returned nothing usable"], 2)
    result = _report_with(attributed)
    assert result.status == "partial"
    assert "How this run went" in result.markdown
    assert "in round 2" in result.markdown


def test_report_event_carries_the_run_duration():
    """The frontend's finished timer prefers the server's figure.

    It has to be attached *after* the event is recorded: recording is what
    marks the bundle complete and stamps the duration, so asking beforehand
    always yields None and the browser silently falls back to its own count.
    """
    import asyncio
    import time

    from backend.research.researcher import deep_research

    os.environ["LLM_MOCK"] = "1"
    os.environ["SEARCH_MOCK"] = "1"
    os.environ["WIKI_ENABLED"] = "0"
    started = time.time()
    seen = {}

    async def run():
        async for ev in deep_research("duration wiring probe", max_rounds=1):
            if ev.get("type") == "report":
                seen.update(ev)
                return

    asyncio.run(run())
    elapsed = seen.get("elapsed_s")
    assert elapsed is not None, "the report event must carry elapsed_s"
    wall = time.time() - started
    assert elapsed > 0
    # Same clock, same instant: allow a generous window for model latency
    # between the bundle stamp and the assertion.
    assert abs(elapsed - wall) < 30, f"elapsed_s={elapsed:.1f} wall={wall:.1f}"


def test_advisory_notes_stay_out_of_the_way():
    """Only integrity notes are promoted; ordinary notes stay in the comment."""
    notes = ["skipped 4 passage(s) from sources already processed",
             E.INTEGRITY_PREFIX + "evidence extraction returned nothing usable"]
    markdown = _report_with(notes).markdown
    assert "How this run went" in markdown
    comment = re.search(r"<!-- pipeline notes: (.*?) -->", markdown, re.S)
    assert comment, "advisory notes should still be recorded"
    assert "already processed" in comment.group(1)
    assert "integrity" not in comment.group(1)