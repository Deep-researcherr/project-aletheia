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
    """A provider that answers for the first N sources, then goes silent."""
    def ask(system, user, **kwargs):
        sources = [int(m) for m in re.findall(r"id=S(\d+)#", user)]
        if not sources or min(sources) <= good_sources:
            return {"evidence": [
                {"chunk": f"P{i + 1}",
                 "claim": f"Measured claim {i} about edge throughput",
                 "detail": "", "quote": "", "dimension": "", "targets": [],
                 "quality": "moderate", "limitations": "", "confidence": 0.6}
                for i in range(len(sources))]}
        return {"evidence": []}
    return ask


# --- 1. the signal exists at all -------------------------------------------

def test_a_silent_empty_batch_is_reported(monkeypatch):
    """No records and nothing discarded means the reply was unusable."""
    monkeypatch.setattr(E, "ask", lambda *a, **k: {"evidence": []})
    monkeypatch.setenv("EXTRACT_BATCH_CHARS", "700")

    _records, notes = E.extract_evidence(
        _scored(3), "edge versus cloud tradeoffs", None, None)

    empties = [n for n in notes if "returned no evidence" in n]
    assert empties, "a batch that produced nothing left no trace"


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
    monkeypatch.setattr(E, "ask", lambda *a, **k: {"evidence": []})
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