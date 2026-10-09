"""Dynamic Save PDF: recorder, builder, and endpoint.

Proves the PDF is compiled from the recorded run bundle (the same payloads
the frontend receives) with working internal citation links, and that the
endpoint 404s without a completed run.
"""
import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.research import pdf_report, runlog


@pytest.fixture(autouse=True)
def clean_runs():
    runlog.reset()
    yield
    runlog.reset()


def _bundle():
    markdown = """# Question?

*Research synthesis generated 2026-10-07. 2 distinct sources, 3 traceable findings.*

## Executive Summary

Solubility ($\\log S$) rises with descriptors [1]. Error falls from 0.84 to 0.35 [1][2].

## Scope and Methodology

Methods compared on AqSolDB [2].

## Evidence by Source

| Ref | Source | Type | Published | Retrieved | Findings | Strength |
| --- | --- | --- | --- | --- | --- | --- |
| [1] | Alpha Paper | web page | 2024-01-01 | full | 2 | strong |
| [2] | Beta Paper | preprint | undated | full | 1 | moderate |

## References

1. **Alpha Paper** — Alice; 2024-01-01; web page.
   <https://a.dev/1>
   Cited from: Abstract

2. **Beta Paper** — Bob; preprint.
   <https://b.dev/2>
   Cited from: Methods
"""
    return {
        "question": "Test question?",
        "max_rounds": 2,
        "plan": {"subquestions": [
            {"question": "s1",
             "searches": [{"source": "web", "query": "q1"}]},
            {"question": "s2",
             "searches": [{"source": "scholar", "query": "q2"}]}]},
        "searches": [
            {"query": "q1", "source": "web", "count": 6, "error": None},
            {"query": "q2", "source": "scholar", "count": 4, "error": None},
        ],
        "evidence_count": 2,
        "retrieval": {"attempted": 2, "retrieved": 2, "full_text": 2,
                      "partial": 0, "metadata_only": 0, "failed": 0,
                      "total_words": 900, "chunks": 4,
                      "references_followed": 0,
                      "sources": [
                          {"status": "full", "method": "html", "words": 500,
                           "title": "Alpha Paper", "url": "https://a.dev/1",
                           "limitation": ""},
                          {"status": "full", "method": "html", "words": 400,
                           "title": "Beta Paper", "url": "https://b.dev/2",
                           "limitation": ""},
                      ]},
        "analysis": {"records": 3, "dimensions": ["throughput"],
                     "sources_summarised": 2, "notes": ["note one"]},
        "gaps": [{"sufficient": True, "missing": "", "follow_ups": []}],
        "contradictions": [],
        "sections": ["Executive Summary", "Scope and Methodology"],
        "report": markdown,
        "report_html": "",
        "sources": [{"url": "https://a.dev/1"},
                    {"url": "https://b.dev/2"}],
        "stats": {"records": 3, "sources": 2,
                  "by_quality": {"strong": 2, "moderate": 1}},
        "export": {"written": True},
        "status": {"status": "completed", "missing_sections": [],
                   "headline": "COMPLETE REPORT"},
        "complete": True,
    }


def test_fix_maths_humanises_notation():
    assert "log <i>S</i>" in pdf_report.fix_maths("$\\log S$")
    assert "R<super>2</super>" in pdf_report.fix_maths("$R^2$")
    assert "<i>N</i> = 500" in pdf_report.fix_maths("$N=500$")
    assert pdf_report.fix_maths("$pK_a$") == "p<i>K</i><sub>a</sub>"
    assert "\u00b1" in pdf_report.fix_maths("\\pm")
    # Divisions inside URLs must never be rewritten.
    url = "https://hdl.handle.net/20.500.14721/39891"
    assert pdf_report.fix_maths(url) == url


def test_fix_cites_links_each_number():
    out = pdf_report.fix_cites("claims [1] and [2][3].")
    assert out.count('href="#ref-') == 3
    assert 'href="#ref-2"' in out and 'href="#ref-3"' in out


def test_fix_cites_skips_numbers_without_references():
    out = pdf_report.fix_cites("claim [9] here.", known={"1"})
    assert 'href="#ref-' not in out
    assert ">9<" in out


def _text_of(pdf: bytes) -> str:
    from pypdf import PdfReader
    import io
    return "\n".join(p.extract_text() or ""
                      for p in PdfReader(io.BytesIO(pdf)).pages)


def test_duration_is_printed_on_the_cover():
    """A reader cannot tell a two-minute lookup from a twenty-minute run
    without it, which is much of what makes the artefact believable."""
    from backend.research.pdf_report import _duration

    assert _duration(9) == "9s"
    assert _duration(252) == "4m 12s"
    assert _duration(3661) == "1h 01m 01s"
    assert _duration(None) == "unknown"
    assert _duration("nonsense") == "unknown"

    pdf = _bundle()
    pdf["elapsed_s"] = 423
    text = _text_of(pdf_report.build_pdf(pdf, "2 rounds", ""))
    assert "7m 03s" in text
    assert "run duration" in text


def test_run_label_omits_an_absent_throttle_mode():
    """throttle_label is empty when the caller did not pin one, which used to
    render "(2 rounds, )"."""
    text = _text_of(pdf_report.build_pdf(_bundle(), "2 rounds", ""))
    assert "(2 rounds)" in text
    assert ", )" not in text


def test_partial_reason_gets_a_readable_band_not_a_cramped_tile():
    """The full headline was being wrapped two words per line inside a 130pt
    stat tile."""
    bundle = _bundle()
    bundle["status"] = {
        "status": "partial",
        "headline": ("PARTIAL REPORT — evidence extraction returned nothing "
                     "usable for 6 of 7 passage batches"),
        "missing_sections": ["Evidence integrity: in round 2, extraction "
                             "returned nothing usable"],
    }
    text = _text_of(pdf_report.build_pdf(bundle, "2 rounds", ""))
    assert "Why:" in text
    assert "PARTIAL" in text
    # The trailing "Missing from this report" line was a fourth copy of the
    # same sentence; the cover band and the body banner already say it.
    assert "Missing from this report" not in text


def test_build_ref_notes_is_deterministic():
    refs = {"1": {"title": "Alpha", "url": "https://a.dev/1",
                  "cited": ["Abstract"]}}
    markdown = ("## Evidence by Source\n\n"
                "| Ref | Source | Type | Published | Retrieved | Findings | Strength |\n"
                "| --- | --- | --- | --- | --- | --- | --- |\n"
                "| [1] | Alpha | web page | 2024 | full | 2 | strong |\n"
                "\n## References\n\n1. **Alpha** — Alice.\n   <https://a.dev/1>\n")
    sources = [{"url": "https://a.dev/1", "status": "full"}]

    first = pdf_report.build_ref_notes(refs, markdown, sources)
    second = pdf_report.build_ref_notes(refs, markdown, sources)

    assert first == second
    assert first["1"] == ("2 finding(s), strong evidence; "
                          "Cited from: Abstract")


def test_builder_produces_pdf_with_working_internal_links():
    pdf = pdf_report.build_pdf(_bundle(), "2 rounds", "throttled")

    assert pdf[:5] == b"%PDF-"

    from pypdf import PdfReader
    import io
    reader = PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) >= 2
    gotos = sum(
        1 for page in reader.pages for annot in (page.get("/Annots") or [])
        if str(annot.get_object().get("/Subtype", "")) == "/Link"
        and annot.get_object().get("/Dest") is not None)
    assert gotos >= 3, "citation taps must resolve inside the document"
    full = "\n".join(p.extract_text() or "" for p in reader.pages)
    assert "Test question?" in full
    assert "Alpha Paper" in full
    assert "log S" in full


def test_builder_rejects_empty_reports():
    bundle = _bundle()
    bundle["report"] = "   "
    with pytest.raises(ValueError):
        pdf_report.build_pdf(bundle)


def test_recorder_accumulates_a_run():
    runlog.begin("Q?", 2)
    runlog.record("Q?", {"type": "plan", "data": {"subquestions": []}})
    runlog.record("Q?", {"type": "results", "query": "q", "source": "web",
                         "count": 3, "error": None})
    assert runlog.get("Q?") is None  # not complete yet
    runlog.record("Q?", {"type": "report", "data": "# R", "html": "",
                         "sources": [], "stats": {}, "export": {},
                         "status": {}})
    bundle = runlog.get("Q?")
    assert bundle["complete"] is True
    assert bundle["searches"][0]["query"] == "q"


def test_pdf_endpoint_404_without_a_run():
    client = TestClient(main.app)

    assert client.get("/api/report/pdf?q=nope").status_code == 404


def test_pdf_endpoint_serves_recorded_run():
    runlog.begin("Test question?", 2)
    bundle = _bundle()
    for kind, payload in [
            ("plan", {"data": bundle["plan"]}),
            ("report", {"data": bundle["report"], "html": "",
                        "sources": bundle["sources"], "stats": bundle["stats"],
                        "export": bundle["export"], "status": bundle["status"]})]:
        event = {"type": kind}
        event.update(payload)
        runlog.record("Test question?", event)

    client = TestClient(main.app)
    response = client.get("/api/report/pdf?q=Test%20question%3F")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content[:5] == b"%PDF-"
    assert "attachment" in response.headers["content-disposition"]
