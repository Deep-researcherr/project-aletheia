"""Tests for the reference-card overflow that killed whole PDFs.

``report.py`` hides non-critical pipeline advisories in an HTML comment --
correct for the ``.md`` export, where a renderer ignores it. The PDF body
renderer skips comment blocks, but ``parse_references`` did not: a comment
landing inside the References section fell through to the catch-all branch and
was absorbed into that reference's ``meta``.

``meta`` is unbounded, so the comment became a ~4300-character field, which
became a single-cell Table 718pt tall against a 703.89pt frame. ReportLab
cannot place a row taller than the frame -- it raises LayoutError, and one
oversized card destroyed the entire document.

Also covers a second, independent defect found in the same card: ``
build_ref_notes`` already prefixes "Cited from:" for the citation list, and
the caller appended the same prefixed list again, printing it twice.
"""
import pytest
from reportlab.platypus import Table

from backend.research import pdf_report as pr

# The frame height that a reference card has to fit inside (A4 minus margins).
FRAME_H = 703.89

REPORT = """# Report

Some body text that makes the document non-trivial.

## References
1. **What Are qLDPC Codes? Surface Codes & Future Outlook** — https://example.com/a
Cited from: p.12, p.19, p.24
Cited from: p.40, p.41

## How this run went

The following affected the evidence this report rests on:

- In round 1, evidence extraction returned nothing usable for 30 of 40 batches.
"""

# Exactly the shape report.py emits: a long advisory hidden in a comment.
LEAKY_REPORT = REPORT.replace(
    "## How this run went",
    "<!-- pipeline notes: wikipedia baseline: 3 article(s) pre-loaded; "
    "71 source(s) could not be retrieved (robots.txt, paywall, bot protection, "
    "rate limit or fetch budget); they are cited as metadata-only; "
    "extraction batch 1 returned no evidence for 2 passage(s); "
    "extraction batch 2 returned no evidence -->\\n\\n## How this run went",
)


def _card_height(markdown, num="1"):
    """Measure a reference card exactly as render_refs builds it."""
    S = pr.styles()
    refs = pr.parse_references(markdown)
    notes = pr.build_ref_notes(refs, markdown, [])
    ref = refs[num]
    link = (ref.get("url") or "").strip()
    html = (
        f'<font size="13" color="#1B2A4A"><b>[{num}]</b></font>'
        f'&nbsp;&nbsp;<b>{pr.esc(ref.get("title", ""))}</b><br/>'
        + (f'<link href="{pr.esc(link)}" color="#1D4ED8">{pr.esc(link)}</link>'
           if link else "")
        + f' &nbsp;<font color="#5B6B7F">'
          f'{pr.esc(pr._cap(ref.get("meta", ""), pr.META_MAX_CHARS))}</font><br/>'
        f'<b>Note &mdash; what [{num}] gives you:</b> '
        f'{pr.esc(pr._cap(notes.get(num, ""), pr.NOTE_MAX_CHARS))}'
    )
    card = Table([[pr.P(html, S, "ref")]], colWidths=[410], splitInRow=1)
    card.setStyle(pr.TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return card.wrap(410, 9999)[1]


class TestStripHtmlComments:
    def test_removes_comment_body(self):
        assert "<!--" not in pr.strip_html_comments(LEAKY_REPORT)

    def test_leaves_visible_sections_alone(self):
        out = pr.strip_html_comments(LEAKY_REPORT)
        assert "## How this run went" in out
        assert "30 of 40" in out, "critical notes must survive, not be stripped"

    def test_preserves_references(self):
        out = pr.strip_html_comments(LEAKY_REPORT)
        assert "What Are qLDPC Codes?" in out
        assert "https://example.com/a" in out

    def test_handles_empty_and_none(self):
        assert pr.strip_html_comments("") == ""
        assert pr.strip_html_comments(None) == ""

    def test_multiline_comment(self):
        assert pr.strip_html_comments("a<!-- x\ny\nz -->b") == "ab"

    def test_unterminated_comment_is_left_alone(self):
        # Better a stray "<!--" than silently eating the rest of a report.
        assert pr.strip_html_comments("a<!-- b") == "a<!-- b"


class TestMetaNoLongerAbsorbsComment:
    def test_comment_does_not_reach_meta(self):
        refs = pr.parse_references(pr.strip_html_comments(LEAKY_REPORT))
        assert "<!--" not in refs["1"]["meta"]

    def test_meta_stays_small(self):
        refs = pr.parse_references(pr.strip_html_comments(LEAKY_REPORT))
        # The real report's ref 15 carried 4,292 chars here.
        assert len(refs["1"]["meta"]) < 200

    def test_visible_integrity_note_survives_into_report(self):
        """The reason the comment exists at all: hidden, not deleted."""
        out = pr.strip_html_comments(LEAKY_REPORT)
        assert "extraction returned nothing usable" in out


class TestCardFitsTheFrame:
    def test_leaky_report_card_fits(self):
        assert _card_height(pr.strip_html_comments(LEAKY_REPORT)) < FRAME_H

    def test_clean_report_card_fits(self):
        assert _card_height(REPORT) < FRAME_H

    def test_absurd_meta_is_capped(self):
        md = REPORT.replace(
            "1. **What Are qLDPC Codes? Surface Codes & Future Outlook** — https://example.com/a",
            "1. **What Are qLDPC Codes? Surface Codes & Future Outlook** — "
            "https://example.com/a " + ("padding words here " * 900),
        )
        refs = pr.parse_references(pr.strip_html_comments(md))
        meta = pr._cap(refs["1"]["meta"], pr.META_MAX_CHARS)
        assert len(meta) <= pr.META_MAX_CHARS + 4   # + ellipsis
        assert _card_height(pr.strip_html_comments(md)) < FRAME_H


class TestCap:
    def test_short_text_untouched(self):
        assert pr._cap("hello world", 100) == "hello world"

    def test_truncates_on_word_boundary(self):
        out = pr._cap("alpha beta gamma delta", 12)
        assert out.startswith("alpha beta")
        assert not out.startswith("alpha betag")

    def test_marks_truncation(self):
        assert "…" in pr._cap("alpha beta gamma delta", 12)

    def test_collapses_whitespace(self):
        assert pr._cap("a   b\n\nc", 100) == "a b c"

    def test_empty(self):
        assert pr._cap("", 10) == ""
        assert pr._cap(None, 10) == ""


class TestNoDuplicateCitedFrom:
    def test_citation_list_printed_once(self):
        notes = pr.build_ref_notes(pr.parse_references(REPORT), REPORT, [])
        assert notes["1"].count("Cited from:") == 1

    def test_locations_not_repeated(self):
        notes = pr.build_ref_notes(pr.parse_references(REPORT), REPORT, [])
        assert notes["1"].count("p.12, p.19, p.24") == 1


class TestBuildPdfEndToEnd:
    def test_leaky_report_still_builds(self):
        """The regression that mattered: build_pdf used to raise here."""
        pdf = pr.build_pdf({"question": "q?", "report": LEAKY_REPORT})
        assert pdf.startswith(b"%PDF-")

    def test_citation_targets_survive(self):
        pdf = pr.build_pdf({"question": "q?", "report": LEAKY_REPORT})
        assert len(pdf) > 1000

    def test_empty_report_still_rejected(self):
        with pytest.raises(ValueError):
            pr.build_pdf({"question": "q?", "report": "<!-- only a comment -->"})


class TestSplitSafetyNet:
    def test_card_declares_split_in_row(self):
        """A card that somehow still overflows must not kill the document."""
        import inspect
        src = inspect.getsource(pr.render_refs)
        assert "splitInRow=1" in src


class TestTrailingDisclosures:
    """The integrity disclosure must reach the PDF.

    ``report.py`` writes "How this run went" *after* the References block.
    ``build_pdf`` cuts the body at "## Evidence by Source", which comes
    *before* References, and ``render_refs`` only draws cards -- so the
    disclosure was rendered nowhere. On a PARTIAL run the evidence-integrity
    warning existed in the .md and was absent from the PDF a reader receives.
    """

    def test_extracts_heading_and_body(self):
        out = pr.trailing_disclosures(REPORT)
        assert out and out[0][0] == "How this run went"
        assert "30 of 40" in out[0][1]

    def test_ignores_reference_lines(self):
        out = pr.trailing_disclosures(REPORT)
        assert all("example.com" not in body for _, body in out)

    def test_no_disclosures_without_references(self):
        md = "## Conclusion\n\nAll good.\n"
        assert pr.trailing_disclosures(md) == []

    def test_handles_references_being_last(self):
        md = "## References\n\n1. **X** — https://e.com\n"
        assert pr.trailing_disclosures(md) == []

    def test_renders_into_pdf(self):
        pdf = pr.build_pdf({"question": "q?", "report": REPORT})
        text = _pdf_text(pdf)
        assert "How this run went" in text, \
            "the integrity disclosure is missing from the PDF"
        assert "30 of 40" in text, \
            "the warning body is missing from the PDF"

    def test_hidden_comment_content_stays_hidden(self):
        """Stripped from the card, but not promoted into visible text."""
        pdf = pr.build_pdf({"question": "q?", "report": LEAKY_REPORT})
        assert pdf.startswith(b"%PDF-")
        assert "wikipedia baseline" not in _pdf_text(pdf)


def _pdf_text(pdf_bytes):
    import io
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join((p.extract_text() or "") for p in reader.pages)