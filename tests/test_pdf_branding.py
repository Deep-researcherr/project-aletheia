"""The exported PDF's branding.

There were no tests covering this at all, which is why the header, the
per-page footer and the document metadata were three independent string literals
that could each say something different. They now share one constant; these
tests exist so a rename cannot half-apply.

The three places matter for different reasons. The header band is what a reader
sees first and names the document. The footer repeats on every page, so a
mismatch reads as a printing fault rather than as branding. The metadata author
survives long after the file is renamed on disk, and is what a PDF viewer
shows in its properties panel.
"""
import re

import pytest

from backend.research.pdf_report import PROJECT_NAME, build_pdf


@pytest.fixture(scope="module")
def pdf_text():
    """One rendered PDF, shared by the assertions below."""
    from pypdf import PdfReader
    import io

    data = build_pdf({
        "question": "What actually causes retrieval failures in agentic search?",
        "report": "## Summary\n\nA finding about retrieval [1].\n",
        "stats": {"sources": 3},
        "elapsed_s": 120,
    })
    reader = PdfReader(io.BytesIO(data))
    return {
        "bytes": data,
        "metadata": reader.metadata,
        "pages": [p.extract_text() or "" for p in reader.pages],
        "text": "\n".join(p.extract_text() or "" for p in reader.pages),
    }


def test_the_project_name_is_the_branded_one(pdf_text):
    assert PROJECT_NAME == "Project Aletheia"


def test_the_header_band_names_the_project(pdf_text):
    first = pdf_text["pages"][0]
    assert "PROJECT ALETHEIA" in first.upper(), \
        "the top band does not carry the project name"
    # Uppercased in the band, which is set in caps.
    assert f"{PROJECT_NAME.upper()}  —  RESEARCH REPORT" in first


def test_no_legacy_name_survives_anywhere_in_the_document(pdf_text):
    """The old name, in any casing, anywhere.

    Checked over the whole rendered text and the metadata, because a leftover
    in the footer is just as visible as one in the header -- it is on every
    page.
    """
    blob = pdf_text["text"] + " " + str(pdf_text["metadata"])
    assert "deep researcher" not in blob.lower(), \
        "the old product name is still present in the exported PDF"
    assert "DEEP RESEARCHER" not in blob


def test_the_footer_carries_the_project_name_on_every_page(pdf_text):
    slug = "what-actually-causes-retrieval-failures-in-agentic-search"
    for number, page in enumerate(pdf_text["pages"], start=1):
        assert PROJECT_NAME in page, f"page {number} footer lost the name"
        assert slug in page, f"page {number} footer lost the question slug"


def test_the_metadata_author_is_the_project(pdf_text):
    assert pdf_text["metadata"].author == PROJECT_NAME


def test_the_metadata_title_is_still_the_question(pdf_text):
    """Renaming the author must not have swallowed the title."""
    assert pdf_text["metadata"].title.startswith("What actually causes")


def test_the_name_is_a_single_constant_not_repeated():
    """Three call sites, one source. A literal in any of them is the drift."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] \
        / "backend" / "research" / "pdf_report.py"
    text = src.read_text()
    body = text.split("PROJECT_NAME =", 1)[1]
    for line in body.splitlines()[1:]:
        assert "Project Aletheia" not in line and \
               "DEEP RESEARCHER" not in line and \
               '"Deep Researcher"' not in line, \
            f"a name literal crept back in: {line.strip()[:70]}"
    assert body.count("PROJECT_NAME") >= 3, \
        "header, footer and metadata should all read the constant"


def test_the_report_still_renders_with_content(pdf_text):
    """Guards against a rename that breaks the document rather than renaming
    it."""
    assert len(pdf_text["bytes"]) > 2000
    assert "Summary" in pdf_text["text"]
    assert PROJECT_NAME in re.sub(r"\s+", " ", pdf_text["text"])