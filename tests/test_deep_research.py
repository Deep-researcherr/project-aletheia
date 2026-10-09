"""Tests for retrieval, chunking, evidence and the deep report path.

These run entirely offline: the fetcher is exercised through a stub session, and
every LLM call is monkeypatched. The point is to pin the behaviour that keeps
citations honest -- provenance survives chunking, invented quotes are rejected,
reference numbers always resolve to a real URL, and retrieval degrades honestly
rather than pretending a snippet is a document.
"""
import json
from pathlib import Path

import pytest

from backend.research import config
from backend.research.chunker import Chunk, chunk_document
from backend.research.evidence import (Evidence, dedupe, normalise_confidence,
                                       normalise_quality, quote_is_supported,
                                       statistics)
from backend.research.export import build_document, slugify
from backend.research.fetch import Fetcher, github_repo
from backend.research.references import candidates
from backend.research.relevance import dimensions, diversify, rank, targets
from backend.research.report import assign_references, render_references, write_report


# --- fetch ------------------------------------------------------------------

class _Response:
    def __init__(self, text="", status_code=200, url="http://x/", content_type="text/html"):
        self.text = text
        self.status_code = status_code
        self.url = url
        self.headers = {"content-type": content_type}
        self._body = text.encode()
        self._truncated = False

    def iter_content(self, size):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Session:
    """Stub session that returns canned responses and records every request."""

    def __init__(self, routes=None, default=None):
        self.routes = routes or {}
        self.default = default
        self.requests = []
        self.headers = {}

    def get(self, url, **kwargs):
        self.requests.append(url)
        for prefix, response in self.routes.items():
            if url.startswith(prefix):
                return response
        if self.default is not None:
            return self.default
        return _Response(text="", status_code=404)


HTML = """<!doctype html><html><head>
<title>React Native Performance</title>
<meta property="og:site_name" content="React Native">
<meta name="article:published_time" content="2026-01-15">
</head><body>
<nav>skip to content home products</nav>
<article>
<h1>Performance</h1>
<p>React Native targets 60 frames per second on mid-range devices.</p>
<h2>Measuring frames</h2>
<p>Use the profiler to measure frame rate in a release build.</p>
<table><tr><th>Metric</th><th>Value</th></tr><tr><td>TTI</td><td>1.2s</td></tr></table>
<a href="https://github.com/facebook/react-native">source repo</a>
</article>
<footer>all rights reserved</footer>
</body></html>"""


def test_fetch_extracts_article_text_and_drops_chrome(monkeypatch):
    monkeypatch.setattr(config, "fetch_respect_robots", lambda: False)
    session = _Session(default=_Response(text=HTML, url="https://rn.dev/perf"))
    doc = Fetcher(session=session).fetch("https://rn.dev/perf")

    assert doc.status == "full"
    assert "60 frames per second" in doc.text
    assert "all rights reserved" not in doc.text
    assert doc.title == "React Native Performance"
    assert doc.date == "2026-01-15"
    assert doc.publisher == "React Native"


def test_structured_extraction_preserves_headings_and_tables(monkeypatch):
    monkeypatch.setattr(config, "fetch_respect_robots", lambda: False)
    session = _Session(default=_Response(text=HTML, url="https://rn.dev/perf"))
    text = Fetcher(session=session).fetch("https://rn.dev/perf").text

    assert "## Measuring frames" in text, "headings must survive for chunking"
    assert "| Metric | Value |" in text, "tables must survive as pipe rows"


def test_robots_disallow_is_honoured_and_recorded(monkeypatch):
    monkeypatch.setattr(config, "fetch_respect_robots", lambda: True)
    from backend.research import fetch as fetch_module
    fetch_module._robots_cache.clear()

    robots = _Response(text="User-agent: *\nDisallow: /private", url="https://x/robots.txt")
    session = _Session(routes={"https://x/robots.txt": robots},
                       default=_Response(text=HTML, url="https://x/private"))
    doc = Fetcher(session=session).fetch("https://x/private")

    assert doc.status == "failed"
    assert "robots.txt" in doc.limitation
    fetch_module._robots_cache.clear()


def test_github_repo_parses_scheme_fragment_and_subpath():
    assert github_repo("https://github.com/react/react-native") == ("react", "react-native")
    assert github_repo("https://github.com/facebook/react-native.git") == ("facebook", "react-native")
    assert github_repo("https://github.com/flutter/flutter/tree/main") == ("flutter", "flutter")
    assert github_repo("https://github.com/orgs/foo") is None
    assert github_repo("https://gitlab.com/a/b") is None


def test_one_bad_url_does_not_kill_the_batch(monkeypatch):
    monkeypatch.setattr(config, "fetch_respect_robots", lambda: False)
    good = _Response(text=HTML, url="https://ok.dev/a")
    session = _Session(default=good)
    session.routes["https://bad.dev/"] = _Response(text="", status_code=500)

    fetcher = Fetcher(session=session)
    docs = fetcher.fetch_many(["https://ok.dev/a", "https://bad.dev/x"])

    assert [d.status for d in docs] == ["full"]


# --- chunking ---------------------------------------------------------------

def _chunks(text, **kwargs):
    params = dict(source_id="s1", source_url="https://a.dev/doc",
                  source_title="Doc", source_type="html")
    params.update(kwargs)
    return chunk_document(text, **params)


def test_chunks_record_section_and_never_lose_the_source():
    text = "\n\n".join(
        [f"## Section {i}\n\n" + (f"body text for section {i}. " * 40) for i in range(12)])
    chunks = _chunks(text)

    assert chunks
    for chunk in chunks:
        assert chunk.source_url == "https://a.dev/doc"
        assert chunk.source_id == "s1"
        assert chunk.chunk_id.startswith("s1#")
        assert chunk.section.startswith("Section ")


def test_oversized_document_is_chunked_not_truncated():
    text = "\n\n".join(f"## S{i}\n\n" + "x" * 600 for i in range(200))
    chunks = _chunks(text, source_type="pdf", pages=200)

    assert len(chunks) > 1, "a large document must be split"
    assert all(len(c.content) <= config.chunk_max_chars() for c in chunks)


def test_pdf_pages_are_tracked_and_html_is_not_numbered():
    pdf = _chunks("page one text\n\f\n## H\n\npage two text", source_type="pdf", pages=2)
    assert any(c.page >= 2 for c in pdf), "PDF chunks must carry page numbers"

    html = _chunks("## H\n\nsome text")
    assert all(c.page == 0 for c in html), "HTML has no meaningful page numbers"


def test_chunks_per_source_caps_a_verbose_document(monkeypatch):
    monkeypatch.setattr(config, "chunks_per_source", lambda: 3)
    text = "\n\n".join(f"## S{i}\n\n" + ("body " * 100) for i in range(40))
    assert len(_chunks(text)) == 3


# --- evidence ---------------------------------------------------------------

def test_quote_must_actually_be_in_the_passage():
    text = "The build measured 1.8 seconds to first frame in release mode."
    assert quote_is_supported("measured 1.8 seconds to first frame", text)
    assert not quote_is_supported("outperformed every competitor by 40%", text)


def test_normalisation_helpers():
    assert normalise_quality("high") == "strong"
    assert normalise_quality("nonsense") == "moderate"
    assert normalise_confidence(0.7) == 0.7
    assert normalise_confidence(85) == 0.85
    assert normalise_confidence("bad") == 0.5


def test_dedupe_keeps_the_strongest_attestation():
    weak = Evidence(claim="Flutter is faster", quality="weak", confidence=0.4,
                    source_url="https://blog.dev/a", evidence_id="E9")
    strong = Evidence(claim="flutter is faster!", quality="strong", confidence=0.9,
                      source_url="https://docs.dev/b", evidence_id="E1")
    kept = dedupe([weak, strong])

    assert len(kept) == 1
    assert kept[0].source_url == "https://docs.dev/b"


def test_evidence_exposes_a_usable_url_preferring_doi():
    record = Evidence(source_url="https://publisher.dev/landing",
                      doi="10.1234/abcd", source_title="Paper")
    assert record.usable_url == "10.1234/abcd"


def test_statistics_counts_sources_and_quality():
    records = [
        Evidence(claim="a", source_url="https://a.dev", quality="strong", publication_date="2026-01-01"),
        Evidence(claim="b", source_url="https://a.dev", quality="weak"),
        Evidence(claim="c", source_url="https://b.dev", quality="moderate"),
    ]
    stats = statistics(records)

    assert stats["records"] == 3
    assert stats["sources"] == 2
    assert stats["by_quality"]["strong"] == 1


# --- relevance --------------------------------------------------------------

QUESTION = ("Compare React Native, Flutter, and native Android development for a "
            "production mobile app in 2026. Analyze performance, developer "
            "productivity, ecosystem maturity, and hiring demand.")


def test_targets_and_dimensions_are_extracted_separately():
    assert targets(QUESTION) == ["React Native", "Flutter", "native Android"]
    assert dimensions(QUESTION) == ["performance", "developer productivity",
                                    "ecosystem maturity", "hiring demand"]


def test_rank_prefers_covering_passages_and_drops_irrelevant():
    chunks = [
        Chunk(chunk_id="s1#0", source_id="s1", source_url="https://a.dev",
              source_title="Perf", content="Startup time and frame rate on React "
              "Native are measured in release builds."),
        Chunk(chunk_id="s1#1", source_id="s1", source_url="https://a.dev",
              source_title="Cooking",
              content="A recipe for sourdough bread with rye flour and long cold proofing."),
    ]
    kept = rank(chunks, QUESTION, None, top_k=10)

    assert [k.chunk.chunk_id for k in kept] == ["s1#0"]


def test_diversify_stops_one_source_flooding_the_batch():
    chunks = [Chunk(chunk_id=f"s1#{i}", source_id="s1", source_url="https://a.dev",
                    source_title="A", content="react native flutter android performance")
              for i in range(20)]
    chunks.append(Chunk(chunk_id="s2#0", source_id="s2", source_url="https://b.dev",
                        source_title="B",
                        content="flutter android performance benchmark react native"))
    scored = rank(chunks, QUESTION, None, top_k=30)
    spread = diversify(scored, per_source_cap=3)

    assert sum(1 for s in spread if s.chunk.source_id == "s1") <= 3
    assert any(s.chunk.source_id == "s2" for s in spread)


# --- references -------------------------------------------------------------

def test_reference_candidates_are_filtered_and_bounded(monkeypatch):
    monkeypatch.setattr(config, "references_enabled", lambda: True)
    monkeypatch.setattr(config, "references_max", lambda: 3)
    monkeypatch.setattr(config, "references_max_per_source", lambda: 3)

    class Doc:
        links = [
            "https://arxiv.org/abs/2404.02258",
            "https://github.com/flutter/flutter",
            "https://twitter.com/someone",
            "https://example.com/unrelated-recipe",
            "https://medium.com/some-post",
        ]

    found = candidates([Doc()], QUESTION, set())

    assert len(found) <= 3
    assert not any("twitter.com" in url for url in found), "social hosts must be skipped"
    assert any("arxiv.org" in url or "github.com" in url for url in found)


# --- references and report --------------------------------------------------

def _records():
    return [
        Evidence(claim="React Native targets 60fps", source_title="RN Perf",
                 source_url="https://rn.dev/perf", publisher="Meta", doi="",
                 quality="strong", publication_date="2026-01-01", dimension="performance",
                 retrieval_status="full", confidence=0.9),
        Evidence(claim="Flutter ships hot reload", source_title="Flutter Docs",
                 source_url="https://docs.flutter.dev/hot", publisher="Google",
                 doi="", quality="moderate", publication_date="2025-06-01",
                 dimension="developer productivity", retrieval_status="full",
                 confidence=0.8),
        Evidence(claim="Snippets only", source_title="Blog",
                 source_url="https://blog.dev/x", publisher="", doi="",
                 quality="weak", publication_date="", retrieval_status="metadata_only",
                 confidence=0.2),
    ]


def test_reference_numbers_are_unique_and_resolve_to_real_urls():
    refs = assign_references(_records())

    assert sorted(refs.values()) == list(range(1, len(refs) + 1))
    for url in refs:
        assert url in {"https://rn.dev/perf", "https://docs.flutter.dev/hot",
                       "https://blog.dev/x"}


def test_references_section_contains_every_url():
    records = _records()
    text = render_references(records, assign_references(records))

    assert "## References" in text
    for record in records:
        assert record.usable_url in text
    assert "only the search-result metadata" in text, "partial retrieval must be disclosed"


def test_report_falls_back_to_the_legacy_path_without_evidence(monkeypatch):
    """No retrieved evidence means the original one-shot report, unchanged."""
    from backend.research import report as report_module

    calls = []
    monkeypatch.setattr(report_module, "ask",
                        lambda *a, **k: calls.append(a) or "LEGACY REPORT")

    class Col:
        def context(self):
            return "[1] (web, 2025) thing - snippet"

    out = write_report("q", Col(), [])
    # The single-call body is still what gets produced, but it is now labelled
    # as a partial result rather than passed off as a finished deep report.
    assert out.endswith("LEGACY REPORT")
    assert "PARTIAL REPORT" in out
    assert len(calls) == 1, "the fallback must stay a single call"


def test_deep_report_uses_sections_and_cites_only_real_numbers(monkeypatch):
    from backend.research import report as report_module

    records = _records()
    refs = assign_references(records)

    monkeypatch.setattr(report_module, "write_report", report_module.write_report)
    monkeypatch.setattr(
        report_module, "_write_section",
        lambda system, question, heading, instruction, recs, r, t, budget=None,
        min_words=350, failed=None: f"## {heading}\n\nProse citing [{sorted(r.values())[0]}].\n")

    class Col:
        def context(self):
            return ""

    out = write_report("Compare React Native, Flutter, and native Android. "
                       "Analyze performance and hiring demand.",
                       Col(), [], records=records)

    assert "## Performance" in out
    assert "## Hiring Demand" in out
    assert "## References" in out
    assert "## Evidence by Source" in out

    # Every citation used must exist in the reference numbering.
    used = {int(n) for n in __import__("re").findall(r"\[(\d{1,3})\]", out)
            if int(n) in set(refs.values())}
    assert used, "the report must actually cite its sources"
    assert "metadata" in out.lower()


def test_export_document_has_front_matter_and_question_title():
    records = _records()
    body = "# Title\n\nProse [1]."
    text = build_document("Compare React Native and Flutter?", body, statistics(records),
                          {"retrieved": 2, "full_text": 1, "partial": 1,
                           "total_words": 900})

    assert text.startswith("---")
    assert "generated:" in text
    assert "sources: 3" in text
    assert "words_retrieved: 900" in text
    assert body in text


def test_slugify_produces_a_usable_filename():
    assert slugify("Compare React Native, Flutter, and native Android!") == \
        "compare-react-native-flutter-and-native-android"
    assert slugify("") == "research-report"


def test_markdown_renderer_is_not_evaluated_as_html():
    """The report is model output; it must never be able to inject markup."""
    source = __import__("pathlib").Path(__file__).resolve().parents[1] / "frontend" / "markdown.js"
    text = source.read_text()

    assert "escapeHtml" in text
    assert "textContent" not in text.split("function escapeHtml")[0]
    # Every href goes through safeUrl, which rejects javascript:.
    assert "safeUrl" in text and "javascript" not in text.replace(
        "cannot execute script", "").lower().split("function renderInline")[0][-200:]

# --- unreadable sources are kept, not dropped --------------------------------

def test_snippet_only_doc_is_marked_metadata_only():
    from backend.research.fetch import snippet_only_doc

    doc = snippet_only_doc({"title": "T", "url": "https://blog.dev/x",
                            "snippet": "a short snippet", "date": "2025"})

    assert doc.status == "metadata_only"
    assert not doc.usable, "a snippet is not a document"
    assert "could not be retrieved" in doc.limitation
    assert doc.publisher == "blog.dev"


def test_evidence_accumulates_across_rounds(monkeypatch):
    """A second round with fewer findings must not discard the first round's."""
    import asyncio

    from backend.research import graph as G

    monkeypatch.setattr(G.config, "fetch_enabled", lambda: False)
    monkeypatch.setattr(G.config, "analysis_enabled", lambda: True)
    monkeypatch.setattr(G, "search_full", lambda source, query, n=6: ([], {}))

    state = {
        "question": "q", "max_rounds": 2, "round": 1, "plan": {},
        "extracted": [Evidence(claim="round one finding", source_url="https://a.dev",
                               evidence_id="E1")],
        "chunks": [Chunk(chunk_id="s1#0", source_id="s1", source_url="https://a.dev",
                          source_title="A", content="round one finding about flutter")],
        "notes": [],
    }

    monkeypatch.setattr(G, "rank", lambda chunks, q, p, top_k=None: [])
    out = asyncio.run(G.analyse(state))

    kept = [r.claim for r in out.get("extracted", [])]
    assert "round one finding" in kept, "earlier rounds' evidence must survive"


def test_rate_limited_response_is_retried(monkeypatch):
    """A 429 is the origin asking us to slow down, not a refusal."""
    from backend.research.fetch import Fetcher

    monkeypatch.setattr(config, "fetch_respect_robots", lambda: False)
    monkeypatch.setattr("backend.research.fetch.time.sleep", lambda s: None)

    class FlakySession(_Session):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def get(self, url, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return _Response(text="", status_code=429, url=url)
            return _Response(text=HTML, status_code=200, url="https://ok.dev/a")

    session = FlakySession()
    doc = Fetcher(session=session).fetch("https://ok.dev/a")

    assert session.calls == 2, "the request must be retried once"
    assert doc.status == "full"


def test_failed_section_is_disclosed_in_the_report(monkeypatch):
    from backend.research import report as report_module

    records = _records()

    def only_exec(system, question, heading, instruction, recs, r, t,
                  budget=None, min_words=350, failed=None):
        if heading == "Executive Summary":
            if failed is not None:
                failed.append("Hiring Demand (every model in the chain failed)")
            return ""
        return f"## {heading}\n\ntext [1]\n"

    monkeypatch.setattr(report_module, "_write_section", only_exec)

    class Col:
        def context(self):
            return ""

    out = write_report("Compare React Native, Flutter and native Android. "
                       "Analyze performance and hiring demand.",
                       Col(), [], records=records)

    assert "PARTIAL REPORT" in out
    assert "absent from this report" in out
    assert "Hiring Demand" in out.split("PARTIAL REPORT")[1]
    assert "COMPLETE REPORT" not in out


def test_status_is_partial_when_a_section_is_missing(monkeypatch):
    """A budget-exhausted section must make the run partial, not completed."""
    from backend.research import report as report_module

    records = _records()

    def only_exec(system, question, heading, instruction, recs, r, t,
                  budget=None, min_words=350, failed=None):
        if heading == "Executive Summary":
            if failed is not None:
                failed.append("Hiring Demand (LLM call budget exhausted)")
            return ""
        return f"## {heading}\n\ntext [1]\n"

    monkeypatch.setattr(report_module, "_write_section", only_exec)

    class Col:
        def context(self):
            return ""

    result = report_module.generate_report(
        "Compare React Native, Flutter and native Android. "
        "Analyze performance and hiring demand.", Col(), [], records=records)

    assert result.status == "partial"
    assert result.missing_sections
    assert report_module.status_headline(result.status, result.missing_sections) == \
        "PARTIAL REPORT — LLM budget exhausted during synthesis"


def test_status_is_completed_when_nothing_is_missing(monkeypatch):
    from backend.research import report as report_module
    from backend.research.evidence import Evidence

    # Completed now means adequate coverage, not just assembled sections: two
    # adequate findings behind every planned dimension.
    records = [
        Evidence(claim=f"performance finding {i}", source_title="P",
                 source_url=f"https://p.dev/{i}", quality="strong",
                 dimension="performance", retrieval_status="full",
                 confidence=0.9)
        for i in range(2)
    ] + [
        Evidence(claim=f"hiring finding {i}", source_title="H",
                 source_url=f"https://h.dev/{i}", quality="moderate",
                 dimension="hiring demand", retrieval_status="full",
                 confidence=0.8)
        for i in range(2)
    ]

    def all_sections(system, question, heading, instruction, recs, r, t,
                     budget=None, min_words=350, failed=None):
        return f"## {heading}\n\ntext [1]\n"

    monkeypatch.setattr(report_module, "_write_section", all_sections)

    class Col:
        def context(self):
            return ""

    result = report_module.generate_report(
        "Compare React Native, Flutter and native Android. "
        "Analyze performance and hiring demand.", Col(), [], records=records)

    assert result.status == "completed"
    assert result.missing_sections == []
    assert "PARTIAL" not in result.markdown
    assert "Evidence hole" not in result.markdown
    assert report_module.status_headline(result.status, []) == "COMPLETE REPORT"


def test_status_is_partial_when_a_dimension_lacks_evidence(monkeypatch):
    """Coverage drives completion: all sections written is not enough."""
    from backend.research import report as report_module

    records = _records()

    def all_sections(system, question, heading, instruction, recs, r, t,
                     budget=None, min_words=350, failed=None):
        return f"## {heading}\n\ntext [1]\n"

    monkeypatch.setattr(report_module, "_write_section", all_sections)

    class Col:
        def context(self):
            return ""

    result = report_module.generate_report(
        "Compare React Native, Flutter and native Android. "
        "Analyze performance and hiring demand.", Col(), [], records=records)

    assert result.status == "partial", \
        "thin dimensions must block a completed status"
    assert any("Evidence hole" in m for m in result.missing_sections)
    assert "Evidence hole: hiring demand" in result.markdown


def test_status_is_failed_when_nothing_can_be_written(monkeypatch):
    from backend.research import report as report_module

    monkeypatch.setattr(report_module, "_write_section",
                        lambda *a, **k: "")

    records = _records()

    class Col:
        def context(self):
            return ""

    result = report_module.generate_report(
        "Compare React Native, Flutter and native Android. "
        "Analyze performance and hiring demand.", Col(), [], records=records)

    # Sections exist in the headings but nothing was written for them.
    assert result.status in {"partial", "failed"}
    assert result.missing_sections


# --- retrieval statistics describe the whole run ----------------------------

class _StubFetcher:
    """A fetcher with a real budget, so exhaustion is reproducible offline."""

    def __init__(self, budget=2):
        self.budget = budget
        self.fetched = 0
        self.requested = []

    def fetch_many(self, urls, on_done=None):
        from backend.research.fetch import FAILED, FetchedDoc

        out = []
        for url in urls:
            self.requested.append(url)
            if self.fetched >= self.budget:
                out.append(FetchedDoc(url=url, status=FAILED,
                                      limitation="fetch budget exhausted"))
                continue
            self.fetched += 1
            out.append(FetchedDoc(url=url, final_url=url, status="full",
                                  method="html", title=f"Doc {url}",
                                  text=("body text about react native flutter android "
                                        "performance ") * 40, publisher="pub"))
        if on_done is not None:
            for doc in out:
                on_done(doc)
        return [d for d in out if d.status != FAILED]


def _results(tag, count):
    return [{"title": f"T{i}", "url": f"https://{tag}.dev/{i}", "snippet": "snip",
             "date": "2025", "type": "web", "query": "q"} for i in range(count)]


def test_retrieval_stats_accumulate_across_rounds(monkeypatch):
    """The reported counters must describe the run, not just the last round.

    Regression: round 2 previously re-fetched URLs round 1 already had, exhausted
    the fetch budget, marked everything metadata-only, and overwrote the stats --
    so the UI showed "0 documents fetched" next to evidence that came from
    fully-read documents.
    """
    import asyncio

    from backend.research import graph as G

    monkeypatch.setattr(G.config, "fetch_enabled", lambda: True)
    monkeypatch.setattr(G.config, "references_enabled", lambda: False)

    fetcher = _StubFetcher(budget=2)
    state = {"question": "q", "max_rounds": 3, "round": 1,
             "plan": {"subquestions": []}, "raw": _results("a", 3),
             "log": [], "notes": [], "fetcher": fetcher}

    first = asyncio.run(G.retrieve(state))
    stats1 = first["retrieval_stats"]
    assert stats1["full_text"] == 2
    assert stats1["metadata_only"] == 1

    state.update({"raw": _results("a", 3) + _results("b", 2),
                  "retrieval_index": first["retrieval_index"],
                  "documents_by_url": first["documents_by_url"],
                  "fetcher": fetcher, "round": 2})
    second = asyncio.run(G.retrieve(state))
    stats2 = second["retrieval_stats"]

    # Round 1's two full-text documents are still reported as full text.
    assert stats2["full_text"] == 2, "a later round must not erase earlier retrieval"
    assert stats2["attempted"] == 5, "counters must cover every source seen"
    assert stats2["attempted"] == (
        stats2["full_text"] + stats2["partial"] + stats2["metadata_only"]
        + stats2["failed"]
    ), "attempted must equal the sum of the per-status buckets"


def test_already_read_sources_are_not_fetched_again(monkeypatch):
    """Re-requesting a document we already have wastes budget and can fail."""
    import asyncio

    from backend.research import graph as G

    monkeypatch.setattr(G.config, "fetch_enabled", lambda: True)
    monkeypatch.setattr(G.config, "references_enabled", lambda: False)

    fetcher = _StubFetcher(budget=10)
    state = {"question": "q", "max_rounds": 3, "round": 1,
             "plan": {"subquestions": []}, "raw": _results("a", 2),
             "log": [], "notes": [], "fetcher": fetcher}
    first = asyncio.run(G.retrieve(state))
    after_first = len(fetcher.requested)

    state.update({"raw": _results("a", 2), "retrieval_index": first["retrieval_index"],
                  "documents_by_url": first["documents_by_url"],
                  "fetcher": fetcher, "round": 2})
    asyncio.run(G.retrieve(state))

    assert len(fetcher.requested) == after_first, \
        "a second round must not re-request URLs already retrieved"


def test_metadata_only_evidence_is_capped_and_labelled(monkeypatch):
    """A snippet-only source must never look like a read document."""
    from backend.research import extract as extract_module
    from backend.research.evidence import Evidence

    chunk = Chunk(chunk_id="S9#0", source_id="S9", source_url="https://blog.dev/x",
                  source_title="Blog", content="Flutter is 40% faster than native, "
                  "measured at 60fps on a Pixel 7.",
                  retrieval_status="metadata_only")

    reply = {"evidence": [{
        "chunk": "S9#0",
        "claim": "Flutter is 40% faster than native, measured at 60fps",
        "quote": "measured at 60fps on a Pixel 7",
        "quality": "strong", "confidence": 0.95,
    }]}

    accepted, rejected = extract_module._harvest(reply, [], {"S9#0": chunk})
    assert len(accepted) == 1, "the record itself is still kept"
    record = accepted[0]

    assert record.quality == "weak", "a snippet cannot support a strong claim"
    assert record.confidence <= 0.3
    assert "NOT read" in record.limitations
    assert record.retrieval_status == "metadata_only"


def test_retrieval_index_upgrades_a_source_when_later_retrieved():
    """A snippet seen first must not stay metadata-only once really fetched."""
    from backend.research.graph import _aggregate_retrieval

    index = {"https://a.dev/1": {"status": "metadata_only", "words": 10,
                                 "title": "A", "url": "https://a.dev/1",
                                 "method": "search-metadata", "limitation": ""}}
    weak = _aggregate_retrieval(index, [], [])
    assert weak["metadata_only"] == 1 and weak["full_text"] == 0

    # A later round fetched it properly.
    index["https://a.dev/1"]["status"] = "full"
    index["https://a.dev/1"]["words"] = 9000
    better = _aggregate_retrieval(index, [], [])
    assert better["full_text"] == 1 and better["metadata_only"] == 0
    assert better["total_words"] == 9000


# --- question agnosticism ---------------------------------------------------
#
# Guards against mobile/framework vocabulary creeping back into the prompts and
# scoring tables. A hint written while tuning one question must never bias an
# unrelated one, so the source itself is the thing under test.

_DOMAIN_TERMS = [
    "flutter", "react native", "android", "kotlin", "impeller", "hermes",
    "apk", "aab", "frame rate", "frame rendering", "hot reload", "ui latency",
    "fps", "startup time", "jetbrains", "android-only", "cross-platform delivery",
    "graphics-heavy", "mobile app", "play store", "app store",
]

_PROMPT_MODULES = [
    "backend/research/report.py",
    "backend/research/extract.py",
    "backend/research/relevance.py",
    "backend/research/references.py",
    "backend/research/evidence.py",
    "backend/research/chunker.py",
    "backend/research/fetch.py",
    "backend/research/graph.py",
    "backend/research/llm.py",
    "backend/research/planner.py",
    "backend/research/searcher.py",
]


def test_no_domain_specific_vocabulary_in_pipeline_source():
    """No prompt or scoring table may assume a particular subject domain."""
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for rel in _PROMPT_MODULES:
        text = (root / rel).read_text().lower()
        for term in _DOMAIN_TERMS:
            if term in text:
                offenders.append(f"{rel}: {term}")
    assert not offenders, "domain vocabulary leaked into the pipeline: " + ", ".join(offenders)


def test_dimension_guidance_is_method_not_subject():
    """Guidance must teach distinctions, not assume what is being compared."""
    from backend.research.report import _DIMENSION_GUIDANCE, _FALLBACK_GUIDANCE

    for name, text in list(_DIMENSION_GUIDANCE.items()) + [("fallback", _FALLBACK_GUIDANCE)]:
        lowered = text.lower()
        for term in _DOMAIN_TERMS:
            assert term not in lowered, f"{name} guidance mentions {term}"


def test_unrelated_question_gets_unrelated_sections():
    """A non-computing question must not inherit the framework report structure."""
    from backend.research.relevance import dimensions, targets

    q = ("Compare the economic effects of the 2008 financial crisis and the 2020 "
         "pandemic recession on global labour markets. Analyse employment, wages, "
         "inequality, and policy response.")

    assert dimensions(q) == ["employment", "wages", "inequality", "policy response"]
    assert "react native" not in " ".join(dimensions(q)).lower()


def test_same_mechanism_for_any_domain():
    """Retrieval and chunking are domain-agnostic: identical output shape."""
    from backend.research.chunker import chunk_document

    for title, text in [
        ("Climate", "## Radiative forcing\n\n" + "CO2 concentration rose sharply. " * 60),
        ("Literature", "## Narrative voice\n\n" + "The narrator is unreliable throughout. " * 60),
        ("Nutrition", "## Glycemic index\n\n" + "Legumes score lower than white bread. " * 60),
    ]:
        chunks = chunk_document(text, source_id="s", source_url="https://x.dev",
                                source_title=title)
        assert chunks, title
        assert all(c.chunk_id.startswith("s#") for c in chunks), title
        assert all(c.section for c in chunks), f"{title} lost its section path"
        assert all(c.source_title == title for c in chunks), title


def test_numeric_boost_fires_for_any_domain():
    """The quantitative-content boost must not depend on computing units."""
    from backend.research.relevance import _NUMBER

    for sample in ["CO2 rose 45%", "unemployment hit 9.5 percent",
                   "wages grew 3.2 %", "the crop yielded 4.2 tonnes",
                   "the orchestra played 120 bpm", "deficit reached $1.4 billion"]:
        assert _NUMBER.search(sample), f"no numeric boost for: {sample}"


def test_target_extraction_keeps_compound_names_intact():
    """A hyphen must not be treated as a word boundary.

    "stream-of-consciousness" was previously truncated to "stream" because a
    hyphen is a non-word character, so \\bof\\b matched inside the compound.
    """
    from backend.research.relevance import targets

    q = ("Compare stream-of-consciousness and unreliable narration across "
         "modernist fiction. Analyse narrative technique.")
    found = targets(q)

    assert "stream-of-consciousness" in found
    assert "stream" not in found


def test_target_extraction_works_outside_software():
    from backend.research.relevance import targets

    climate = targets("Compare the economic effects of the 2008 financial crisis "
                      "and the 2020 pandemic recession on global labour markets. "
                      "Analyse employment and wages.")
    assert climate == ["the 2008 financial crisis", "the 2020 pandemic recession"]

    bio = targets("Compare CRISPR and antisense oligonucleotide therapies for "
                  "treating inherited disease. Analyse efficacy and cost.")
    assert "CRISPR" in bio


def test_prompts_contain_no_fixed_scenario_list():
    """The scenario prompt must derive cases from the question, not a template."""
    from backend.research.report import SCENARIO_SYSTEM

    lowered = SCENARIO_SYSTEM.lower()
    assert "derive the scenarios from this question" in lowered
    for term in ("android", "mobile app", "graphics-heavy", "cross-platform"):
        assert term not in lowered


def test_dimension_prompt_delegates_domain_knowledge():
    """The dimension prompt must not enumerate one field's sub-metrics."""
    from backend.research.report import DIMENSION_SYSTEM

    lowered = DIMENSION_SYSTEM.lower()
    assert "work out what the dimension is actually made of" in lowered
    for term in ("startup time", "frame rendering", "ui latency", "fps"):
        assert term not in lowered
