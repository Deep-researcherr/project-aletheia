"""The structured evidence layer.

A search snippet and a fully-parsed benchmark paper are not the same kind of
evidence, and the final report must not blur them. Every finding the pipeline
produces is carried as an :class:`Evidence` record that keeps the whole chain
intact::

    claim -> evidence record -> source -> URL -> section / page / chunk

Records also carry how much of the source we actually managed to read, and how
strong the evidence is. The report generator is told to surface both instead of
smoothing them away.
"""
from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field, asdict

# Ordered strongest first. Used for ranking and for the quality legend.
QUALITY_ORDER = ("strong", "moderate", "weak", "anecdotal", "speculative")
_QUALITY_RANK = {q: i for i, q in enumerate(QUALITY_ORDER)}

_SOURCE_TYPE_LABELS = {
    "pdf": "PDF document",
    "arxiv": "arXiv preprint",
    "github": "GitHub repository",
    "patent": "patent",
    "gutenberg": "Project Gutenberg ebook",
    "springer": "Springer article",
    "pmc": "PubMed Central full text",
    "plain": "plain text / code",
    "html": "web page",
    "none": "unknown",
}

# Legend text for every registered provider comes from the registry, so adding
# a provider there puts it in the report without a second edit here. Keys are
# the search source name and the fetch method name, which is why a provider
# with both roles carries two distinct strings.
try:
    from .providers import for_role as _for_role

    for _provider in _for_role("search"):
        if _provider.source_label:
            _SOURCE_TYPE_LABELS.setdefault(_provider.source, _provider.source_label)
    for _provider in _for_role("extract"):
        if _provider.method_label:
            _SOURCE_TYPE_LABELS.setdefault(_provider.method_name,
                                           _provider.method_label)
except Exception:                      # noqa: BLE001 - legend text is not load-bearing
    pass


@dataclass
class Evidence:
    """One traceable finding."""

    evidence_id: str = ""
    claim: str = ""                     # the finding, stated as a checkable claim
    detail: str = ""                    # specifics: numbers, conditions, caveats
    quote: str = ""                     # verbatim span from the source passage
    dimension: str = ""                 # which analysis dimension it serves
    targets: list[str] = field(default_factory=list)   # which compared options

    source_id: str = ""
    source_title: str = ""
    source_url: str = ""
    source_type: str = "web"
    publisher: str = ""
    authors: list[str] = field(default_factory=list)
    publication_date: str = ""
    doi: str = ""

    section: str = ""
    page: int = 0
    chunk_id: str = ""
    char_start: int = 0

    retrieval_date: str = ""
    retrieval_status: str = "full"      # full | partial | metadata_only
    retrieval_limitation: str = ""

    quality: str = "moderate"           # see QUALITY_ORDER
    limitations: str = ""
    confidence: float = 0.5
    citation_count: int = 0           # Scholar cited-by total, when known

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def rank(self) -> int:
        return _QUALITY_RANK.get(self.quality, len(QUALITY_ORDER))

    @property
    def usable_url(self) -> str:
        """Canonical, user-inspectable link. Never an internal id alone."""
        return self.doi or self.source_url

    @property
    def citation(self) -> str:
        """Reference-list entry."""
        who = self.publisher or (self.authors[0] if self.authors else "")
        bits = [self.source_title or self.source_url]
        if who:
            bits.append(who)
        if self.publication_date:
            bits.append(self.publication_date[:10])
        bits.append(self.source_type)
        return " — ".join(bits)

    def locator(self) -> str:
        """Where inside the document this came from."""
        bits = []
        if self.section:
            bits.append(self.section)
        if self.page:
            bits.append(f"page {self.page}")
        if not bits:
            bits.append(self.chunk_id)
        return ", ".join(bits)

    def to_prompt(self, index: int | None = None) -> str:
        """Compact, LLM-friendly rendering used inside prompts."""
        tag = f"[{index}]" if index is not None else f"[{self.evidence_id}]"
        lines = [f"{tag} CLAIM: {self.claim}"]
        if self.detail:
            lines.append(f"    DETAIL: {self.detail}")
        lines.append(f"    SOURCE: {self.citation}")
        lines.append(f"    URL: {self.usable_url}")
        loc = self.locator()
        if loc:
            lines.append(f"    LOCATION: {loc}")
        if self.quote:
            lines.append(f"    QUOTE: \"{self.quote[:300]}\"")
        lines.append(f"    QUALITY: {self.quality} | RETRIEVAL: {self.retrieval_status}"
                     + (f" ({self.retrieval_limitation})" if self.retrieval_limitation else ""))
        if self.citation_count:
            lines.append(f"    CITED BY: {self.citation_count} later works")
        if self.limitations:
            lines.append(f"    LIMITATIONS: {self.limitations}")
        return "\n".join(lines)


def from_chunk(chunk, index: int) -> Evidence:
    """Build a metadata-only record for a source we could not read in full.

    Used when retrieval produced a snippet but no document: the source is still
    worth citing, but it is explicitly marked as not having been read.
    """
    return Evidence(
        evidence_id=f"E{index}",
        claim=(chunk.content or "").strip()[:400],
        source_id=chunk.source_id,
        source_title=chunk.source_title,
        source_url=chunk.source_url,
        source_type=chunk.source_type,
        publisher=chunk.source_publisher,
        authors=list(chunk.authors or []),
        publication_date=chunk.publication_date,
        doi=chunk.doi,
        section=chunk.section,
        page=chunk.page,
        chunk_id=chunk.chunk_id,
        char_start=chunk.char_start,
        retrieval_date=datetime.date.today().isoformat(),
        retrieval_status="metadata_only",
        retrieval_limitation="only the search-result snippet was available; "
                             "the document itself was not retrieved",
        quality="weak",
        confidence=0.2,
        citation_count=int(getattr(chunk, "cited_by", 0) or 0),
    )


def normalise_quality(value: str) -> str:
    value = (value or "").strip().lower()
    aliases = {
        "high": "strong", "medium": "moderate", "low": "weak",
        "primary": "strong", "peer-reviewed": "strong", "official": "strong",
        "benchmark": "moderate", "industry report": "moderate",
        "journalism": "weak", "blog": "weak", "opinion": "anecdotal",
        "forum": "anecdotal", "speculation": "speculative", "hypothesis": "speculative",
    }
    value = aliases.get(value, value)
    return value if value in _QUALITY_RANK else "moderate"


def normalise_confidence(value, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number > 1:                       # a 0-100 scale is common in model output
        number = number / 100
    return max(0.0, min(1.0, number))


def source_type_label(source_type: str) -> str:
    """Human label for a search result type or a fetch method.

    Falls back to "web page" rather than raising: this is called while writing
    a report, and a report must not fail to print because one document carried
    a type string nothing has heard of.
    """
    return _SOURCE_TYPE_LABELS.get((source_type or "").lower(), "web page")


_WS = re.compile(r"\s+")


def quote_is_supported(quote: str, chunk_text: str, threshold: float = 0.72) -> bool:
    """Is this quote actually in the passage?

    Cheap guard against the model inventing a citation. Compares normalised word
    sequences rather than exact substrings so light reformatting still passes,
    while a genuinely fabricated quote does not.
    """
    if not quote or not chunk_text:
        return False
    haystack = _WS.sub(" ", chunk_text.lower()).strip()
    needle = _WS.sub(" ", quote.lower()).strip()
    if not needle:
        return False
    if needle in haystack:
        return True

    quote_words = needle.split()
    if len(quote_words) < 4:
        return False
    window = len(quote_words)
    hay_words = haystack.split()
    if len(hay_words) <= window:
        return False

    # Best sliding-window word overlap over the passage.
    best = 0.0
    step = max(1, window // 4)
    for start in range(0, len(hay_words) - window + 1, step):
        segment = set(hay_words[start:start + window])
        overlap = len(segment & set(quote_words))
        best = max(best, overlap / len(set(quote_words)))
        if best >= threshold:
            return True
    return best >= threshold


def dedupe(records: list[Evidence]) -> list[Evidence]:
    """Drop near-duplicate claims, keeping the best-attested one."""
    best: dict[str, Evidence] = {}
    for record in records:
        # Key on the FULL normalized claim, not a truncated prefix: two
        # findings that share a long lead-in but differ in their trailing
        # numbers or conditions are different claims, and a length cap
        # silently merged them.
        key = re.sub(r"[^a-z0-9]+", " ", (record.claim or "").lower()).strip()
        if not key:
            continue
        incumbent = best.get(key)
        if incumbent is None or _better(record, incumbent):
            best[key] = record
    return sorted(best.values(),
                  key=lambda r: (r.rank, -r.confidence, r.evidence_id))


def _better(candidate: Evidence, incumbent: Evidence) -> bool:
    if candidate.rank != incumbent.rank:
        return candidate.rank < incumbent.rank
    if candidate.retrieval_status != incumbent.retrieval_status:
        return candidate.retrieval_status == "full"
    return candidate.confidence > incumbent.confidence


def statistics(records: list[Evidence]) -> dict:
    """Counts the report and the UI use to describe the evidence base honestly."""
    by_quality: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_type: dict[str, int] = {}
    by_dimension: dict[str, int] = {}
    sources, dated, quantified = set(), 0, 0

    for record in records:
        by_quality[record.quality] = by_quality.get(record.quality, 0) + 1
        by_status[record.retrieval_status] = by_status.get(record.retrieval_status, 0) + 1
        label = source_type_label(record.source_type)
        by_type[label] = by_type.get(label, 0) + 1
        if record.dimension:
            by_dimension[record.dimension] = by_dimension.get(record.dimension, 0) + 1
        if record.source_url:
            sources.add(record.source_url)
        if record.publication_date:
            dated += 1
        if record.quote and re.search(r"\d", record.quote):
            quantified += 1

    return {
        "records": len(records),
        "sources": len(sources),
        "dated": dated,
        "quantified": quantified,
        "by_quality": by_quality,
        "by_retrieval": by_status,
        "by_source_type": by_type,
        "by_dimension": by_dimension,
    }