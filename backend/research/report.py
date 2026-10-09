"""Deep report generation.

The old generator handed one model call a wall of snippets and asked for a
report. That compressed everything, dropped most citations, and made it
impossible to tell which finding came from where.

The generator here is section-wise. Each section is written by its own call,
given only the evidence relevant to it, and citing from a fixed reference
numbering that is assigned deterministically before any model runs. That last
part is what makes traceability structural rather than aspirational: a citation
in the prose can always be resolved to a real document URL, because the
reference list was built from the evidence records themselves and the model
cannot invent a number that does not exist there.

Sections are chosen deterministically from the dimensions the user asked about,
not by a model, so the report shape is predictable and the call budget holds.
"""
from __future__ import annotations

import datetime
import logging
import re

from . import config
from .evidence import (Evidence, QUALITY_ORDER, source_type_label, statistics)
from .extract import INTEGRITY_PREFIX, integrity_notes
from .llm import GROUNDING, LLMChainError, ask
from .output_pool import OutputMemoryPool
from .relevance import dimensions as question_dimensions
from .relevance import target_variant_mismatch
from .relevance import targets as question_targets

log = logging.getLogger(__name__)

# The rules that apply to every section. Repeated per call because each section
# is generated independently and would otherwise drift.
DISCIPLINE = f"""{GROUNDING}

CITATION RULES (these are not optional):
- Cite with the bracketed reference numbers given to you, e.g. [3] or [3][7].
- Place the citation immediately after the specific claim it supports, not at
  the end of a paragraph containing several unrelated claims.
- Cite every number, date, price, measurement and comparative claim.
- Use ONLY reference numbers that appear in your evidence list. Never invent one.
- If the evidence does not support a claim, do not make the claim. Write
  "the evidence gathered does not establish this" instead.
- Never fabricate a statistic, benchmark result, paper finding, author or URL.

EVIDENCE HONESTY RULES:
- Distinguish what a source measured from what someone concluded about it.
- State the conditions behind a number: device, build type, workload, date,
  sample size, methodology. If the evidence does not say, say it does not say.
- A vendor's claim about its own product is an interested party, not an
  independent measurement. Label it as such.
- One benchmark is not a general ranking. Say what the benchmark covered.
- When sources disagree, present both positions and say whether the evidence
  resolves the disagreement. If it does not, say so plainly.
- Never turn limited evidence into a universal conclusion.
- Say "the evidence is insufficient on this point" when it is. That is a
  legitimate and valuable finding, not a failure.

STYLE RULES:
- Write in continuous prose under the given heading. Do not use bullet lists
  for analysis; use them only for compact enumerations of scenarios.
- Every paragraph must add evidence, comparison or reasoning. Never restate a
  conclusion, pad, or paraphrase the previous paragraph.
- Be specific and technical. Name the frameworks, mechanisms, versions and
  measurements involved.
- Do not open with filler such as "In today's fast-moving world".
- Aim for the requested depth. Length must come from evidence, not repetition."""

EXEC_SYSTEM = f"""You write the executive summary of a research report.

Summarise the findings that the detailed sections below establish. Lead with
the substantive answer, then the principal trade-offs, then what remains
uncertain. Do not introduce any fact, number or claim that does not appear in
the evidence given to you.

{DISCIPLINE}"""

METHOD_SYSTEM = f"""You write the Scope and Methodology section of a research report.

State plainly and concretely: what was investigated, which options were
compared and how they were defined, the date range and recency of the evidence,
how sources were selected, how conflicting evidence was handled, and what the
method cannot establish. Be specific about the actual limitations of this
research run -- which sources were only partially retrievable, which questions
got thin coverage, and what a reader should therefore be cautious about.

Do not claim a comprehensiveness the run did not have.

{DISCIPLINE}"""

DIMENSION_SYSTEM = f"""You write one analytical section of a research report, covering a
single dimension of a comparison.

Structure the section as analysis, not as a list:
- Open by stating what this dimension actually means and why it is contested or
  trade-off shaped, where that applies.
- Analyse each option being compared separately and concretely.
- Compare them directly against each other.
- Present concrete evidence with its conditions attached.
- Explain the caveats, the methodology behind each measurement, and the
  situations in which the ranking would change.
- Close with what the evidence does and does not settle about this dimension.

Where the dimension involves measurement, treat it as multidimensional rather
than as one headline number. Decide for yourself which sub-metrics matter in
this domain -- work out what the dimension is actually made of, then say which
components you can evidence and which you cannot. Wherever a figure appears,
state what was measured, on what, under what conditions, with what method, and
whether it generalises beyond the case that was measured.

Where the dimension involves cost, separate the one-off cost from the recurring
cost, and separate the components of each rather than quoting a single total.

If evidence for part of the dimension is thin or missing, analyse what you have
and state plainly what is missing.

{DISCIPLINE}"""

CONFLICT_SYSTEM = f"""You write the section reconciling conflicting evidence.

For each disagreement present both positions, name the sources, then compare
them on everything that could explain the gap: methodology, date, population or
sample, scale, configuration, definitions, and the incentives of whoever
produced each result. Offer the most likely reasons for the disagreement. Say
which result is more generalisable and why, or state that the evidence does not
resolve it.

Do not silently prefer the position that fits a tidy conclusion.

{DISCIPLINE}"""

SCENARIO_SYSTEM = f"""You write the scenario analysis section of a research report.

Derive the scenarios from THIS question rather than from any fixed template.
Identify the options being compared and the constraints that actually vary
between real cases in this domain, then write scenarios that differ along those
axes -- for example scale and resources, degree of specialisation, sensitivity
to a particular risk, the audience or population affected, or the time horizon
over which the decision plays out. Ignore any example that does not fit.

For each scenario, state what the constraints imply for the trade-offs that
decide the outcome, give a reasoned recommendation grounded in cited evidence,
and name the condition that would flip it. Do not produce a universal ranking.

{DISCIPLINE}"""

DECISION_SYSTEM = f"""You write the decision framework section of a research report.

Do not tell the reader which option to choose. Instead give them the reasoning
they need to choose: what to prioritise under each constraint, which trade-offs
become decisive under which condition, how existing team expertise changes the
economics, and what the reversal conditions are.

Organise this as conditional guidance -- "if X matters most, then Y" -- tied to
the evidence. Make clear what is a judgement rather than a finding.

{DISCIPLINE}"""

QUALITY_SYSTEM = f"""You write the evidence quality and limitations section.

State what the evidence base actually consisted of: how many sources, of what
types, how recent, how much was fully retrievable versus snippet-only, and how
that shapes confidence. Explain the quality tiers in use and why findings were
placed in them. Name the specific weaknesses of this research run and what a
reader should not conclude from it.

Be candid. This section exists to stop over-reading the rest of the report.

{DISCIPLINE}"""

CONCLUSION_SYSTEM = f"""You write the conclusion of a research report.

Synthesise the strongest evidence into a balanced position. Separate what the
evidence establishes from what is interpretation. Name the genuinely uncertain
areas and the trade-offs that do not resolve. Give no universal verdict the
evidence does not support. Close with what would change the answer.

{DISCIPLINE}"""


# ---------------------------------------------------------------------------
# Reference numbering (deterministic -- no model involved)
# ---------------------------------------------------------------------------

def assign_references(records: list[Evidence]) -> dict[str, int]:
    """Map each distinct source URL to a stable reference number.

    Numbered in order of evidence strength so that the strongest sources tend to
    carry the low, easy-to-cite numbers. Because the model can only cite numbers
    that exist here, a fabricated citation is structurally impossible.
    """
    order: dict[str, int] = {}
    best: dict[str, tuple] = {}

    for record in records:
        url = record.usable_url
        if not url:
            continue
        rank = (record.rank, -record.confidence, -record.citation_count,
                record.evidence_id)
        if url not in best or rank < best[url]:
            best[url] = rank
            order.setdefault(url, None)

    for number, url in enumerate(sorted(best, key=lambda u: best[u]), 1):
        order[url] = number
    return {k: v for k, v in order.items() if v}


def render_references(records: list[Evidence], refs: dict[str, int]) -> str:
    """Build the references section. Purely deterministic."""
    grouped: dict[int, list[Evidence]] = {}
    for record in records:
        number = refs.get(record.usable_url)
        if number:
            grouped.setdefault(number, []).append(record)
    if not grouped:
        return ""

    lines = ["## References", ""]
    for number in sorted(grouped):
        group = grouped[number]
        head = group[0]
        title = head.source_title or head.source_url
        bits = []
        if head.authors:
            bits.append(", ".join(head.authors[:3]) + (" et al." if len(head.authors) > 3 else ""))
        if head.publisher and head.publisher != (head.authors[0] if head.authors else ""):
            bits.append(head.publisher)
        if head.publication_date:
            bits.append(head.publication_date[:10])
        bits.append(source_type_label(head.source_type))
        lines.append(f"{number}. **{title}** — {'; '.join(b for b in bits if b)}.  ")
        lines.append(f"   <{head.usable_url}>")
        locations = sorted({r.locator() for r in group if r.section or r.page})
        if locations:
            lines.append(f"   Cited from: {', '.join(locations[:4])}")
        statuses = {r.retrieval_status for r in group}
        if "metadata_only" in statuses:
            lines.append("   *Note: only the search-result metadata was retrievable "
                         "for this source; the document itself was not read.*")
        elif statuses == {"partial"}:
            lines.append("   *Note: this source was only partially retrievable.*")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _evidence_block(records: list[Evidence], refs: dict[str, int],
                    limit: int) -> str:
    lines = []
    for record in records[:limit]:
        number = refs.get(record.usable_url)
        if not number:
            continue
        lines.append(f"[{number}] {record.claim}")
        if record.detail:
            lines.append(f"    detail: {record.detail}")
        lines.append(f"    source: {record.source_title or record.source_url} "
                     f"({source_type_label(record.source_type)}, "
                     f"{record.publication_date or 'undated'}, "
                     f"{record.publisher or 'publisher unknown'})")
        if record.section or record.page:
            lines.append(f"    location: {record.locator()}")
        if record.quote:
            lines.append(f"    quote: \"{record.quote[:260]}\"")
        lines.append(f"    evidence strength: {record.quality}"
                     + (f" | retrieval: {record.retrieval_status}" if record.retrieval_status != "full" else ""))
        if record.citation_count:
            lines.append(f"    cited by: {record.citation_count} later works")
        if record.limitations:
            lines.append(f"    limitations: {record.limitations}")
    return "\n".join(lines)


def _select_for_dimension(records: list[Evidence], dimension: str,
                         targets: list[str]) -> list[Evidence]:
    """Evidence most relevant to one dimension: tagged for it, or about its targets."""
    words = {w.lower() for w in re.findall(r"[a-z]+", dimension.lower()) if len(w) > 3}
    scored: list[tuple[float, Evidence]] = []
    for record in records:
        score = 0.0
        if record.dimension and record.dimension.lower().strip() == dimension.lower().strip():
            score += 3.0
        elif record.dimension and words and \
                words & {w.lower() for w in re.findall(r"[a-z]+", record.dimension.lower())}:
            score += 1.5
        if targets and record.targets:
            overlap = {t.lower() for t in record.targets} & {t.lower() for t in targets}
            score += 0.8 * len(overlap)
        if targets and target_variant_mismatch(record.targets, record.claim, targets):
            # Related-variant evidence ranks below directly applicable evidence
            # for the same dimension instead of competing with it equally.
            score -= 1.5
        claim_words = set(re.findall(r"[a-z]+", (record.claim + " " + record.detail).lower()))
        score += 0.6 * len(words & claim_words)
        score += (len(QUALITY_ORDER) - record.rank) * 0.15
        scored.append((score, record))
    scored.sort(key=lambda pair: -pair[0])
    return [r for _, r in scored]


# ---------------------------------------------------------------------------
# Section generation
# ---------------------------------------------------------------------------

def _write_section(system: str, question: str, heading: str, instruction: str,
                   records: list[Evidence], refs: dict[str, int],
                   targets: list[str], budget=None,
                   min_words: int = 350,
                   failed: list[str] | None = None) -> str:
    if not records:
        if failed is not None:
            failed.append(f"{heading} (no evidence was selected for this section)")
        return ""
    if budget is not None and not budget.take(1):
        log.info("budget exhausted before writing section %r", heading)
        if failed is not None:
            failed.append(f"{heading} (LLM call budget exhausted)")
        return ""
    try:
        block = _evidence_block(records, refs, config.report_evidence_per_section())
    except Exception:
        log.warning("could not render evidence for %r", heading, exc_info=True)
        if failed is not None:
            failed.append(f"{heading} (no usable evidence was selected)")
        return ""
    if not block.strip():
        return ""

    user = (
        f"RESEARCH QUESTION: {question}\n"
        + (f"OPTIONS COMPARED: {', '.join(targets)}\n" if targets else "")
        + f"\nTODAY'S DATE: {datetime.date.today().isoformat()}\n"
        + f"\nEVIDENCE AVAILABLE FOR THIS SECTION (cite these numbers):\n{block}\n\n"
        f"Write the section titled \"{heading}\".\n"
        f"{instruction}\n"
        f"Target roughly {min_words}-{int(min_words * 1.8)} words. "
        f"Return markdown starting with the heading line '## {heading}'. No preamble."
    )
    try:
        return ask(system, user, role="synth").strip()
    except LLMChainError as exc:
        log.warning("section %r failed: %s", heading, exc)
        if failed is not None:
            failed.append(f"{heading} (every model in the chain failed)")
        return ""
    except Exception as exc:
        log.warning("section %r errored: %s", heading, exc)
        if failed is not None:
            failed.append(f"{heading} ({type(exc).__name__})")
        return ""


# Analytic guidance keyed on a dimension word. Deliberately about METHOD rather
# than domain content: it teaches the model which distinctions matter without
# assuming what the question is about. Anything named here that would only make
# sense for one domain belongs in the question, not in the code.
_DIMENSION_GUIDANCE = {
    "performance": ("Treat as multidimensional, not as one headline number. Work out "
                    "what the dimension is actually made of in this domain, then say "
                    "which components you can evidence and which you cannot. For every "
                    "figure state what was measured, on what, under what conditions, with "
                    "what method, and whether it generalises beyond the measured case."),
    "developer productivity": ("Separate the speed of producing a first version from the "
                               "speed of iterating on it, and both from the quality of the "
                               "result over time. Cover the feedback loop, the toolchain, "
                               "debugging and verification, and the cost of bringing a new "
                               "person up to speed."),
    "ecosystem maturity": ("Separate size from quality from activity -- they are not "
                           "interchangeable. Cover first-party support, third-party breadth, "
                           "documentation, maintenance and abandonment risk, stability of "
                           "releases, upgrade and migration risk, and indicators of "
                           "long-term viability."),
    "ai tooling": ("Separate documented capability from observed adoption from analysis "
                   "from speculation. State plainly whether any effect on outcomes is "
                   "measured or merely inferred, and never present a prediction as a "
                   "finding."),
    "app size": ("Establish the measurement basis before comparing any figure: transfer "
                 "size versus installed size, compressed versus uncompressed, minimal "
                 "example versus realistic artifact, per-platform versus aggregate, and "
                 "which build variant was measured. Explain why apparently contradictory "
                 "figures can both be correct."),
    "maintenance costs": ("Separate the one-off cost from the recurring cost, and break "
                          "each into components rather than quoting a single total. Name "
                          "the assumptions behind every figure and refuse to present a "
                          "generic industry estimate as a precise cost."),
    "hiring demand": ("Distinguish raw posting volume, share of postings, talent-pool size, "
                      "hiring difficulty and compensation -- these are different claims "
                      "needing different data. Name the evidence behind any statement "
                      "rather than asserting that one option simply has more demand."),
}
_FALLBACK_GUIDANCE = (
    "Analyse this dimension thoroughly: analyse each option separately, compare them "
    "directly, and state what the evidence does and does not settle. Work out what this "
    "dimension means in the context of the question rather than assuming a fixed reading."
)


class ReportResult:
    """A generated report plus an honest statement of how complete it is.

    A report that lost sections to an exhausted budget is not a completed
    research run, and must not be presented as one. ``status`` is one of:

    ``completed``  every planned section was written
    ``partial``    some planned sections are missing; ``missing_sections`` says which
    ``failed``     no usable report could be produced

    ``sections`` holds the finished (heading, markdown) pairs in document
    order, mirroring the per-run output pool, so the stream can emit progress
    without re-reading the assembled document.
    """

    __slots__ = ("markdown", "status", "missing_sections", "notes", "sections")

    def __init__(self, markdown: str, status: str,
                 missing_sections: list[str] | None = None,
                 notes: list[str] | None = None,
                 sections: list[tuple[str, str]] | None = None):
        self.markdown = markdown
        self.status = status
        self.missing_sections = missing_sections or []
        self.notes = notes or []
        self.sections = sections or []

    def __bool__(self) -> bool:
        return self.status != "failed"

    def as_dict(self) -> dict:
        return {"status": self.status,
                "missing_sections": list(self.missing_sections),
                "notes": list(self.notes)}


def classify_status(markdown: str, missing: list[str],
                    planned: int, written: int) -> str:
    """Decide completed / partial / failed from what actually got written."""
    if not markdown.strip():
        return "failed"
    if missing:
        return "partial"
    if planned and written < planned:
        return "partial"
    return "completed"


STATUS_LABELS = {
    "completed": "COMPLETE REPORT",
    "partial": "PARTIAL REPORT",
    "failed": "REPORT FAILED",
}


def _dim_hit(record, dimension: str) -> bool:
    """Whether this record actually evidences the dimension (not just mentions)."""
    tag = (record.dimension or "").lower().strip()
    name = dimension.lower().strip()
    if tag and tag == name:
        return True
    words = {w.lower() for w in re.findall(r"[a-z]+", name) if len(w) > 3}
    if not words:
        return False
    if tag and words & {w.lower() for w in re.findall(r"[a-z]+", tag)}:
        return True
    claim_words = set(re.findall(r"[a-z]+", (record.claim + " " + record.detail).lower()))
    return bool(words & claim_words)


def evidence_holes(records, dims: list[str], needed: int | None = None) -> list[str]:
    """Dimensions without enough adequate evidence to support a section.

    A dimension counts as covered only with ``needed`` strong/moderate
    findings behind it (2 for deep reports, 1 for brief). Mentioning a
    dimension is not covering it. Returned strings are ready for status
    lists and gap-check prompts alike.
    """
    need = needed if needed is not None else (1 if config.report_depth() == "brief" else 2)
    holes = []
    for dim in dims or []:
        count = sum(1 for r in records or []
                    if r.quality in ("strong", "moderate") and _dim_hit(r, dim))
        if count < need:
            holes.append(f"Evidence hole: {dim} "
                         f"(only {count} adequate finding(s), need {need})")
    return holes


def status_headline(status: str, missing: list[str]) -> str:
    """A single line the UI and the report both show."""
    if status == "completed":
        return STATUS_LABELS["completed"]
    if status == "failed":
        return STATUS_LABELS["failed"]
    reason = ""
    if any("budget" in m for m in missing):
        reason = " — LLM budget exhausted during synthesis"
    elif missing:
        reason = " — " + "; ".join(sorted(set(missing)))
    return STATUS_LABELS["partial"] + reason


def partial_banner(status: str, missing: list[str]) -> str:
    """The visible warning block placed in the markdown itself.

    The exported file has to carry this too: a report saved to disk that omits
    half its sections must not read as finished when reopened later.

    Two different things end a run short, and conflating them misleads: a
    missing section was never written, whereas an evidence-integrity failure
    means the written sections rest on less evidence than they appear to. The
    banner says which one happened rather than claiming sections are absent
    when they are not.
    """
    if status == "completed":
        return ""
    if status == "failed":
        return ("> **" + STATUS_LABELS["failed"] + "**\n>\n"
                "> No usable report could be generated for this question.\n")

    integrity = [str(m)[len(INTEGRITY_PREFIX):] for m in missing
                 if str(m).startswith(INTEGRITY_PREFIX)]
    sections = [str(m) for m in missing if not str(m).startswith(INTEGRITY_PREFIX)]

    lines = ["> **" + status_headline(status, missing) + "**", ">"]
    if sections:
        lines.append("> These planned sections could not be written and are "
                     "absent from this report: "
                     + "; ".join(sorted(set(sections))) + ".")
    if integrity:
        lines.append("> Every section below was written and cited normally, but "
                     "the evidence behind them was degraded: "
                     + "; ".join(sorted(set(integrity))) + ".")
    return "\n".join(lines) + "\n"


def generate_report(question, collector, contradictions,
                    records: list[Evidence] | None = None,
                    source_synth: dict | None = None,
                    notes: list[str] | None = None,
                    plan: dict | None = None,
                    budget=None) -> ReportResult:
    """Generate the report and report how complete it is."""
    records = records or []
    if not records:
        # No structured evidence means the deep section-wise path never ran and
        # this is a snippet-based summary instead. Saying "completed" here is
        # exactly the false assurance this state exists to prevent.
        text = _write_report_legacy(question, collector, contradictions)
        reason = ("no structured evidence was available, so this is a "
                  "snippet-based summary rather than a deep synthesis")
        status = classify_status(text, [reason], 4, 4)
        return ReportResult(partial_banner(status, [reason]) + text, status,
                            [reason], notes=list(notes or []))

    refs = assign_references(records)
    if not refs:
        text = _write_report_legacy(question, collector, contradictions)
        reason = ("evidence carried no usable source URLs, so no references "
                  "could be cited")
        status = classify_status(text, [reason], 4, 4)
        return ReportResult(partial_banner(status, [reason]) + text, status,
                            [reason], notes=list(notes or []))

    today = datetime.date.today().isoformat()
    dims = question_dimensions(question)
    if not dims and plan:
        dims = [sq.get("question", "")
                for sq in plan.get("subquestions", []) if sq.get("question")]
    targets = question_targets(question)
    stats = statistics(records)
    depth = config.report_depth()
    is_deep = depth != "brief"

    failed_sections: list[str] = []
    planned = 0
    written = 0

    parts: list[str] = [
        f"# {question.strip().rstrip('?')}",
        "",
        f"*Research synthesis generated {today}. "
        f"{stats['sources']} distinct sources, {stats['records']} traceable findings.*",
        "",
    ]

    cap = config.report_max_sections() if is_deep else 4
    dims = dims[:cap]

    # Coverage holes are computed from the final evidence set: a dimension
    # mentioned in the report must have adequate findings behind it, or the
    # gap is named explicitly instead of polished over.
    holes = evidence_holes(records, dims)

    # Per-run output memory pool: every finished section is stored verbatim
    # as it completes. A fallback retrying a failed section reads this pool
    # (read-only copies) so it continues the report instead of restarting it,
    # and completed entries are never modified by a retry.
    pool = OutputMemoryPool(question)
    # Specs of sections whose model call failed and may be resumed from pool.
    pending_resume: dict[str, tuple] = {}

    def _pool_grounding(exclude: str = "") -> str:
        digest = pool.context(config.output_pool_context_chars(), exclude=exclude)
        if not digest:
            return ""
        return ("\n\nSECTIONS ALREADY WRITTEN (read-only context: do not restate, "
                "rewrite or alter them; write only the requested section so it "
                "continues the report):\n" + digest)

    def compose(system, heading, instruction, recs, words, ground: bool = False):
        """Write one section and guarantee the missing-section list stays honest.

        The writer records its own failure reason, but the invariant is enforced
        here as well: if a section produced nothing and nobody said why, the
        report is still marked partial with a stated reason.

        Successful sections are stored verbatim in the output pool; model
        failures are stashed for a single pool-grounded resume pass.

        Section headings pass through unaltered: no length gate is applied,
        so full technical headings reach the document canvas intact.
        """
        if ground:
            instruction = instruction + _pool_grounding(exclude=heading)
        before = len(failed_sections)
        section = _write_section(system, question, heading, instruction, recs, refs,
                                 targets, budget, min_words=words,
                                 failed=failed_sections)
        if section:
            pool.complete(heading, section)
            return section
        if len(failed_sections) == before:
            failed_sections.append(f"{heading} (section produced no output)")
        reason = failed_sections[-1] if failed_sections else ""
        if _resumable_failure(reason):
            pending_resume.setdefault(heading, (system, instruction, recs, words))
            pool.fail(heading, reason)
        return ""

    def resume_from_pool() -> None:
        """Retry model-failed sections once, grounded in completed output.

        Each retry reads post-processed sections from the pool but cannot alter
        them: pool entries are immutable and the retry only writes its own
        missing section. Deterministic failures (no evidence, spent budget)
        are not retried.
        """
        if not pending_resume or not config.output_pool_resume():
            return
        for heading, (system, instruction, recs, words) in list(pending_resume.items()):
            if heading in pool.snapshot():
                continue
            if budget is not None and not budget.available():
                break
            retry_instruction = instruction + _pool_grounding(exclude=heading)
            before = len(failed_sections)
            tmp_failed: list[str] = []
            section = _write_section(system, question, heading, retry_instruction,
                                     recs, refs, targets, budget,
                                     min_words=words, failed=tmp_failed)
            if section:
                pool.complete(heading, section)
                # Drop the original failure: the section now exists verbatim.
                failed_sections[:] = [f for f in failed_sections
                                      if not f.startswith(heading + " (")]
            elif tmp_failed and before == len(failed_sections):
                # Keep the original reason; the retry failing too adds no
                # new information, only a duplicate entry.
                pass
            del pending_resume[heading]

    body_specs: list[tuple] = []
    for dimension in dims:
        relevant = _select_for_dimension(records, dimension, targets)
        if not relevant:
            continue
        guidance = _DIMENSION_GUIDANCE.get(dimension.lower().strip(),
                                            _FALLBACK_GUIDANCE)
        body_specs.append((DIMENSION_SYSTEM, dimension.title(),
                            guidance, relevant, 700 if is_deep else 300))

    if contradictions:
        body_specs.append(
            (CONFLICT_SYSTEM, "Where the Evidence Conflicts",
             "Reconcile the disagreements recorded below.",
             _select_for_dimension(records, "conflicting evidence", targets) or records,
             500 if is_deep else 250))

    if is_deep:
        for heading, system, instruction, words in (
            ("Scenario Analysis", SCENARIO_SYSTEM,
             "Work through the scenarios that change the answer.", 600),
            ("Decision Framework", DECISION_SYSTEM,
             "Give the reader conditional guidance for choosing.", 500),
        ):
            body_specs.append((system, heading, instruction, records, words))

    quality_instruction = "Assess the evidence base and its weaknesses."
    if holes:
        # The limitations section must name each hole plainly: an unresolved
        # gap stated in the report is a finding, not a failure.
        quality_instruction += ("\n\nUNRESOLVED EVIDENCE GAPS this run could not "
                                "close (state each as unresolved; do not write "
                                "around them):\n" + "\n".join(f"- {h}" for h in holes))
    body_specs.append(
        (QUALITY_SYSTEM, "Evidence Quality and Limitations",
         quality_instruction,
         records, 400 if is_deep else 200))

    summary_pool = records if len(records) <= 40 else _select_for_dimension(records, "", targets)
    tail_specs: list[tuple] = [
        (METHOD_SYSTEM, "Scope and Methodology",
         "Describe what was investigated and what this method cannot establish.",
         records, 350 if is_deep else 200),
        (EXEC_SYSTEM, "Executive Summary",
         "Summarise the findings the detailed sections establish.",
         summary_pool, 450 if is_deep else 250),
        (CONCLUSION_SYSTEM, "Conclusion",
         "Give the balanced synthesis and name what would change the answer.",
         records, 400 if is_deep else 250),
    ]

    planned = len(body_specs) + len(tail_specs)
    written = 0
    # First pass in planned order; the pool accumulates verbatim sections.
    for spec in body_specs + tail_specs:
        # Summarising sections are grounded in finished output, not just raw
        # evidence, so a mid-report failure cannot silently change what they say.
        ground = spec[1] in ("Executive Summary", "Conclusion", "Scope and Methodology")
        if compose(*spec, ground=ground):
            written += 1

    # Second pass: fallbacks resume only the model-failed sections, reading
    # the pool but never altering completed data.
    resume_from_pool()
    done = pool.snapshot()
    written = sum(1 for spec in body_specs + tail_specs if spec[1] in done)

    ordered = [s for s in tail_specs[1:2]] + [tail_specs[0]] + body_specs + [tail_specs[2]]
    body_sections = [done[spec[1]] for spec in body_specs if spec[1] in done]
    methodology = done.get("Scope and Methodology", "")
    executive = done.get("Executive Summary", "")
    conclusion = done.get("Conclusion", "")
    sections: list[tuple[str, str]] = [(spec[1], done[spec[1]])
                                       for spec in ordered if spec[1] in done]

    if executive:
        parts.append(executive)
        parts.append("")
    if methodology:
        parts.append(methodology)
        parts.append("")
    parts.extend(body_sections)
    if conclusion:
        parts.append(conclusion)
        parts.append("")

    evidence_table = _evidence_table(records, refs)
    if evidence_table:
        parts.append(evidence_table)
        parts.append("")

    # Completion reflects evidence adequacy, not just section assembly: named
    # holes join the missing list so the run reports partial until coverage
    # exists, and the banner exposes each gap instead of hiding it.
    #
    # Evidence-integrity failures join the same list. Every section can be
    # written and every citation can resolve while the run has silently lost
    # most of its extraction batches -- the report would read as complete and
    # be resting on a fraction of the evidence it appears to summarise.
    integrity = [str(n) for n in (notes or [])
                 if str(n).startswith(INTEGRITY_PREFIX)]
    missing = (list(failed_sections)
               + [h for h in holes if h not in failed_sections]
               + [i for i in integrity if i not in failed_sections])
    status = classify_status("x", missing, planned, written)
    if missing:
        # Never silently present a section-less report as a finished one, in the
        # UI or in the exported file.
        parts.append(partial_banner(status, missing))

    parts.append(render_references(records, refs))

    # Ordinary notes stay out of the way in a comment, but anything that
    # compromised the evidence base is printed where a reader will see it: a
    # pipeline problem hidden in markup is a pipeline problem nobody reads.
    advisory = [str(n) for n in (notes or [])
                if not str(n).startswith(INTEGRITY_PREFIX)]
    if advisory:
        parts.append("\n<!-- pipeline notes: " +
                     "; ".join(advisory).replace("--", "") + " -->")
    visible = integrity_notes(notes)
    if visible:
        parts.append("")
        parts.append("### How this run went")
        parts.append("")
        parts.append("The following affected the evidence this report rests on:")
        parts.append("")
        for item in visible:
            parts.append(f"- {str(item)[len(INTEGRITY_PREFIX):].capitalize()}")
        parts.append("")

    markdown = "\n".join(parts).strip() + "\n"
    return ReportResult(markdown, status, missing,
                        list(notes or []), sections=sections)


def _resumable_failure(reason: str) -> bool:
    """Only transient model failures deserve a pool-grounded resume.

    Missing evidence and an exhausted budget are deterministic: retrying them
    reproduces the same outcome while burning calls. A failed model call may
    succeed on a fallback that can read the pool.
    """
    text = str(reason or "")
    if "budget" in text:
        return False
    if "no evidence was selected" in text or "no usable evidence" in text:
        return False
    return True


def write_report(question, collector, contradictions,
                 records: list[Evidence] | None = None,
                 source_synth: dict | None = None,
                 notes: list[str] | None = None,
                 plan: dict | None = None,
                 budget=None) -> str:
    """Markdown only. See :func:`generate_report` for the completion state."""
    return generate_report(question, collector, contradictions, records,
                           source_synth, notes, plan, budget).markdown


def _evidence_table(records: list[Evidence], refs: dict[str, int]) -> str:
    """Source-by-source table. Deterministic: it only restates real metadata."""
    grouped: dict[int, list[Evidence]] = {}
    for record in records:
        number = refs.get(record.usable_url)
        if number:
            grouped.setdefault(number, []).append(record)
    if not grouped:
        return ""

    lines = ["## Evidence by Source", "",
             "| Ref | Source | Type | Published | Retrieved | Findings | Strength |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for number in sorted(grouped):
        group = grouped[number]
        head = group[0]
        strengths = {r.quality for r in group}
        strength = min(strengths, key=lambda q: QUALITY_ORDER.index(q) if q in QUALITY_ORDER else 9)
        retrieved = {r.retrieval_status for r in group}
        retrieved_label = ("full" if retrieved == {"full"}
                           else "partial" if "partial" in retrieved else "metadata only")
        lines.append(
            f"| [{number}] | {head.source_title or head.source_url} "
            f"| {source_type_label(head.source_type)} "
            f"| {head.publication_date or 'undated'} "
            f"| {retrieved_label} | {len(group)} | {strength} |"
        )
    return "\n".join(lines) + "\n"


def _write_report_legacy(question, collector, contradictions) -> str:
    """Original single-call path, kept for runs with no retrieved evidence."""
    from .report_prompts import SYSTEM as LEGACY_SYSTEM
    user = (f"Question: {question}\n\nEVIDENCE:\n{collector.context()}\n\n"
            f"CONTRADICTIONS FOUND:\n{contradictions}")
    return ask(LEGACY_SYSTEM.replace("{today}", str(datetime.date.today())), user, role="synth")