"""Batched LLM evidence extraction and per-source synthesis.

The design constraint is that evidence throughput per model call has to be high:
extracting one finding at a time would mean hundreds of calls for a single
report. So passages are packed to a character budget and many chunks go into
each request, and one request yields every finding those chunks support.

Two deterministic guards run afterwards, because a model will happily produce a
plausible citation for a passage that does not contain it:

  * a record naming a chunk that was not in the batch is discarded
  * a quote that cannot be found in its passage is downgraded, then discarded

That means a surviving record's provenance is real rather than asserted.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from . import config
from .chunker import Chunk
from .evidence import (Evidence, dedupe, from_chunk, normalise_confidence,
                       normalise_quality, quote_is_supported)
from .llm import GROUNDING, LLMChainError, ask
from .relevance import (Scored, dimensions as question_dimensions,
                        target_variant_mismatch)
from .relevance import targets as question_targets

log = logging.getLogger(__name__)

EXTRACT_SYSTEM = GROUNDING + """

You extract research evidence from source passages.

You are given numbered passages, each with an id like S3#2 (source 3, chunk 2).
For each passage, extract only the findings that actually bear on the research
question and its dimensions.

Rules you must follow:

1. GROUND EVERY RECORD IN ITS PASSAGE. Copy the supporting sentence verbatim into
   "quote". If you cannot quote it, do not report it. Never invent a number,
   a benchmark result, a paper finding, or a URL.
2. STATE FINDINGS AS CHECKABLE CLAIMS, not as impressions. Give the specific
   measured value or documented fact and the conditions it holds under -- not a
   vague statement that something is "better", "faster" or "more mature".
3. KEEP THE CONDITIONS WITH THE NUMBER. If a figure only holds for a debug build,
   one device, one workload or one date, say so in "detail".
4. RECORD DISAGREEMENT. If a passage contradicts another passage or a known
   vendor claim, note it in "limitations".
5. CLASSIFY HONESTLY:
   strong     = primary data, transparent measurement, official documentation
   moderate   = credible secondary analysis, well-sourced industry reporting
   weak       = single anecdote, undated blog, opinion, unverified claim
   anecdotal  = a person's experience report
   speculative= prediction, forecast, vendor marketing with no measurement
   Marketing claims from a vendor about its own product are never "strong".
6. SKIP PASSAGES WITH NOTHING USEFUL. Return no record for them. An empty
   result for a passage is correct and cheap.

Return JSON only:
{"evidence":[{"chunk":"S3#2","claim":"...","detail":"...","quote":"verbatim sentence",
"dimension":"one of the dimensions, or an empty string","targets":["..."],
"quality":"strong|moderate|weak|anecdotal|speculative","limitations":"...","confidence":0.0}]}"""

SYNTH_SYSTEM = GROUNDING + """

You summarise what a single source actually establishes.

Using only the evidence records from ONE source, state what that source
supports, what it does not support, and how much weight it can bear.

Be blunt about weak sources. A blog post that asserts a claim without evidence
establishes only that the claim was made. Say so. Do not import knowledge from
outside the records, and do not soften a source's limitations.

Return JSON only:
{"summary":"2-4 sentences on what this source establishes",
"supports":["claim this source genuinely backs, with its evidence ids"],
"does_not_support":["things a reader might assume it supports but it does not"],
"weight":"strong|moderate|weak",
"caveats":"...",
"evidence_ids":["E1","E4"]}"""


def plan_batches(scored: list[Scored]) -> list[list[Scored]]:
    """Pack scored chunks into requests of at most extract_batch_chars.

    Related chunks from the same source stay together where possible, because a
    passage that continues the previous one extracts better with it in view.
    """
    budget = config.extract_batch_chars()
    batches: list[list[Scored]] = []
    current: list[Scored] = []
    current_len = 0

    for item in scored:
        size = len(item.chunk.content)
        if current and (current_len + size > budget or len(current) >= 12):
            batches.append(current)
            current, current_len = [], 0
            if len(batches) >= config.extract_max_batches():
                return batches
        current.append(item)
        current_len += size

    if current:
        batches.append(current)
    return batches[: config.extract_max_batches()]


def _run_parallel(items: list, worker, parallelism: int) -> list:
    """Apply ``worker`` across ``items`` concurrently, returning results in ORDER.

    Completion order is deliberately discarded: callers depend on positional
    stability because evidence_id and the notes list are both derived from
    submission order. A worker that raises yields None rather than taking the
    whole stage down, matching the per-batch isolation of the serial loop.
    """
    if parallelism <= 1 or len(items) <= 1:
        out = []
        for item in items:
            try:
                out.append(worker(item))
            except Exception as exc:               # pragma: no cover - defensive
                log.warning("parallel worker failed: %s: %s",
                            type(exc).__name__, str(exc)[:160])
                out.append(None)
        return out

    results: list = [None] * len(items)

    def guarded(index: int, item):
        try:
            return index, worker(item)
        except Exception as exc:
            log.warning("parallel worker failed: %s: %s",
                        type(exc).__name__, str(exc)[:160])
            return index, None

    with ThreadPoolExecutor(max_workers=min(parallelism, len(items))) as pool:
        for index, value in pool.map(
                lambda pair: guarded(pair[0], pair[1]), list(enumerate(items))):
            results[index] = value
    return results


def extract_evidence(
    scored: list[Scored],
    question: str,
    plan: dict | None = None,
    budget=None,
) -> tuple[list[Evidence], list[str]]:
    """Turn ranked passages into verified, traceable evidence records.

    Returns (records, notes). ``notes`` explains per-batch failures so the
    report can say what went wrong rather than silently reporting thin evidence.
    """
    notes: list[str] = []
    if not scored:
        return [], ["no passages passed relevance filtering"]

    dims = question_dimensions(question) or [sq.get("question", "")[:60]
                                            for sq in (plan or {}).get("subquestions", [])]
    tgts = question_targets(question)

    batches = plan_batches(scored)
    by_id = {item.chunk.chunk_id: item.chunk for item in scored}
    records: list[Evidence] = []
    calls = 0

    # Batches are independent: each sends its own passages and _harvest
    # validates the reply against exactly those. So they run concurrently, in
    # waves of `parallelism`, with the throttle as the real governor.
    #
    # Results are reassembled in submission order, never completion order:
    # evidence_id is assigned by enumerate() after dedupe() below, so collecting
    # in completion order would renumber every citation in the report from one
    # run of the same question to the next.
    parallelism = max(1, config.llm_parallelism())

    def run_batch(number: int, batch: list[Scored]):
        """One extraction call. Returns (records, rejected, error) and never raises."""
        payload = _render_batch(batch, question, dims, tgts)
        try:
            reply = ask(EXTRACT_SYSTEM, payload, json_mode=True, role="judge")
        except LLMChainError as exc:
            log.warning("extraction batch %d failed: %s", number, exc)
            return [], 0, f"failed: {exc}"
        except Exception as exc:                       # a bad batch must not kill the run
            log.warning("extraction batch %d errored: %s: %s", number,
                        type(exc).__name__, str(exc)[:160])
            return [], 0, f"errored: {type(exc).__name__}"
        accepted, rejected = _harvest(reply, batch, by_id)
        return accepted, rejected, None

    index = 0
    while index < len(batches):
        full_wave = batches[index:index + parallelism]
        wave = full_wave
        clipped = False
        if budget is not None:
            # Claim the wave's calls up front, one at a time, so a budget of 3
            # with parallelism 4 runs 3 batches and stops -- the same
            # truncation the serial loop produced.
            claimed = 0
            for _ in wave:
                if not budget.take(1):
                    break
                claimed += 1
            wave = wave[:claimed]
            clipped = len(wave) < len(full_wave)
        if not wave:
            notes.append(
                f"LLM call budget reached after {index} extraction "
                f"batches; {len(batches) - index} batches left unanalysed")
            break
        numbered = list(enumerate(wave, index + 1))
        results = _run_parallel(
            [(n, b) for n, b in numbered],
            lambda item: run_batch(item[0], item[1]),
            parallelism,
        )

        for (number, batch), (accepted, rejected, error) in zip(numbered, results):
            if error is not None:
                kind, _, detail = error.partition(": ")
                notes.append(f"extraction batch {number} {kind}: {detail}")
                continue
            calls += 1
            records.extend(accepted)
            if rejected:
                notes.append(f"extraction batch {number}: discarded {rejected} unverified "
                             f"record(s) whose quote was not present in the passage")
            log.info("extraction batch %d: %d passages -> %d records (%d rejected)",
                     number, len(batch), len(accepted), rejected)
        index += len(wave)
        if clipped:
            # Say so even when the clipped wave still ran: a report that quietly
            # analysed 3 of 8 batches must not read as a complete pass.
            notes.append(
                f"LLM call budget reached after {index} extraction "
                f"batches; {len(batches) - index} batches left unanalysed")
            break

    records = dedupe(records)
    for position, record in enumerate(records, 1):
        record.evidence_id = f"E{position}"

    # Target fidelity: evidence about a related-but-different variant (another
    # release, another architecture class) is capped at moderate and labelled,
    # so synthesis can tell it apart from directly applicable evidence instead
    # of silently spending strong findings on the wrong target.
    for record in records:
        reason = target_variant_mismatch(record.targets, record.claim, tgts)
        if reason and record.quality == "strong":
            record.quality = "moderate"
            record.limitations = ((record.limitations + " " if record.limitations else "")
                                  + f"applicability: {reason}")
            log.info("capped %s to moderate: %s", record.evidence_id, reason)

    cap = config.max_evidence_records()
    if len(records) > cap:
        records.sort(key=lambda r: (r.rank, -r.confidence))
        notes.append(f"evidence set trimmed from {len(records)} to {cap} records")
        records = records[:cap]
        for position, record in enumerate(records, 1):
            record.evidence_id = f"E{position}"

    if calls == 0 and batches:
        notes.append("no extraction calls were made")
    return records, notes


def _render_batch(batch: list[Scored], question: str,
                  dims: list[str], tgts: list[str]) -> str:
    lines = [f"RESEARCH QUESTION: {question}"]
    if tgts:
        lines.append(f"OPTIONS BEING COMPARED: {', '.join(tgts)}")
    if dims:
        lines.append(f"DIMENSIONS: {', '.join(dims)}")
    lines.append("\nPASSAGES:")

    for number, item in enumerate(batch, 1):
        chunk = item.chunk
        lines.append(
            f"\n[P{number}] id={chunk.chunk_id}\n"
            f"  title: {chunk.source_title or '(untitled)'}\n"
            f"  url: {chunk.source_url}\n"
            f"  publisher: {chunk.source_publisher or 'unknown'}"
            f" | published: {chunk.publication_date or 'undated'}\n"
            f"  section: {chunk.section or '(none)'}"
            f"{f' | page {chunk.page}' if chunk.page else ''}\n"
            f"  retrieval: {chunk.retrieval_status}\n"
            f"  TEXT:\n{chunk.content}\n"
            f"  (use id \"{chunk.chunk_id}\" in your records)"
        )
    return "\n".join(lines)


def _harvest(reply, batch: list[Scored],
             by_id: dict[str, Chunk]) -> tuple[list[Evidence], int]:
    """Validate model output against the passages that were actually sent."""
    if not isinstance(reply, dict):
        return [], 0
    raw = reply.get("evidence")
    if not isinstance(raw, list):
        return [], 0

    # P-numbers in the prompt map to the chunk actually sent under them; the
    # real chunk id is also accepted, since that is what the prompt asks for.
    position = {f"P{i}": item.chunk for i, item in enumerate(batch, 1)}
    position.update({item.chunk.chunk_id: item.chunk for item in batch})

    accepted: list[Evidence] = []
    rejected = 0

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("chunk") or entry.get("chunk_id") or "").strip()
        chunk = position.get(key) or by_id.get(key)
        if chunk is None:
            rejected += 1                      # invented passage reference
            continue

        claim = str(entry.get("claim") or "").strip()
        if len(claim) < 12:
            rejected += 1
            continue

        quote = str(entry.get("quote") or "").strip().strip('"')
        supported = quote_is_supported(quote, chunk.content)
        if quote and not supported:
            # Not fabricated outright, but not verbatim either: keep the claim
            # only if it is clearly not carrying a specific unverifiable number.
            if any(ch.isdigit() for ch in claim) or any(ch.isdigit() for ch in quote):
                rejected += 1
                continue
            quote = ""

        targets = entry.get("targets")
        if isinstance(targets, str):
            targets = [targets]
        elif not isinstance(targets, list):
            targets = []

        record = Evidence(
            claim=claim,
            detail=str(entry.get("detail") or "").strip()[:600],
            quote=quote[:400],
            dimension=str(entry.get("dimension") or "").strip()[:80],
            targets=[str(t)[:60] for t in targets][:6],
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
            retrieval_status=chunk.retrieval_status,
            retrieval_limitation=chunk.retrieval_limitation,
            quality=normalise_quality(entry.get("quality")),
            limitations=str(entry.get("limitations") or "").strip()[:400],
            confidence=normalise_confidence(entry.get("confidence")),
            citation_count=int(getattr(chunk, "cited_by", 0) or 0),
        )

        # A passage we only partly read cannot support strong evidence.
        if chunk.retrieval_status == "metadata_only":
            # Only the search snippet existed for this source. Anything derived
            # from it is a statement about the snippet, not about the document,
            # so it can never outrank directly-retrieved evidence.
            record.quality = "weak"
            record.confidence = min(record.confidence, 0.3)
            record.limitations = (
                (record.limitations + " " if record.limitations else "") +
                "only the search-result metadata/snippet was retrievable for this "
                "source; the underlying document was NOT read, so this says only "
                "what the snippet asserted"
            )
        elif chunk.retrieval_status != "full" and record.quality == "strong":
            record.quality = "moderate"
            record.limitations = (record.limitations + " " if record.limitations else "") + \
                "source was only partially retrievable, so the claim may rest on unseen text"

        accepted.append(record)
    return accepted, rejected


def synthesise_sources(
    records: list[Evidence],
    budget=None,
) -> dict[str, dict]:
    """Per-source synthesis, one honest call per source.

    Answers, for each source, what it actually supports and what a reader might
    wrongly assume it supports -- the difference the report needs in order not
    to over-claim on a source's behalf. Each call covers exactly one source so
    the reply can only ever be attributed to the source it assessed; batching
    several sources into one prompt silently dropped all but the first.
    """
    if not records:
        return {}

    grouped: dict[str, list[Evidence]] = {}
    for record in records:
        grouped.setdefault(record.source_id or record.source_url, []).append(record)

    # One call per source, and each call is independent of every other, so they
    # run concurrently in waves. Results are reassembled in submission order so
    # the returned dict iterates sources in the same order as before.
    candidates: list[tuple[str, list[Evidence]]] = []
    for source_id, group_records in grouped.items():
        # Low-value work is skipped, not paid for: a source whose findings are
        # all weak or worse contributes no assessable weight, so synthesising
        # it would spend quota without improving the report.
        if not any(r.quality in ("strong", "moderate") for r in group_records):
            log.info("source synthesis skipped weak-only source %s", source_id)
            continue
        candidates.append((source_id, group_records))

    def run_one(source_id: str, group_records: list[Evidence]):
        head = group_records[0]
        payload = (
            f"\n=== SOURCE {source_id} ===\n{head.citation}\n{head.usable_url}\n"
            + "\n".join(f"  {r.to_prompt()}" for r in group_records[:14])
        )
        try:
            reply = ask(SYNTH_SYSTEM, payload, json_mode=True, role="synth")
        except LLMChainError as exc:
            log.warning("source synthesis failed: %s", exc)
            return None
        except Exception as exc:
            log.warning("source synthesis errored: %s", exc)
            return None
        if isinstance(reply, dict) and isinstance(reply.get("summary"), str):
            return {
                "summary": reply["summary"][:900],
                "supports": _str_list(reply.get("supports"))[:8],
                "does_not_support": _str_list(reply.get("does_not_support"))[:6],
                "weight": normalise_quality(reply.get("weight")),
                "caveats": str(reply.get("caveats") or "")[:400],
            }
        return None

    out: dict[str, dict] = {}
    parallelism = max(1, config.llm_parallelism())
    index = 0
    while index < len(candidates):
        wave = candidates[index:index + parallelism]
        if budget is not None:
            claimed = 0
            for _ in wave:
                if not budget.take(1):
                    break
                claimed += 1
            if claimed < len(wave):
                log.info("source synthesis truncated: budget exhausted")
            wave = wave[:claimed]
            if not wave:
                break
        results = _run_parallel(wave, lambda item: run_one(*item), parallelism)
        for (source_id, _records), value in zip(wave, results):
            if value:
                out[source_id] = value
        index += len(wave)
        if budget is not None and claimed < parallelism:
            break
    return out


def _str_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return []


CONFLICT_SYSTEM = GROUNDING + """

You identify genuine disagreements between research findings.

You are given numbered findings, each from a specific source with its date and
type. Find places where sources genuinely contradict each other -- different
conclusions about the same thing, or incompatible numbers for the same
measurement.

Do NOT report a conflict merely because two sources emphasise things
differently, or measure different versions, devices or workloads without saying
so. Before declaring a conflict, check whether the difference is explained by
date, configuration, workload, methodology or scope; if it is, that belongs in
the explanation rather than being called a contradiction.

Return JSON only:
{"contradictions":[{"topic":"what is disputed","side_a":"position A","sources_a":["E1"],
"side_b":"position B","sources_b":["E4"],"likely_cause":"differences in ... or unknown",
"more_general":"which is more generalisable, or empty"}]}
Return an empty list if the findings genuinely agree."""

CONFLICT_BATCH_CHARS = 11_000


def find_conflicts(records: list[Evidence], budget=None,
                   rounds: int = 2) -> list[dict]:
    """Batch the evidence and ask for disagreements, then keep only real ones.

    Runs over the extracted findings rather than raw snippets, so the comparison
    is between what sources actually established.
    """
    if len(records) < 2:
        return []

    batches = _conflict_batches(records)
    out: list[dict] = []
    for batch in batches[:rounds]:
        if budget is not None and not budget.take(1):
            break
        payload = "\n\n".join(r.to_prompt() for r in batch)
        try:
            reply = ask(CONFLICT_SYSTEM,
                         f"FINDINGS:\n{payload}", json_mode=True, role="judge")
        except LLMChainError as exc:
            log.warning("contradiction detection failed: %s", exc)
            continue
        except Exception as exc:
            log.warning("contradiction detection errored: %s", exc)
            continue
        if isinstance(reply, dict) and isinstance(reply.get("contradictions"), list):
            out.extend(c for c in reply["contradictions"] if isinstance(c, dict))
    return out


def _conflict_batches(records: list[Evidence]) -> list[list[Evidence]]:
    batches, current, size = [], [], 0
    for record in records:
        rendered = len(record.claim) + len(record.detail) + 200
        if current and size + rendered > CONFLICT_BATCH_CHARS:
            batches.append(current)
            current, size = [], 0
        current.append(record)
        size += rendered
    if current:
        batches.append(current)
    return batches


def findings_from_contradictions(contradictions: list[dict],
                                 records: list[Evidence]) -> list[Evidence]:
    """Record disagreements found earlier as first-class evidence."""
    by_number = {index + 1: record for index, record in enumerate(records)}
    out: list[Evidence] = []
    for item in contradictions or []:
        if not isinstance(item, dict):
            continue
        side_a = str(item.get("side_a") or "").strip()
        side_b = str(item.get("side_b") or "").strip()
        if not side_a or not side_b:
            continue
        anchor = None
        for number in (item.get("sources_a") or []):
            try:
                anchor = by_number.get(int(number)) or anchor
            except (TypeError, ValueError):
                continue
        if anchor is None and records:
            anchor = records[0]
        if anchor is None:
            continue
        out.append(Evidence(
            claim=f"Conflicting evidence on: {item.get('topic', 'an unspecified topic')}",
            detail=f"Position A: {side_a} | Position B: {side_b}",
            dimension="conflicting evidence",
            targets=[],
            source_id=anchor.source_id,
            source_title=anchor.source_title,
            source_url=anchor.source_url,
            source_type=anchor.source_type,
            publisher=anchor.publisher,
            publication_date=anchor.publication_date,
            section=anchor.section,
            page=anchor.page,
            chunk_id=anchor.chunk_id,
            retrieval_status=anchor.retrieval_status,
            quality="moderate",
            limitations="the two positions were detected automatically and have not "
                        "been checked for a shared cause such as differing workloads",
            confidence=0.45,
        ))
    return out


def snippet_evidence(chunk: Chunk) -> Evidence:
    """Metadata-only record for a source that could not be downloaded."""
    return from_chunk(chunk, 0)