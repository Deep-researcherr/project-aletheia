"""Question → sub-questions, deduplicated.

A planner asked to split one question into sub-questions frequently returns the
same sub-question twice, or two phrasings of it. Those duplicates waste a
parallel search slot and skew the evidence base, because the same material is
retrieved and cited twice while another dimension goes unresearched.

Duplicates are detected deterministically, then replaced by asking the planner
again for fresh ones. If it cannot supply replacements, the run proceeds with
what survived -- a smaller plan is better than no plan.
"""
import datetime
import logging
import math
import re

from . import config
from .llm import ask

log = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")

_STOP = frozenset("""
a an and are as at be by between differ difference for from how in into is it
its of on or the their there these this to what which who why with do does did
about across during over under
""".split())

# SerpApi engines plus every source the provider registry offers. Built from
# the registry so a provider added there becomes plannable here without a
# second edit -- and, more importantly, so it cannot be added to the registry
# and then be silently dropped, which would spend a call per sub-question and
# return nothing.
def _sources() -> tuple[str, ...]:
    from .providers import source_names
    return ("web", "news", "scholar", "github", "patent", "arxiv") + source_names()


SOURCES = _sources()

SYSTEM = """You are a research planner. Break the user's question into 10-14 sub-questions.
Sub-questions must be DISTINCT from each other: each must cover a different
aspect, and none may restate another. Treat any overlap as failure: if two
sub-questions could be answered by the same three sources, they are the same
question and one of them must go. Do not create several sub-questions that
differ only in wording.
Unlike the report writers downstream, you are encouraged to elaborate: make
each sub-question DEEPER AND LONGER than the user's question by adding domain
context, the compared entities, the conditions that matter, and concrete
technical terms. "How does X compare to Y for Z under condition W?" beats
"Tell me about X". Terse questions produce generic search results; long,
specific questions produce detailed ones.
For each, give 2-3 concrete search queries and the best source for each:
- web: docs, blogs, comparisons   - news: recent events/announcements
- scholar: peer-reviewed papers, benchmarks
- github: repos, issues, ecosystem activity
- patent: patented mechanisms, filings, prior art (technical questions only)
- arxiv: preprints and very recent methods (quantitative/technical questions
  only; skip for law, history or other non-technical subjects)
- exa: pages that mean what the question means even when the wording differs;
  use when the obvious keywords would miss the right page
- tinyfish: an independent web index; use as a second opinion on a claim
- jina: pages returned with their own summaries attached
These three rank differently from web, so route to them only when a sub-question
would genuinely benefit -- spreading every query across all of them retrieves
six near-copies of the same page and wastes the budget.
Return JSON only:
{"subquestions":[{"question":"...","searches":[{"source":"web","query":"..."}]}]}
Today's date is {today}. Include the current year in queries where recency matters.
Make every query self-contained: include the compared entities and the domain,
never a bare ambiguous term (write "attention mechanism transformer", not
"attention"). For technical comparisons, route at least one query to scholar
or patent rather than web commentary."""

REGEN_SYSTEM = """You are a research planner replacing sub-questions that were rejected as duplicates.

You are given the original question, the sub-questions already kept, and the
rejected ones. Produce replacement sub-questions that cover DIFFERENT ground:
an aspect of the original question that none of the kept sub-questions covers.

Rules:
- Every replacement must be clearly distinct from every kept sub-question and
  from every other replacement. Check each pair before answering.
- Do not paraphrase or restate a rejected sub-question; go somewhere new.
- Elaborate like the first planning pass: each replacement should be deep and
  specific, naming entities, conditions and technical terms, never terse.
- If the original question is already fully covered by the kept sub-questions,
  return an empty list rather than padding it out.
- If you can only offer replacements you are confident are distinct, offer fewer.

Return JSON only:
{"subquestions":[{"question":"...","searches":[{"source":"web","query":"..."}]}]}
Today's date is {today}."""

# Deliberately simple: a plan is short, and a full stemmer would cost more than
# it is worth at this size. The trailing-"e" rule is what lets "compare" and
# "comparison" -- or "benchmark" and "benchmarks" -- land on the same term, which
# is the commonest way a duplicate sub-question is merely reworded.
_SUFFIXES = ("ational", "ization", "iveness", "fulness", "ousness", "ations",
             "ation", "ments", "ment", "nesses", "ness", "ances", "ance",
             "ences", "ence", "ingly", "ings", "ing", "ies", "ied", "ers", "est",
             "ions", "ion", "ive", "ful", "ous", "ed", "es", "s")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[:-len(suffix)]
    if len(word) > 5 and word.endswith("e"):
        return word[:-1]
    return word


def _terms(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall((text or "").lower())
            if len(w) > 2 and w not in _STOP}


def _char_grams(text: str, n: int = 3) -> set[str]:
    """Character n-grams of the normalised text.

    Catches rewordings that term overlap misses, because it is blind to
    morphology: "compare"/"comparison" and "benchmark"/"benchmarks" collapse
    without needing a stemmer, and a light stemmer cannot be trusted not to
    over-merge unrelated words.
    """
    squashed = " ".join(_WORD.findall((text or "").lower()))
    if len(squashed) < n:
        return {squashed} if squashed else set()
    return {squashed[i:i + n] for i in range(len(squashed) - n + 1)}


def _pair_similarity(a_terms: set[str], b_terms: set[str],
                     freq: dict[str, int], total: int) -> float:
    """Rare-term overlap of two sub-questions, in [0, 1].

    Sub-questions from one plan inevitably share the names of the entities being
    compared, so a plain Jaccard score would call genuinely different dimensions
    duplicates. Weighting by how *rare* a term is across the set fixes that:
    sharing only ubiquitous entity names costs almost nothing, while sharing a
    distinctive term is what marks a real restatement.
    """
    shared = a_terms & b_terms
    if not shared:
        return 0.0

    def weight(term: str) -> float:
        return math.log(1.0 + total / (1 + freq.get(term, 0)))

    numerator = sum(weight(t) for t in shared)
    denominator = sum(weight(t) for t in (a_terms | b_terms))
    return numerator / denominator if denominator else 0.0


def _gram_similarity(left: str, right: str) -> float:
    a, b = _char_grams(left), _char_grams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _clean_plan(plan: dict) -> dict:
    """Keep only well-formed sub-questions.

    A sub-question with no usable search contributes nothing: it is never
    dispatched, so retaining it would overstate coverage in the plan view. It is
    dropped, unless doing so would empty the plan entirely -- a plan with
    questions but no searches still beats no research at all.
    """
    usable, searchless = [], []
    for sq in plan.get("subquestions", []):
        if not isinstance(sq, dict):
            continue
        text = str(sq.get("question") or "").strip()
        if not text:
            continue
        searches = [{"source": s["source"], "query": str(s["query"]).strip()}
                    for s in (sq.get("searches") or [])
                    if isinstance(s, dict)
                    and s.get("source") in SOURCES
                    and str(s.get("query") or "").strip()]
        (usable if searches else searchless).append(
            {"question": text, "searches": searches})

    return {"subquestions": usable or searchless}


def dedupe_subquestions(plan: dict, question: str = "") -> tuple[dict, list[dict]]:
    """Drop sub-questions that restate an earlier one.

    Returns ``(clean_plan, rejected)``. The first occurrence of a duplicate
    family is kept -- ordering is meaningful because the planner tends to put
    the broadest question first.
    """
    cleaned = _clean_plan(plan)
    subs = cleaned["subquestions"]
    if (len(subs) < 2
            or config.plan_dedup_threshold() >= 1.0
            or config.plan_dedup_gram_threshold() >= 1.0):
        return cleaned, []

    term_sets = [_terms(sq["question"]) for sq in subs]
    freq: dict[str, int] = {}
    for bag in term_sets:
        for term in bag:
            freq[term] = freq.get(term, 0) + 1
    total = len(term_sets)

    term_threshold = config.plan_dedup_threshold()
    gram_threshold = config.plan_dedup_gram_threshold()

    kept_terms: list[set[str]] = []
    kept_grams: list[set[str]] = []
    kept: list[dict] = []
    rejected: list[dict] = []

    for sub, bag in zip(subs, term_sets):
        grams = _char_grams(sub["question"])
        duplicate_of = None
        for other, other_bag, other_grams in zip(kept, kept_terms, kept_grams):
            # Two signals on different scales, so each gets its own threshold.
            # Rare-term overlap catches a hard restatement and is deliberately
            # strict, because shared entity names are common and harmless.
            # Character overlap catches the same question reworded. Measured on
            # real plans the reworded pair scores ~0.61 while genuinely distinct
            # dimensions score 0.05-0.09, so 0.45 sits in a wide gap.
            term_score = _pair_similarity(bag, other_bag, freq, total)
            gram_score = (len(grams & other_grams) / len(grams | other_grams)
                          if grams and other_grams else 0.0)
            if term_score >= term_threshold:
                score = term_score
            elif gram_score >= gram_threshold:
                score = gram_score
            else:
                continue
            duplicate_of = {"question": sub["question"],
                            "duplicate_of": other["question"],
                            "similarity": round(score, 3)}
            break
        if duplicate_of:
            rejected.append(duplicate_of)
        else:
            kept.append(sub)
            kept_terms.append(bag)
            kept_grams.append(grams)

    if rejected:
        log.info("planner: dropped %d duplicate sub-question(s) "
                 "(term >= %.2f or text >= %.2f)",
                 len(rejected), term_threshold, gram_threshold)
    return {"subquestions": kept}, rejected


def regenerate(question: str, kept: list[dict], rejected: list[dict]) -> list[dict]:
    """Ask for replacement sub-questions that cover new ground.

    Failure is not an error: an empty or unusable reply simply leaves the
    surviving plan as it is.
    """
    if not config.plan_dedup_regenerate() or not rejected:
        return []

    kept_text = "\n".join(f"- {sq['question']}" for sq in kept) or "(none)"
    rejected_text = "\n".join(f"- {r['question']}" for r in rejected)
    prompt = (
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"SUB-QUESTIONS ALREADY KEPT (do not repeat or rephrase these):\n{kept_text}\n\n"
        f"THESE WERE REJECTED AS DUPLICATES (go somewhere new, do not restate):\n"
        f"{rejected_text}\n\n"
        f"Provide at most {min(6, len(rejected))} replacement sub-question(s)."
    )

    try:
        reply = ask(REGEN_SYSTEM.replace("{today}", str(datetime.date.today())),
                    prompt, json_mode=True)
    except Exception as exc:
        log.warning("sub-question regeneration failed, proceeding with the "
                    "surviving plan: %s", exc)
        return []

    if not isinstance(reply, dict):
        return []
    return _clean_plan(reply)["subquestions"]


def make_plan(question: str) -> dict:
    """Plan, deduplicate, and replace duplicates where possible.

    A model failure here must not kill the run: fall back to a single
    generic sub-question so retrieval and synthesis still have something to
    work with. Thin coverage is then reported honestly downstream.
    """
    try:
        raw = ask(SYSTEM.replace("{today}", str(datetime.date.today())),
                  question, json_mode=True)
    except Exception as exc:
        log.warning("planner model failed (%s); using a single-question "
                    "fallback plan", type(exc).__name__)
        asked = (question or "").strip()
        return {"subquestions": [{"question": asked or "research topic",
                                  "searches": [
                                      {"source": "web", "query": asked},
                                      {"source": "news", "query": asked}]}]}
    if not isinstance(raw, dict):
        return {"subquestions": []}

    cleaned, rejected = dedupe_subquestions(raw, question)
    if not rejected:
        return _expand_plan(cleaned)

    replacements = regenerate(question, cleaned["subquestions"], rejected)

    # Replacements may collide with each other or with what was kept, so the
    # whole set is filtered again rather than trusting the new text.
    for extra in replacements:
        cleaned["subquestions"].append(extra)

    final, still_rejected = dedupe_subquestions(cleaned, question)
    dropped = len(still_rejected) - len(rejected)
    if dropped > 0:
        log.info("planner: %d replacement(s) were also duplicates and were dropped",
                 dropped)
    if not final["subquestions"]:
        # Everything was a duplicate of something. Keeping one is strictly better
        # than planning nothing, so fall back to the first cleaned sub-question.
        log.warning("planner: every sub-question looked duplicated; keeping one")
        fallback = cleaned if cleaned["subquestions"] else {"subquestions": []}
        return _expand_plan(fallback)
    return _expand_plan(final)


def _expand_plan(plan: dict) -> dict:
    """Sharpen each query with autocomplete phrasing, when enabled.

    Bounded to one suggestion per query and strictly additive: a suggestion
    replaces the draft only when it introduces a new technical term, and any
    failure keeps the original. Never raises.
    """
    try:
        from . import config as _config
        from .relevance import terms as _terms
        from .searcher import expand_query
        if not _config.search_expand_queries():
            return plan
    except Exception:
        return plan
    for sq in plan.get("subquestions", []):
        if not isinstance(sq, dict):
            continue
        for search in sq.get("searches", []):
            if not isinstance(search, dict):
                continue
            draft = str(search.get("query") or "")
            if not draft:
                continue
            try:
                suggestion = expand_query(draft)
            except Exception:
                continue
            if not suggestion:
                continue
            novel = set(_terms(suggestion)) - set(_terms(draft))
            if novel and len(suggestion.split()) <= 12:
                log.info("planner: expanded %r -> %r", draft, suggestion)
                search["query"] = suggestion
    return plan
