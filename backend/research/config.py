"""Tunable limits for the research pipeline.

Every knob is environment-driven with a working default, so the pipeline runs
with no configuration at all. Values are read on each access (not cached at
import) so tests and the .env file can change them without a restart.
"""
import os

# Rough English prose ratio. Deliberately conservative: underestimating tokens
# is how a batch silently gets truncated by the provider.
CHARS_PER_TOKEN = 3.5

# Verified reachable on 2026-10-04. Any litellm model string works, but it must
# carry a provider prefix and the matching provider key must be present. The
# default mirrors LLM_MODEL so an unset env var does not silently put a
# rate-limited provider (Groq free tier) in front of the chain.
#
# Lives here rather than in llm.py so that modules which only need to *name*
# the chain (the service dashboard) do not have to import litellm, which costs
# several seconds cold.
DEFAULT_MODEL = "gemini/gemini-3.5-flash"


def _int(name: str, default: int, low: int = 0, high: int = 10**9) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def _float(name: str, default: float, low: float = 0.0, high: float = 1e9) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- stage toggles ---------------------------------------------------------

def fetch_enabled() -> bool:
    """Retrieve real source documents behind search results. Default on."""
    return _bool("RESEARCH_FETCH", True)


def analysis_enabled() -> bool:
    """LLM evidence extraction over retrieved chunks. Default on."""
    return _bool("RESEARCH_ANALYZE", True)


# --- wikipedia baseline ------------------------------------------------------

def wiki_enabled() -> bool:
    """Fetch Wikipedia summaries before any LLM call. Default on."""
    return _bool("WIKI_ENABLED", True)


def wiki_max_docs() -> int:
    """Wikipedia baseline documents per run."""
    return _int("WIKI_MAX_DOCS", 3, 0, 10)


def wiki_titles_per_query() -> int:
    """Title hits kept per Wikipedia search (main question + each sub-question)."""
    return _int("WIKI_TITLES_PER_QUERY", 2, 1, 10)


def gap_dup_threshold() -> float:
    """Token-overlap (Jaccard) above which a follow-up repeats a seen query."""
    return _float("GAP_DUP_THRESHOLD", 0.60, 0.0, 1.0)


def model_health_path() -> str:
    """File-backed registry for model cooldowns and prompt ceilings."""
    return os.environ.get("MODEL_HEALTH_PATH", ".cache/model_health.json")


# --- search / source limits ------------------------------------------------

def max_sources() -> int:
    """Distinct sources kept after dedupe, across all searches."""
    return _int("RESEARCH_MAX_SOURCES", 120, 1, 1000)


def plan_dedup_threshold() -> float:
    """Similarity above which two sub-questions count as the same question.

    Scored on rare terms only, so shared entity names do not trigger it. Set to
    1.0 to disable deduplication entirely.
    """
    return _float("PLAN_DEDUP_THRESHOLD", 0.70, 0.0, 1.0)


def plan_dedup_gram_threshold() -> float:
    """Similarity above which two sub-questions count as the same question.

    Measured on character overlap, which is blind to wording. On real plans a
    reworded duplicate scores ~0.61 and genuinely distinct dimensions score
    0.05-0.09, so this sits in a wide gap. Set to 1.0 to disable this signal.
    """
    return _float("PLAN_DEDUP_GRAM_THRESHOLD", 0.45, 0.0, 1.0)


def plan_dedup_regenerate() -> bool:
    """Ask the planner for replacement sub-questions when duplicates are dropped."""
    return _bool("PLAN_DEDUP_REGENERATE", True)


def analyse_reuse_sources() -> bool:
    """Skip sources already processed for evidence in an earlier round.

    Each round re-visits the whole corpus otherwise, so extra rounds cost nearly
    full price for no new material. Sources are re-processed only when their
    retrieval grade improves, which happens when a later fetch finally succeeds.
    """
    return _bool("ANALYSE_REUSE_SOURCES", True)


def ai_overview_enabled() -> bool:
    """Fetch full AI Overviews for reference links. Default on.

    Only reference URLs are kept -- the generated prose never enters
    evidence. Costs one extra search per google response carrying one.
    """
    return _bool("AI_OVERVIEW_ENABLED", True)


def results_per_search() -> int:
    return _int("RESEARCH_RESULTS_PER_SEARCH", 6, 1, 20)


def search_expand_queries() -> bool:
    """Sharpen planner queries with autocomplete suggestions. Default on.

    Each expansion is one cached autocomplete call; failures keep the
    original query, so this can never break planning.
    """
    return _bool("SEARCH_EXPAND_QUERIES", True)


def scholar_forward_max() -> int:
    """Highly-cited papers chased forward for citing documents per run."""
    return _int("SCHOLAR_FORWARD_MAX", 2, 0, 10)


def search_blocked_hosts() -> tuple[str, ...]:
    """Hosts never kept from web/news results, comma-separated in env.

    Social and video pages are unreadable to the fetcher (JS walls, login
    walls) and contribute nothing but snippet noise, so they are dropped at
    the search layer before spending fetch budget on them. Scholar and GitHub
    are untouched: their hosts are publishers and repos, not noise. Wikipedia
    is deliberately kept: it feeds the pre-LLM baseline.
    """
    raw = os.environ.get(
        "SEARCH_BLOCKED_HOSTS",
        "twitter.com,x.com,facebook.com,instagram.com,tiktok.com,"
        "youtube.com,pinterest.com",
    )
    return tuple(h.strip().lower().removeprefix("www.")
                 for h in raw.split(",") if h.strip())


def rounds_ceiling() -> int:
    """Hard upper bound on research depth, shared by the API and the UI."""
    return _int("RESEARCH_MAX_ROUNDS_LIMIT", 10, 1, 50)


def max_rounds() -> int:
    """Default research depth (gap-check loop passes)."""
    return _int("RESEARCH_MAX_ROUNDS", 2, 1, rounds_ceiling())


# --- retrieval limits ------------------------------------------------------

def fetch_max_sources() -> int:
    """How many sources we actually attempt to download."""
    return _int("FETCH_MAX_SOURCES", 80, 0, 500)


def fetch_max_bytes() -> int:
    """Per-document download ceiling. Bigger responses are truncated."""
    return _int("FETCH_MAX_BYTES", 4_000_000, 10_000, 64_000_000)


def fetch_timeout_s() -> float:
    return _float("FETCH_TIMEOUT_S", 20.0, 1.0, 300.0)


def fetch_max_workers() -> int:
    return _int("FETCH_MAX_WORKERS", 10, 1, 64)


def fetch_max_pages() -> int:
    """PDF pages parsed. Keeps huge papers bounded."""
    return _int("FETCH_MAX_PAGES", 40, 1, 500)


def fetch_respect_robots() -> bool:
    """Honour robots.txt. Default on: we are a crawler, so behave like one."""
    return _bool("FETCH_RESPECT_ROBOTS", True)


def provider_max_per_run(name: str) -> int:
    """Per-run cap for one extract provider. 0 disables that tier.

    The default lives in the registry (``Provider.cap_default``) rather than
    here, so adding a provider does not also mean editing the config module:
    the env var is the override, the registry entry is the default.

    The env var is the provider name uppercased with a ``_MAX_PER_RUN``
    suffix, which is what TAVILY_MAX_PER_RUN and FIRECRAWL_MAX_PER_RUN already
    were -- so the two existing variables keep working unchanged and the new
    providers follow the same shape.
    """
    try:
        from .providers import for_role

        provider = next((p for p in for_role("extract") if p.name == name), None)
        default = provider.cap_default if provider else 10
    except Exception:
        default = 10
    return _int(f"{name.upper()}_MAX_PER_RUN", default, 0, 500)


def tavily_max_per_run() -> int:
    """Paid Tavily fallback pages per research run. 0 disables it."""
    return provider_max_per_run("tavily")


def firecrawl_max_per_run() -> int:
    """Paid Firecrawl fallback pages per research run. 0 disables it."""
    return provider_max_per_run("firecrawl")


def extract_tier_order() -> str:
    """Fallback order for unreadable pages, as configured.

    Read here so the env-only policy knob sits with the other env knobs; the
    authoritative list still lives in providers.py, which owns the defaults.
    """
    return (os.environ.get("EXTRACT_TIER_ORDER") or "").strip()


def search_fanout() -> bool:
    """Ask every wired open-web engine each general query, concurrently.

    Default on. The engines rank differently -- keyword, semantic, independent
    index -- so the union of their answers is more evidence than the best
    single one, and the collector dedupes by URL so the overlap costs nothing
    but a fetch attempt.

    Turn it off to go back to one engine per planned query, which is a third
    of the search spend and a thinner report.
    """
    return _bool("SEARCH_FANOUT", True)


def search_fanout_max_engines() -> int:
    """Ceiling on engines per general query. 0 means no ceiling."""
    return _int("SEARCH_FANOUT_MAX_ENGINES", 0, 0, 12)


def fetch_max_chars() -> int:
    """Text kept from one document after cleaning."""
    return _int("FETCH_MAX_CHARS", 400_000, 1_000, 4_000_000)


# --- chunking limits -------------------------------------------------------

def chunk_target_chars() -> int:
    return _int("CHUNK_TARGET_CHARS", 1400, 200, 20_000)


def chunk_max_chars() -> int:
    return _int("CHUNK_MAX_CHARS", 2600, 400, 40_000)


def chunks_per_source() -> int:
    """Cap on chunks kept per source, so one long doc cannot dominate."""
    return _int("CHUNKS_PER_SOURCE", 8, 1, 60)


# --- relevance + extraction ------------------------------------------------

def relevance_top_chunks() -> int:
    """Chunks that survive deterministic relevance scoring."""
    return _int("RELEVANCE_TOP_CHUNKS", 150, 4, 2000)


def relevance_min_score() -> float:
    return _float("RELEVANCE_MIN_SCORE", 0.0, 0.0, 100.0)


def extract_batch_chars() -> int:
    """Chars of chunk text per extraction call. Drives evidence per LLM call."""
    return _int("EXTRACT_BATCH_CHARS", 6_000, 1_000, 200_000)


def extract_max_batches() -> int:
    """Hard ceiling on extraction calls, whatever the evidence volume."""
    return _int("EXTRACT_MAX_BATCHES", 40, 1, 200)


def llm_parallelism() -> int:
    """Model calls issued concurrently within a stage.

    Extraction batches and per-source synthesis are independent of each other, so
    they no longer run one after another. This is how many at a time; the
    throttle (THROTTLE_LLM_MAX_CONCURRENT / _PER_MINUTE) still governs what the
    provider actually sees, and stays the real limit -- raise this first and
    lower it if the chain starts leaning on fallbacks.
    """
    return _int("LLM_PARALLELISM", 4, 1, 32)


def max_evidence_records() -> int:
    """Cap on structured evidence kept for the report."""
    return _int("MAX_EVIDENCE_RECORDS", 400, 4, 2000)


# --- reference discovery ---------------------------------------------------

def references_enabled() -> bool:
    """Mine outbound links/DOIs from fetched documents for extra leads."""
    return _bool("RESEARCH_REFERENCES", True)


def references_max() -> int:
    """Total extra sources pulled in by reference discovery."""
    return _int("REFERENCES_MAX", 6, 0, 60)


def references_max_per_source() -> int:
    return _int("REFERENCES_PER_SOURCE", 3, 0, 20)


def references_max_depth() -> int:
    """Citation-traversal depth. 1 = references of retrieved docs only."""
    return _int("REFERENCES_MAX_DEPTH", 1, 0, 3)


# --- report generation -----------------------------------------------------

def report_depth() -> str:
    """'deep' (default) or 'brief'."""
    return os.environ.get("REPORT_DEPTH", "deep").strip().lower() or "deep"


def report_max_tokens() -> int:
    return _int("REPORT_MAX_TOKENS", 14_000, 500, 64_000)


def report_min_words() -> int:
    """Depth guidance only. The model is told to earn its length, not hit it."""
    return _int("REPORT_MIN_WORDS", 3_000, 0, 100_000)


def report_max_sections() -> int:
    """Cap on generated body sections, to bound the writing calls."""
    return _int("REPORT_MAX_SECTIONS", 20, 1, 60)


def report_evidence_per_section() -> int:
    """Evidence records handed to each section writer.

    Kept selective on purpose: the top-ranked records carry the section, and
    a smaller prompt stays clear of provider token limits while forcing the
    writer to cite the strongest findings instead of padding count.
    """
    return _int("REPORT_EVIDENCE_PER_SECTION", 40, 4, 500)


def report_quality_floor() -> float:
    """Sections scoring below this on evidence support are not written."""
    return _float("REPORT_QUALITY_FLOOR", 0.0, 0.0, 1.0)


def output_pool_resume() -> bool:
    """Retry model-failed sections once with pool context before giving up.

    The retry reads the post-processed sections already in the pool so the
    fallback continues the report instead of restarting it. Completed pool
    entries are never modified by the retry.
    """
    return _bool("OUTPUT_POOL_RESUME", True)


def output_pool_context_chars() -> int:
    """Max chars of completed-section digest handed to a fallback retry."""
    return _int("OUTPUT_POOL_CONTEXT_CHARS", 4000, 0, 60_000)


# --- export ----------------------------------------------------------------

# --- throttling ------------------------------------------------------------
#
# Free-tier providers do not fail politely: one 429 can cost the rest of the
# day. Spacing calls out protects a per-minute budget by construction, and the
# concurrency ceiling stops two concurrent runs from jointly exceeding it.
#
# Turning throttling off is the right choice once you are on paid subscriptions
# with headroom; until then it trades wall-clock time for not being cut off.


def throttle_llm() -> bool:
    """Pace model calls. On by default while providers are free-tier."""
    return _bool("THROTTLE_LLM", True)


def throttle_llm_per_minute() -> float:
    """Ceiling on model calls per minute. 0 disables pacing.

    Each call waits for its own slot, so this sets the floor on how long a
    stage of N calls takes: N * 60/per_minute seconds.

    14, not higher: the lead model (Gemini free tier) enforces its own
    per-minute quota below this, and running above it does not go faster -- it
    converts calls into 429s, which burn a fallback attempt each and push the
    work onto weaker models. Measured at 30/min: three separate 429 storms in
    one run, with extraction quality visibly worse on the fallback model.
    Raise this only with a provider that has the headroom, and watch the log
    for "rate limited: 429 per-minute quota".
    """
    return _float("THROTTLE_LLM_PER_MINUTE", 14.0, 0.0, 10_000.0)


def throttle_llm_max_concurrent() -> int:
    """Model calls in flight at once.

    Was 1, which serialised every stage even though extraction batches and
    per-source synthesis calls are independent of each other. Extraction and
    synthesis now issue calls concurrently (see LLM_PARALLELISM), so this has
    to allow more than one or the pool just queues behind itself.
    """
    return _int("THROTTLE_LLM_MAX_CONCURRENT", 4, 1, 64)


def throttle_search() -> bool:
    """Pace search calls. Off by default: SerpApi is a paid, metered quota."""
    return _bool("THROTTLE_SEARCH", False)


def throttle_search_per_minute() -> float:
    return _float("THROTTLE_SEARCH_PER_MINUTE", 60.0, 0.0, 10_000.0)


def throttle_search_max_concurrent() -> int:
    """Search already fans out across workers; this caps the total."""
    return _int("THROTTLE_SEARCH_MAX_CONCURRENT", 4, 1, 64)


def export_dir() -> str:
    return os.environ.get("EXPORT_DIR", "exports")


def export_enabled() -> bool:
    return _bool("EXPORT_ENABLED", True)


# --- global budgets (Sec 36: no uncontrolled crawling) ---------------------

# Measured at 6k-char extraction batches on a broad run: up to ~35 batches
# plus synthesis groups, so each extra round costs roughly this many model
# calls, on top of a fixed plan + contradiction + report cost. Used only to
# size a default budget; an explicit RESEARCH_LLM_BUDGET always wins.
# The ceiling only bounds spend -- the pipeline uses what it needs.
LLM_FIXED_CALL_ESTIMATE = 25
LLM_CALLS_PER_ROUND_ESTIMATE = 60


def budget_estimate(rounds: int) -> int:
    """Rough number of model calls a run of this depth is expected to need."""
    return LLM_FIXED_CALL_ESTIMATE + LLM_CALLS_PER_ROUND_ESTIMATE * max(1, rounds)


def total_llm_budget(rounds: int | None = None) -> int:
    """Ceiling on model calls for one research run.

    Defaults to a value scaled to the requested depth. A flat ceiling interacts
    badly with the round limit: more rounds means more extraction, so a budget
    sized for two rounds silently starves synthesis at four and the report comes
    back partial. Set RESEARCH_LLM_BUDGET to pin it explicitly.
    """
    if os.environ.get("RESEARCH_LLM_BUDGET", "").strip():
        return _int("RESEARCH_LLM_BUDGET", 40, 4, 500)
    return budget_estimate(rounds if rounds else max_rounds())


def budget_is_explicit() -> bool:
    return bool(os.environ.get("RESEARCH_LLM_BUDGET", "").strip())


def total_fetch_budget() -> int:
    """Ceiling on documents downloaded for one research run."""
    return _int("RESEARCH_FETCH_BUDGET", 100, 0, 1000)


def wall_clock_budget_s() -> float:
    """Soft ceiling on a research run. 0 disables the check."""
    return _float("RESEARCH_TIME_BUDGET_S", 600.0, 0.0, 86_400.0)