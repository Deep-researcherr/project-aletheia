"""LLM access with per-model retries and a fallback chain across models."""
from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time

import litellm

from .config import CHARS_PER_TOKEN

try:  # litellm's provider exceptions subclass openai's hierarchy, not litellm.APIError
    from openai import OpenAIError as ProviderError
except ImportError:  # pragma: no cover - openai ships with litellm
    ProviderError = litellm.APIError

log = logging.getLogger(__name__)

# Shared anti-hallucination rule, prepended to every prompt that processes
# collected material. Deliberately short (~60 tokens): it rides along on
# calls that already send kilobytes of evidence, instead of duplicating the
# longer per-section discipline blocks.
GROUNDING = """HARD RULE: use ONLY the material given in this prompt. Never add facts,
numbers, names, or URLs from your own knowledge. If the material does not
support a point, say it is not established. An unsupported sentence is a
failure, no matter how plausible it reads."""

# Verified reachable on 2026-10-04. Any litellm model string works, but it must
# carry a provider prefix and the matching provider key must be present.
# The default mirrors LLM_MODEL so an unset env var does not silently put a
# rate-limited provider (Groq free tier) in front of the chain.
DEFAULT_MODEL = "gemini/gemini-3.5-flash"

MAX_ATTEMPTS = 2
BACKOFF_BASE_SECONDS = 1.0
CALL_TIMEOUT_SECONDS = 60
LOG_MESSAGE_CHARS = 120
LOG_MESSAGE_CHARS_RATE_LIMIT = 300
# When every model is cooling down, wait this long for the soonest if it is
# close enough that waiting beats failing outright.
WAIT_FOR_COOLDOWN_S = 10.0

# Provider failures that are worth retrying on the same model.
TRANSIENT = (
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.APIConnectionError,
    litellm.Timeout,
)


class LLMChainError(RuntimeError):
    """Every model in the chain failed."""


class CallBudget:
    """Counts pipeline LLM calls and refuses to go past its ceiling.

    Retrieval and chunking are free; model calls are not. Several stages draw
    from one budget so that a run can never fan out without limit, and so the
    expensive writing stages get guaranteed calls after the cheap ones.
    """

    def __init__(self, total: int | None = None, reserve: int = 0):
        import os as _os
        from . import config as _config
        self.total = total if total is not None else _config.total_llm_budget()
        self.reserve = reserve            # calls held back for later stages
        self.spent = 0
        self._os = _os

    @property
    def remaining(self) -> int:
        usable = self.total - self.reserve
        return max(0, usable - self.spent)

    def available(self) -> bool:
        """True when the *general* pool still has room."""
        return self.remaining > 0

    def take(self, n: int = 1) -> bool:
        """Reserve n calls. False when the general pool is exhausted."""
        if self.remaining < n:
            return False
        self.spent += n
        return True

    def spend(self, n: int = 1) -> None:
        """Record calls already made, bypassing the general-pool ceiling."""
        self.spent += n

    def snapshot(self) -> dict:
        return {"total": self.total, "spent": self.spent,
                "remaining": self.remaining, "reserve": self.reserve}


def _models(role: str) -> list[str]:
    """Role-specific model first, then the default, then LLM_FALLBACKS (comma-separated).

    Read from the environment on every call so a changed .env takes effect
    without restarting, and so tests can drive the chain.
    """
    default = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    chain = [os.getenv(f"LLM_MODEL_{role.upper()}", default), default]
    chain += [m.strip() for m in os.getenv("LLM_FALLBACKS", "").split(",") if m.strip()]
    return list(dict.fromkeys(chain))  # dedupe, keep order


def sleep(seconds: float) -> None:
    """Indirection so tests can patch out the wait."""
    time.sleep(seconds)


def _now() -> float:
    """Monotonic clock, indirected so tests can move time without waiting."""
    return time.monotonic()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def max_attempts() -> int:
    return max(1, int(_env_float("LLM_MAX_ATTEMPTS", MAX_ATTEMPTS)))


def chain_retry_wait_s() -> float:
    """Longest to wait before giving the whole chain one second pass.

    When every model failed only with temporary rate limits, one bounded wait
    for the soonest recovery beats collapsing the entire research stage. Hard
    failures (auth, oversized prompts) never trigger a retry.
    """
    return _env_float("LLM_CHAIN_RETRY_WAIT_S", 30.0)


def daily_cooldown_s() -> float:
    return _env_float("LLM_DAILY_COOLDOWN_S", 3600.0)


def per_minute_cooldown_s() -> float:
    return _env_float("LLM_PER_MINUTE_COOLDOWN_S", 60.0)


def overload_cooldown_s() -> float:
    return _env_float("LLM_OVERLOAD_COOLDOWN_S", 30.0)


def oversized_cooldown_s() -> float:
    """Cooldown after a provider rejects a request for exceeding its token limit.

    Long on purpose. When a single request is larger than the provider's whole
    per-minute window, retrying the same prompt against the same model cannot
    ever succeed, so a short cooldown only produces a repeated failure loop.
    """
    return _env_float("LLM_OVERSIZED_COOLDOWN_S", 900.0)


# --- model health ---------------------------------------------------------
#
# Shared file-backed registry so a restart, reload or second worker does not
# forget which models are cooling down or which prompt sizes they reject.
# In-memory dicts stay the hot path; the JSON file at MODEL_HEALTH_PATH
# (.cache/model_health.json by default) is the durable copy.

_health_lock = threading.Lock()
_cooldowns: dict[str, tuple[float, str]] = {}   # model -> (until_monotonic, reason)
_last_health_mtime: float | None = None


def _repo_root() -> str:
    from pathlib import Path
    return str(Path(__file__).resolve().parents[2])


def _health_path() -> str:
    from pathlib import Path
    raw = os.environ.get("MODEL_HEALTH_PATH", ".cache/model_health.json")
    path = Path(raw)
    if not path.is_absolute():
        path = Path(_repo_root()) / path
    return str(path)


def _load_health_from_disk() -> None:
    """Merge durable cooldowns/ceilings into memory if the file changed."""
    global _last_health_mtime
    import json as _json
    path = _health_path()
    try:
        import os as _os
        mtime = _os.path.getmtime(path)
    except OSError:
        return
    if _last_health_mtime is not None and mtime <= _last_health_mtime:
        return
    try:
        with open(path, encoding="utf-8") as handle:
            data = _json.load(handle)
    except Exception:
        return
    now_mono = _now()
    now_wall = time.time()
    for model, entry in (data.get("cooldowns") or {}).items():
        try:
            until_wall = float(entry.get("until_wall", 0))
            reason = str(entry.get("reason", "cooling down"))
        except Exception:
            continue
        left = until_wall - now_wall
        if left <= 0:
            continue
        current = _cooldowns.get(model)
        if current is None or current[0] < now_mono + left:
            _cooldowns[model] = (now_mono + left, reason)
    for model, tokens in (data.get("ceilings") or {}).items():
        try:
            tokens = int(tokens)
        except Exception:
            continue
        if tokens > 0 and _prompt_ceilings.get(model) != tokens:
            previous = _prompt_ceilings.get(model)
            if previous is None or tokens < previous:
                _prompt_ceilings[model] = tokens
    _last_health_mtime = mtime


def _save_health_to_disk() -> None:
    import json as _json
    path = _health_path()
    try:
        from pathlib import Path as _Path
        now_wall = time.time()
        payload = {"cooldowns": {}, "ceilings": dict(_prompt_ceilings)}
        for model, (until_mono, reason) in _cooldowns.items():
            left = until_mono - _now()
            if left <= 0:
                continue
            payload["cooldowns"][model] = {
                "until_wall": now_wall + left,
                "reason": reason,
            }
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            _json.dump(payload, handle)
        import os as _os
        _os.replace(tmp, path)
        try:
            global _last_health_mtime
            _last_health_mtime = _os.path.getmtime(path)
        except OSError:
            pass
    except Exception as exc:
        log.debug("model health persist failed: %s", exc)


def _sync_health() -> None:
    with _health_lock:
        _load_health_from_disk()


def _cool_down(model: str, seconds: float, reason: str) -> None:
    with _health_lock:
        _cooldowns[model] = (_now() + seconds, reason)
    _save_health_to_disk()


def _clear_cooldown(model: str) -> None:
    removed = False
    with _health_lock:
        removed = _cooldowns.pop(model, None) is not None
    if removed:
        _save_health_to_disk()


def _remaining(model: str) -> tuple[float, str] | None:
    """Seconds left on this model's cooldown, or None if it is usable."""
    _sync_health()
    with _health_lock:
        entry = _cooldowns.get(model)
        if entry is None:
            return None
        until, reason = entry
        left = until - _now()
        if left <= 0:
            _cooldowns.pop(model, None)
            return None
        return left, reason


# --- learned prompt ceilings ------------------------------------------------
#
# When a provider rejects a request because it is larger than the provider's own
# limit, that model cannot serve a prompt of that size, ever. Remembering the
# size means a later call can skip the model outright instead of paying a failed
# round trip to rediscover it -- which is what otherwise turns into a slow
# loop: fail, cool down, retry the same oversized prompt, fail again.

_prompt_ceilings: dict[str, int] = {}


def reset_health(clear_file: bool = True) -> None:
    """Forget every cooldown and learned ceiling. Intended for tests."""
    global _last_health_mtime
    with _health_lock:
        _cooldowns.clear()
        _prompt_ceilings.clear()
        _last_health_mtime = None
    if clear_file:
        try:
            import os as _os
            _os.remove(_health_path())
        except OSError:
            pass


def _note_ceiling(model: str, requested_tokens: int) -> None:
    if requested_tokens <= 0:
        return
    changed = False
    with _health_lock:
        previous = _prompt_ceilings.get(model)
        if previous is None or requested_tokens < previous:
            _prompt_ceilings[model] = requested_tokens
            changed = True
    if changed:
        _save_health_to_disk()


def _ceiling(model: str) -> int | None:
    _sync_health()
    with _health_lock:
        return _prompt_ceilings.get(model)


def _exceeds_ceiling(model: str, approx_tokens: int) -> bool:
    """True when this prompt is known to be too large for this model."""
    if approx_tokens <= 0:
        return False
    ceiling = _ceiling(model)
    return ceiling is not None and approx_tokens >= ceiling


# --- 429 classification ----------------------------------------------------
#
# Google's RESOURCE_EXHAUSTED bodies carry a quota id naming the window
# ("PerDay" / "PerMinute") and sometimes a retry delay, e.g.
#   "retryDelay": "34s"   or   "Please retry in 34s"
# UNVERIFIED against a live 429: the parser accepts several spellings because
# only a truncated prefix has been observed on this project.

_DAILY = re.compile(r"per[\s_-]?day|daily", re.I)
_MINUTE = re.compile(r"per[\s_-]?minute|per[\s_-]?min", re.I)
_DELAY_FIELD = re.compile(r'"retryDelay"\s*:\s*"?(\d+)\s*([smh])', re.I)
_RETRY_IN = re.compile(r"retry in\s+(\d+)\s*([smh])", re.I)
_DELAY_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0}

# A request the provider will never accept because it is larger than the limit
# itself. Distinct from a throttle: no amount of waiting fixes this one, because
# the same prompt will be the same size next time.
#   "Request too large ... TPM: Limit 8000, Requested 9838"
_OVERSIZED = re.compile(
    r"request too large|"
    r"requested\s+(\d+)\s*tokens?|"
    r"(?:limit|max(?:imum)?)\s*(?:context|tokens?)\s*(?:length|window)?\s*exceed",
    re.I)
_REQUESTED = re.compile(r"requested[:\s]+(\d+)", re.I)
_LIMIT = re.compile(r"limit[:\s]+(\d+)", re.I)


def _parse_retry_delay(message: str) -> float | None:
    for pattern in (_DELAY_FIELD, _RETRY_IN):
        match = pattern.search(message)
        if match:
            return float(match.group(1)) * _DELAY_UNITS[match.group(2).lower()]
    return None


def classify_rate_limit(message: str) -> tuple[str, float | None]:
    """Classify a provider rate-limit body.

    Returns ``(kind, retry_delay_seconds)`` where kind is one of:

    ``daily``           per-day quota exhausted; will not clear for hours
    ``request-too-large`` the request itself exceeds the provider's limit, so it
                        can never succeed on this model no matter how long we wait
    ``per-minute``      an ordinary throughput throttle that will clear

    Order matters: the oversized case must be tested before the per-minute case,
    because an oversized rejection is often *also* reported through a TPM limit.
    """
    delay = _parse_retry_delay(message)
    if _DAILY.search(message):
        return "daily", delay
    if _OVERSIZED.search(message):
        return "request-too-large", delay
    return "per-minute", delay


def oversized_request_size(message: str) -> int | None:
    """The token count the provider says the request needed, if it said so."""
    match = _REQUESTED.search(message)
    return int(match.group(1)) if match else None


def _rate_limit_summary(exc: BaseException) -> str:
    kind, delay = classify_rate_limit(str(exc))
    if kind == "request-too-large":
        requested = oversized_request_size(str(exc))
        limit = _LIMIT.search(str(exc))
        summary = "request exceeds the provider's token limit"
        if requested:
            summary += f" (requested {requested}"
            if limit:
                summary += f" > limit {limit.group(1)}"
            summary += ")"
        return summary
    summary = f"429 {kind} quota"
    if delay is not None:
        summary += f", retry in {delay:.0f}s"
    return summary


def _chain_for_call(role: str, approx_tokens: int = 0) -> tuple[list[str], list[tuple[str, str]]]:
    """The chain for this call, with unusable models removed.

    Returns ``(models, skipped)`` where ``skipped`` records ``(model, reason)``
    for every model that was dropped, so the caller can explain itself in the
    log instead of silently shortening the chain.
    """
    chain = _models(role)
    active: list[str] = []
    blocked: list[tuple[str, str]] = []

    for model in chain:
        cooling = _remaining(model)
        if cooling is not None:
            seconds, reason = cooling
            blocked.append((model, f"cooling down {seconds:.0f}s ({reason})"))
            continue
        if _exceeds_ceiling(model, approx_tokens):
            ceiling = _ceiling(model)
            blocked.append((model,
                            f"prompt ~{approx_tokens} tok exceeds its known "
                            f"limit of {ceiling} tok"))
            continue
        active.append(model)

    for model, reason in blocked:
        log.info("skipping %s (%s)", model, reason)

    if not active and blocked:
        # Everything is unusable. Rather than fail outright, try whichever model
        # recovers soonest. A learned-ceiling block does not recover within the
        # call, so it is only used when nothing is cooling down.
        recovering: list[tuple[float, str]] = []
        for model, _ in blocked:
            left = _remaining(model)
            if left is not None:
                recovering.append((left[0], model))

        if recovering:
            recovering.sort()
            seconds, model = recovering[0]
            log.info("all %d models unavailable; %s recovers in %.0fs",
                     len(blocked), model, seconds)
            if seconds <= WAIT_FOR_COOLDOWN_S:
                sleep(seconds)
        else:
            model = blocked[0][0]
            log.warning("all %d models rejected this prompt; trying %s anyway",
                        len(blocked), model)
        active.append(model)

    return active, blocked


def _backoff(attempt: int) -> float:
    """1s, 2s, 4s plus jitter."""
    return BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5)


def _describe(exc: BaseException, chars: int = LOG_MESSAGE_CHARS) -> str:
    return f"{type(exc).__name__}: {str(exc)[:chars]}"


def _retryable(exc: BaseException) -> bool:
    """True for transient provider failures and 5xx responses."""
    if isinstance(exc, TRANSIENT):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status >= 500


# ---------------------------------------------------------------------------
# Mock mode (dev/test only). Off unless LLM_MOCK=1.
# ---------------------------------------------------------------------------

_MOCK_MARKER = "followup-pass-2"
_MOCK_WARNED = False


def _mock_enabled() -> bool:
    return os.environ.get("LLM_MOCK", "") == "1"


def _mock_kind(system: str, role: str) -> str:
    """Planner and gap_check share role='default', so sniff the system prompt."""
    if role == "judge":
        return "judge"
    if role == "synth":
        return "synth"
    text = system.lower()
    if "research lead" in text:
        return "gap"
    if "checking evidence for contradictions" in text:
        return "judge"
    if "you write research reports" in text:
        return "synth"
    return "planner"


def _mock_topic(user: str) -> str:
    """The research question, for planner-style prompts."""
    for line in user.splitlines():
        if line.startswith("Question:"):
            return line.split(":", 1)[1].strip()
    return user.strip().splitlines()[0][:80] if user.strip() else "the topic"


def _mock_evidence_ids(user: str) -> list[int]:
    return [int(n) for n in re.findall(r"\[(\d+)\]", user)]


def _mock_planner(question: str) -> dict:
    topic = question.strip().rstrip("?") or "the topic"
    return {"subquestions": [
        {"question": f"What is {topic}?",
         "searches": [{"source": "web", "query": f"{topic} overview"},
                      {"source": "news", "query": f"{topic} latest news"}]},
        {"question": f"How does {topic} compare to the alternatives?",
         "searches": [{"source": "scholar", "query": f"{topic} benchmark"},
                      {"source": "web", "query": f"{topic} vs alternatives comparison"}]},
        {"question": f"What is the current state of {topic}?",
         "searches": [{"source": "github", "query": f"{topic} repository"}]},
    ]}


def _mock_gap(user: str) -> dict:
    """First round asks for more; once the follow-up results are in the evidence, stop."""
    if _MOCK_MARKER in user:
        return {"sufficient": True, "missing": "", "follow_ups": []}
    topic = _mock_topic(user).rstrip("?") or "the topic"
    return {"sufficient": False,
            "missing": "no primary source for the follow-up question",
            "follow_ups": [
                {"source": "news", "query": f"{topic} {_MOCK_MARKER}"},
                {"source": "web", "query": f"{topic} {_MOCK_MARKER} primary source"},
            ]}


def _mock_judge(user: str) -> dict:
    ids = _mock_evidence_ids(user)
    if len(ids) < 2:
        return {"contradictions": []}
    a, b = ids[0], ids[1]
    return {"contradictions": [
        {"topic": "maturity and adoption",
         "side_a": "The first source calls it widely adopted in production.",
         "sources_a": [a],
         "side_b": "The second source says adoption is still limited to experiments.",
         "sources_b": [b]},
    ]}


def _mock_synth(user: str) -> str:
    ids = _mock_evidence_ids(user)
    picks = ids[:4] or [1]
    combined = f"[{', '.join(str(n) for n in picks)}]"      # real-model style
    separate = ", ".join(f"[{n}]" for n in picks[:2])        # legacy style
    topic = _mock_topic(user) or "the topic"
    return (
        f"## Short answer\n"
        f"Based on the gathered evidence, {topic} is best approached incrementally: "
        f"start with the well-sourced basics, validate against the documented "
        f"limitations, and only then commit to the more speculative directions. "
        f"I recommend treating it as a good default rather than a universal answer "
        f"{separate}.\n\n"
        f"## Key findings\n"
        f"* The primary documentation describes the core mechanics clearly {combined}.\n"
        f"* Independent write-ups agree on the fundamentals {combined}.\n"
        f"* Recent material suggests the ecosystem is still moving {combined}.\n\n"
        f"## Where sources disagree\n"
        f"One disagreement was detected between the collected sources: they differ on "
        f"how mature the ecosystem is. Treat maturity claims as unsettled.\n\n"
        f"## Freshness & caveats\n"
        f"Some sources are older than the freshness threshold and are marked STALE in "
        f"the evidence list. The mock dataset is synthetic, so these notes exist only "
        f"to exercise the report pipeline end to end."
    )


def _mock_reply(kind: str, user: str, json_mode: bool) -> object:
    if kind == "gap":
        return _mock_gap(user)
    if kind == "judge":
        return _mock_judge(user)
    if kind == "synth":
        return _mock_synth(user)
    return _mock_planner(user)


def _mock(kind: str, user: str, json_mode: bool):
    global _MOCK_WARNED

    if not _MOCK_WARNED:
        _MOCK_WARNED = True
        log.warning("LLM MOCK MODE ACTIVE: no real model calls")

    failing = os.environ.get("LLM_MOCK_FAIL", "").strip()
    if failing and (failing == "all" or failing == kind):
        raise LLMChainError(f"LLM mock failure for role: {kind}")

    delay_ms = os.environ.get("LLM_MOCK_DELAY_MS", "400")
    try:
        delay = max(0.0, float(delay_ms) / 1000.0)
    except ValueError:
        delay = 0.4
    if delay:
        sleep(delay)

    return _mock_reply(kind, user, json_mode)


def _loads_json(text: str):
    """Parse model JSON, salvaging common model-output defects.

    Models wrap payloads in prose, trail commas, or get cut off mid-stream.
    Each repair is tried in order; if nothing parses, the last decoder error
    propagates so callers never receive a half-object silently.
    """
    candidates = [text or ""]
    start, end = candidates[0].find("{"), candidates[0].rfind("}")
    if 0 <= start < end:
        span = candidates[0][start:end + 1]
        candidates.append(span)
        candidates.append(re.sub(r",\s*([}\]])", r"\1", span))
    last: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last = exc
    assert last is not None
    raise last


def _clean_model_output(text: str | None) -> str:
    """Strip reasoning blocks some models prepend to their reply.

    Thinking/reasoning models (Qwen, gpt-oss family) wrap chain-of-thought
    in <think> tags. Left in place it pollutes reports and breaks JSON
    parsing; it is never part of the answer. Non-string input passes
    through untouched.
    """
    if not isinstance(text, str):
        return text
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return re.sub(r"<think>.*?</think>", "", text,
                  flags=re.DOTALL | re.IGNORECASE).strip()


def _approx_tokens(system: str, user: str) -> int:
    """Cheap prompt-size estimate, used only to compare against learned limits."""
    return int((len(system) + len(user)) / CHARS_PER_TOKEN)


def _pacer():
    """The shared throttle, resolved lazily so import order never matters."""
    from .throttle import get
    return get("llm")


def _chain_retry_delay(chain: list[str], approx_tokens: int) -> float | None:
    """Seconds until the soonest cooling model recovers, or None to give up.

    Returns None when retrying is futile: a model with a learned oversized
    ceiling would reject the same prompt again, and an empty chain has
    nothing to retry with.
    """
    if not chain:
        return None
    soonest: float | None = None
    for model in chain:
        if _exceeds_ceiling(model, approx_tokens):
            return None
        remaining = _remaining(model)
        if remaining is None:
            return 0.0
        left, _ = remaining
        soonest = left if soonest is None else min(soonest, left)
    return soonest if soonest is not None else 0.0


def _advance_notice(model: str, index: int, total: int,
                    chain: list[str], reason: str) -> str:
    position = index + 1
    if index + 1 < len(chain):
        return (f"{model} -> {reason} -> trying fallback {position + 1}/{total}: "
                f"{chain[index + 1]}")
    return f"{model} -> {reason} -> no fallback models left in the chain"


def ask(system: str, user: str, json_mode: bool = False, role: str = "default",
        _retried: bool = False):
    if _mock_enabled():
        return _mock(_mock_kind(system, role), user, json_mode)

    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    approx = _approx_tokens(system, user)
    chain, skipped = _chain_for_call(role, approx)
    attempts = max_attempts()
    last: BaseException | None = None
    # How each model in this pass failed. Only a pass where EVERY failure was
    # a temporary rate limit earns one more pass after a bounded wait; auth
    # rejections, oversized prompts and programming errors fail fast instead
    # of burning quota on a doomed retry.
    fail_kinds: list[str] = []

    if skipped:
        log.debug("model chain for role %s: %d available, %d skipped",
                  role, len(chain), len(skipped))

    for index, model in enumerate(chain):
        total = len(chain)

        for attempt in range(1, attempts + 1):
            try:
                # Throttled around the call itself, not the retries: pacing
                # applies to what the provider sees.
                with _pacer().slot():
                    r = litellm.completion(
                        model=model,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                        timeout=CALL_TIMEOUT_SECONDS,
                        num_retries=0,  # retries are handled here, not by litellm
                        **kw,
                    )
                out = _clean_model_output(r.choices[0].message.content)
                _clear_cooldown(model)
                if not json_mode:
                    return out
                try:
                    return _loads_json(out)
                except json.JSONDecodeError as exc:
                    # Unparsable output is a per-model sampling failure: move
                    # to the next model rather than retrying the same prompt
                    # or collapsing the stage on a raw decoder traceback.
                    last = exc
                    notice = _advance_notice(model, index, total, chain,
                                             "returned invalid JSON")
                    log.warning("llm %s | attempt %d/%d | %s", model, attempt,
                                attempts, notice)
                    fail_kinds.append("hard")
                    break

            except litellm.RateLimitError as exc:
                # Never retry a rate-limited model: the same request would fail
                # the same way. Park it for a reason-appropriate period and move
                # to the next configured model immediately.
                last = exc
                kind, delay = classify_rate_limit(str(exc))
                requested = None

                if kind == "request-too-large":
                    requested = oversized_request_size(str(exc))
                    _note_ceiling(model, requested or approx)
                    seconds, reason = oversized_cooldown_s(), "request too large"
                    fail_kinds.append("hard")
                elif kind == "daily":
                    seconds, reason = daily_cooldown_s(), "per-day quota"
                    fail_kinds.append("temp")
                else:
                    seconds, reason = (delay or per_minute_cooldown_s()), "per-minute quota"
                    fail_kinds.append("temp")

                _cool_down(model, seconds, reason)
                notice = _advance_notice(model, index, total, chain,
                                         f"rate limited: {_rate_limit_summary(exc)}")
                log.warning("llm %s | attempt %d/%d | %s", model, attempt, attempts, notice)
                if requested:
                    log.warning("  %s: cooling down %.0fs; this model cannot serve "
                                "a ~%d token prompt", model, seconds, requested)
                else:
                    log.warning("  %s: cooling down %.0fs", model, seconds)
                break

            except ProviderError as exc:
                last = exc
                if _retryable(exc) and attempt < attempts:
                    log.warning("llm %s | attempt %d/%d | %s | retrying same model",
                                model, attempt, attempts, _describe(exc))
                    sleep(_backoff(attempt))
                    continue
                if _retryable(exc):
                    # Overloaded across every attempt: park it briefly so the
                    # next pipeline stage does not walk straight back into it.
                    _cool_down(model, overload_cooldown_s(), "overloaded")
                    notice = _advance_notice(model, index, total, chain,
                                             "overloaded after all attempts")
                    log.warning("llm %s | %s | cooling down %.0fs",
                                model, notice, overload_cooldown_s())
                    fail_kinds.append("temp")
                else:
                    notice = _advance_notice(model, index, total, chain,
                                             f"rejected: {type(exc).__name__}")
                    log.warning("llm %s | attempt %d/%d | %s", model, attempt,
                                attempts, notice)
                    fail_kinds.append("hard")
                break

            # Non-provider exceptions (TypeError, KeyError, ...) propagate on purpose.

    if (not _retried and fail_kinds
            and all(kind == "temp" for kind in fail_kinds)):
        delay = _chain_retry_delay(chain, approx)
        cap = chain_retry_wait_s()
        if delay is not None and delay <= cap:
            log.warning("llm chain exhausted by temporary rate limits; "
                        "waiting %.0fs for recovery, then one final pass", delay)
            sleep(delay)
            return ask(system, user, json_mode, role, _retried=True)
        log.warning("llm chain exhausted by temporary rate limits; soonest "
                    "recovery in %s (cap %.0fs): failing instead of waiting",
                    f"{delay:.0f}s" if delay is not None else "unknown", cap)

    detail = "; ".join(f"{m} ({r})" for m, r in skipped) or "none"
    attempted = ", ".join(chain)
    raise LLMChainError(
        f"All {len(chain)} models failed (attempted: {attempted}); "
        f"last error {_describe(last, LOG_MESSAGE_CHARS_RATE_LIMIT)}. "
        f"Skipped before this call: {detail}"
    ) from last