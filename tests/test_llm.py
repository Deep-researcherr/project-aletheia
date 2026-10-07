"""Retry, fallback, and model-health behaviour for backend.research.llm.ask."""

import os
import threading
from unittest.mock import patch

import pytest
import litellm

from backend.research import llm


class FakeResponse:
    def __init__(self, content):
        message = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": message})()]


class Clock:
    """A monotonic clock the tests move by hand."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture(autouse=True)
def world(monkeypatch):
    """Never sleep for real, and never let health state leak between cases.

    Autouse, but also requestable as `world` by tests that need the clock or
    the recorded sleeps.
    """
    slept = []
    clock = Clock()
    monkeypatch.setattr(llm, "sleep", lambda seconds: slept.append(seconds))

    monkeypatch.setattr(llm, "_now", clock)

    llm.reset_health()

    for name in (
        "LLM_MODEL",
        "LLM_FALLBACKS",
        "LLM_MAX_ATTEMPTS",
        "LLM_DAILY_COOLDOWN_S",
        "LLM_PER_MINUTE_COOLDOWN_S",
        "LLM_OVERLOAD_COOLDOWN_S",
        "LLM_MODEL_SYNTH",
        "LLM_MODEL_JUDGE",
    ):
        monkeypatch.delenv(name, raising=False)

    return clock, slept


def _service_unavailable():
    return litellm.ServiceUnavailableError("503 high demand", None, None)


def _rate_limited(message):
    return litellm.RateLimitError(message, None, None)


def _unauthorized():
    return litellm.AuthenticationError("401 bad key", None, None)


def _chain(*models):
    os.environ["LLM_MODEL"] = models[0]
    os.environ["LLM_FALLBACKS"] = ",".join(models[1:])


def completing(fake):
    """Run ask() with litellm.completion replaced."""
    return patch.object(litellm, "completion", fake)


# --- retry / fallback ------------------------------------------------------


def test_transient_503_is_retried_on_the_same_model():
    calls = []
    errors = [_service_unavailable(), _service_unavailable()]

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if errors:
            raise errors.pop(0)
        return FakeResponse("recovered")

    _chain("gemini/gemini-3.8-flash")
    os.environ["LLM_MAX_ATTEMPTS"] = "3"

    with completing(fake_completion):
        assert llm.ask("sys", "user") == "recovered"

    assert calls == ["gemini/gemini-3.8-flash"] * 3


def test_rate_limit_moves_to_next_model_without_retrying():
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "gemini/gemini-3.8-flash":
            raise _rate_limited("You exceeded your current quota")
        return FakeResponse("second model answered")

    _chain("gemini/gemini-3.8-flash", "gemini/gemini-3.7-flash")

    with completing(fake_completion):
        assert llm.ask("sys", "user") == "second model answered"

    assert calls == ["gemini/gemini-3.8-flash", "gemini/gemini-3.7-flash"]


def test_authentication_error_moves_to_next_model_immediately():
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "gemini/gemini-3.8-flash":
            raise _unauthorized()
        return FakeResponse("fell back")

    _chain("gemini/gemini-3.8-flash", "gemini/gemini-3.7-flash")

    with completing(fake_completion):
        assert llm.ask("sys", "user") == "fell back"

    assert calls == ["gemini/gemini-3.8-flash", "gemini/gemini-3.7-flash"]


def test_programming_errors_are_not_swallowed():
    def fake_completion(**kwargs):
        raise TypeError("result shape is wrong")

    _chain("gemini/gemini-3.8-flash", "gemini/gemini-3.7-flash")

    with completing(fake_completion):
        with pytest.raises(TypeError, match="result shape is wrong"):
            llm.ask("sys", "user")


def test_litellm_internal_retries_are_disabled():
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return FakeResponse("ok")

    _chain("gemini/gemini-3.8-flash")

    with completing(fake_completion):
        llm.ask("sys", "user")

    assert seen["num_retries"] == 0
    assert seen["timeout"] == llm.CALL_TIMEOUT_SECONDS
    assert "temperature" not in seen


# --- 429 classification ----------------------------------------------------

DAILY_MESSAGE = (
    'litellm.RateLimitError: GeminiException - {"error": {"code": 429, '
    '"message": "You exceeded your current quota, please check your plan and '
    'billing details.", "status": "RESOURCE_EXHAUSTED", '
    '"details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", '
    '"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}]}}'
)

PER_MINUTE_MESSAGE = (
    'litellm.RateLimitError: GeminiException - {"error": {"code": 429, '
    '"message": "Resource has been exhausted (e.g. check quota).", '
    '"status": "RESOURCE_EXHAUSTED", "details": [{"retryDelay": "34s"}]}}'
)

PER_MINUTE_PROSE = (
    "litellm.RateLimitError: GeminiException - You exceeded your current quota. "
    "Please retry in 34s."
)

PER_MINUTE_NO_DELAY = (
    'litellm.RateLimitError: GeminiException - {"error": {"code": 429, '
    '"message": "Resource exhausted.", "status": "RESOURCE_EXHAUSTED"}}'
)


def test_classify_daily_quota():
    assert llm.classify_rate_limit(DAILY_MESSAGE)[0] == "daily"


def test_classify_per_minute_with_retry_delay_field():
    kind, delay = llm.classify_rate_limit(PER_MINUTE_MESSAGE)
    assert kind == "per-minute"
    assert delay == 34.0


def test_classify_per_minute_with_prose_retry_hint():
    kind, delay = llm.classify_rate_limit(PER_MINUTE_PROSE)
    assert kind == "per-minute"
    assert delay == 34.0


def test_classify_per_minute_without_delay():
    kind, delay = llm.classify_rate_limit(PER_MINUTE_NO_DELAY)
    assert kind == "per-minute"
    assert delay is None


def test_rate_limit_summary_is_short_and_specific():
    assert "daily" in llm._rate_limit_summary(_rate_limited(DAILY_MESSAGE))
    summary = llm._rate_limit_summary(_rate_limited(PER_MINUTE_MESSAGE))
    assert "per-minute" in summary
    assert "34s" in summary


# --- health tracking -------------------------------------------------------


def test_daily_429_skips_that_model_for_later_calls(world):
    clock, _ = world
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "gemini/gemini-3.6-flash":
            raise _rate_limited(DAILY_MESSAGE)
        return FakeResponse("answered by fallback")

    _chain("gemini/gemini-3.6-flash", "gemini/gemini-3.7-flash")
    os.environ["LLM_DAILY_COOLDOWN_S"] = "3600"

    with completing(fake_completion):
        llm.ask("sys", "user")
        assert calls.count("gemini/gemini-3.6-flash") == 1

        clock.advance(60)
        calls.clear()
        llm.ask("sys", "user")

    assert "gemini/gemini-3.6-flash" not in calls, (
        "a daily-quota model must be skipped by the next stage"
    )
    assert calls == ["gemini/gemini-3.7-flash"]


def test_per_minute_429_skips_for_the_retry_delay_then_returns(world):
    clock, _ = world
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        first_call = calls.count("gemini/gemini-3.6-flash") == 1
        if kwargs["model"] == "gemini/gemini-3.6-flash" and first_call:
            raise _rate_limited(PER_MINUTE_MESSAGE)
        return FakeResponse("answered")

    _chain("gemini/gemini-3.6-flash", "gemini/gemini-3.7-flash")

    with completing(fake_completion):
        llm.ask("sys", "user")

        clock.advance(33)
        calls.clear()
        llm.ask("sys", "user")
        assert "gemini/gemini-3.6-flash" not in calls

        clock.advance(2)          # past the 34s delay
        calls.clear()
        llm.ask("sys", "user")

    assert calls[0] == "gemini/gemini-3.6-flash", (
        "the model should be retried once its cooldown expires"
    )


def test_503_exhaustion_parks_the_model(world):
    clock, _ = world
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "gemini/gemini-3.6-flash":
            raise _service_unavailable()
        return FakeResponse("answered by the healthy model")

    _chain("gemini/gemini-3.6-flash", "gemini/gemini-3.7-flash")
    os.environ["LLM_OVERLOAD_COOLDOWN_S"] = "30"

    with completing(fake_completion):
        assert llm.ask("sys", "user") == "answered by the healthy model"

        parked = llm._remaining("gemini/gemini-3.6-flash")
        assert parked is not None
        assert parked[0] == pytest.approx(30.0, abs=1)
        assert parked[1] == "overloaded"

        # the next pipeline stage must walk straight past the parked model
        calls.clear()
        llm.ask("sys", "user")

    assert calls == ["gemini/gemini-3.7-flash"], (
        "an overloaded model must be skipped while another is available"
    )


def test_all_models_cooling_down_uses_the_soonest_instead_of_failing(world):
    clock, slept = world
    calls = []

    def always_daily(**kwargs):
        calls.append(kwargs["model"])
        raise _rate_limited(DAILY_MESSAGE)

    _chain("gemini/gemini-3.6-flash", "gemini/gemini-3.7-flash")
    os.environ["LLM_DAILY_COOLDOWN_S"] = "3600"

    with completing(always_daily):
        with pytest.raises(llm.LLMChainError):
            llm.ask("sys", "user")

        # both models are now cooling down
        assert llm._remaining("gemini/gemini-3.6-flash") is not None
        assert llm._remaining("gemini/gemini-3.7-flash") is not None

        calls.clear()
        # give one model a much shorter cooldown so it is clearly soonest
        llm._cool_down("gemini/gemini-3.7-flash", 5.0, "per-day quota")

        def now_answers(**kwargs):
            calls.append(kwargs["model"])
            return FakeResponse("answered by the soonest model")

        with completing(now_answers):
            answer = llm.ask("sys", "user")

    assert answer == "answered by the soonest model"
    assert calls == ["gemini/gemini-3.7-flash"], "the soonest model should be used"
    assert slept, "a cooldown under 10s should be waited out rather than retried blind"


def test_success_clears_the_cooldown(world):
    clock, _ = world
    state = {"fail": True}

    def fake_completion(**kwargs):
        if state["fail"]:
            raise _rate_limited(DAILY_MESSAGE)
        return FakeResponse("recovered")

    _chain("gemini/gemini-3.6-flash")
    os.environ["LLM_DAILY_COOLDOWN_S"] = "3600"

    with completing(fake_completion):
        with pytest.raises(llm.LLMChainError):
            llm.ask("sys", "user")

        assert llm._remaining("gemini/gemini-3.6-flash") is not None

        clock.advance(3601)
        state["fail"] = False
        llm.ask("sys", "user")

        assert llm._remaining("gemini/gemini-3.6-flash") is None

# --- provider token-limit routing (Groq TPM / request-too-large) -------------
#
# Groq answers two different conditions with HTTP 429, which litellm surfaces as
# RateLimitError. They need different handling:
#
#   "Rate limit reached ... TPM: Limit 8000"                  a throttle; clears
#   "Request too large ... TPM: Limit 8000, Requested 9838"   never clears for
#                                                              this prompt

GROQ_TPM = ("Rate limit reached for groq model. Please try again later. "
            "rate_limit_error: Rate limit reached ... tokens per minute (TPM): "
            "Limit 8000")

GROQ_TOO_LARGE = ("Rate limit reached for groq model: Request too large for groq "
                  "model. TPM: Limit 8000, Requested 9838")


def test_groq_tpm_rate_limit_advances_to_the_next_fallback():
    """Requirement 1 + 4: a 429 must not be retried on the same model."""
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "groq/openai/gpt-oss-120b":
            raise _rate_limited(GROQ_TPM)
        return FakeResponse("fallback answered")

    _chain("groq/openai/gpt-oss-120b", "openrouter/google/gemma-3-27b-it")

    with completing(fake_completion):
        assert llm.ask("sys", "user") == "fallback answered"

    # Exactly one attempt against groq: no retry, straight to the fallback.
    assert calls == ["groq/openai/gpt-oss-120b",
                     "openrouter/google/gemma-3-27b-it"]


def test_request_too_large_advances_to_the_next_fallback():
    """Requirement 2: an oversized request is not a throttle; move on."""
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "groq/openai/gpt-oss-120b":
            raise _rate_limited(GROQ_TOO_LARGE)
        return FakeResponse("bigger context model answered")

    _chain("groq/openai/gpt-oss-120b", "openrouter/google/gemma-3-27b-it")

    with completing(fake_completion):
        assert llm.ask("sys", "user") == "bigger context model answered"

    assert calls == ["groq/openai/gpt-oss-120b",
                     "openrouter/google/gemma-3-27b-it"]


def test_oversized_request_gets_a_long_cooldown_not_a_throttle_cooldown():
    """The two 429 shapes must not be treated identically.

    A per-minute cooldown would put groq back in front of the chain after 60s
    for the very same oversized prompt, producing a slow failure loop.
    """
    def fake_completion(**kwargs):
        raise _rate_limited(GROQ_TOO_LARGE)

    _chain("groq/openai/gpt-oss-120b", "openrouter/google/gemma-3-27b-it")

    with completing(fake_completion):
        with pytest.raises(llm.LLMChainError):
            llm.ask("sys", "user")

    assert llm._remaining("groq/openai/gpt-oss-120b")[0] >= 900


def test_transient_failure_still_retries_the_same_model():
    """Requirement 3: genuine transients keep their retry behaviour."""
    calls = []
    errors = [_service_unavailable(), _service_unavailable()]

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if errors:
            raise errors.pop(0)
        return FakeResponse("recovered")

    _chain("groq/openai/gpt-oss-120b", "openrouter/google/gemma-3-27b-it")
    os.environ["LLM_MAX_ATTEMPTS"] = "3"

    with completing(fake_completion):
        assert llm.ask("sys", "user") == "recovered"

    assert calls == ["groq/openai/gpt-oss-120b"] * 3


def test_all_models_failing_raises_chain_error_naming_the_problem():
    """Requirement 5: the existing terminal behaviour is preserved."""
    def fake_completion(**kwargs):
        raise _rate_limited(GROQ_TOO_LARGE)

    _chain("groq/openai/gpt-oss-120b", "openrouter/google/gemma-3-27b-it")

    with completing(fake_completion):
        with pytest.raises(llm.LLMChainError) as info:
            llm.ask("sys", "user")

    message = str(info.value)
    assert "All 2 models failed" in message
    assert "groq/openai/gpt-oss-120b" in message


def test_rate_limited_model_is_not_retried_by_the_next_call(world):
    """The cooldown has to survive between calls, not just within one.

    Two sequential ask() calls are enough to prove the second one skips groq
    entirely rather than re-paying a failed round trip.
    """
    calls = []
    fail_groq = {"on": True}

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "groq/openai/gpt-oss-120b" and fail_groq["on"]:
            raise _rate_limited(GROQ_TPM)
        return FakeResponse("ok")

    _chain("groq/openai/gpt-oss-120b", "openrouter/google/gemma-3-27b-it")

    with completing(fake_completion):
        llm.ask("sys", "user")
        llm.ask("sys", "user")

    # groq is tried once overall, not once per call.
    assert calls.count("groq/openai/gpt-oss-120b") == 1
    assert calls == ["groq/openai/gpt-oss-120b",
                     "openrouter/google/gemma-3-27b-it",
                     "openrouter/google/gemma-3-27b-it"]


def test_concurrent_calls_do_not_all_hammer_a_rate_limited_model():
    """Requirement 6: parallel callers share one cooldown.

    ask() runs in worker threads (asyncio.to_thread in the graph), so the health
    table has to keep one rate-limited model out of everyone's chain.
    """
    from concurrent.futures import ThreadPoolExecutor

    calls = []
    lock = threading.Lock()

    def fake_completion(**kwargs):
        with lock:
            calls.append(kwargs["model"])
        if kwargs["model"] == "groq/openai/gpt-oss-120b":
            raise _rate_limited(GROQ_TPM)
        return FakeResponse("ok")

    _chain("groq/openai/gpt-oss-120b", "openrouter/google/gemma-3-27b-it")

    with completing(fake_completion):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: llm.ask("sys", "user"), range(8)))

    assert results == ["ok"] * 8
    # One groq attempt across eight concurrent callers, not eight.
    assert calls.count("groq/openai/gpt-oss-120b") == 1


def test_oversized_model_is_skipped_for_large_prompts_but_kept_for_small_ones(world):
    """The learned ceiling stops the re-probe loop without sidelining groq.

    After the cooldown lapses, groq is usable again for prompts small enough to
    fit it, and still avoided for prompts at or above the size it rejected.
    """
    clock, _ = world
    big = "x" * (40000 * 4)          # ~40k tokens
    small = "x" * 100

    def fake_completion(**kwargs):
        if kwargs["model"] == "groq/openai/gpt-oss-120b":
            raise _rate_limited(GROQ_TOO_LARGE)
        return FakeResponse("ok")

    _chain("groq/openai/gpt-oss-120b", "openrouter/google/gemma-3-27b-it")

    with completing(fake_completion):
        # The fallback succeeds, which is the point: an oversized prompt on the
        # primary must not fail the call.
        assert llm.ask("sys", big) == "ok"

        # While the cooldown is live groq is out for any prompt size.
        models, _ = llm._chain_for_call("default", llm._approx_tokens("sys", small))
        assert "groq/openai/gpt-oss-120b" not in models

        # Once the cooldown lapses only the ceiling still excludes it.
        clock.advance(901)
        models, skipped = llm._chain_for_call("default", llm._approx_tokens("sys", big))
        assert "groq/openai/gpt-oss-120b" not in models
        assert any("exceeds its known limit" in reason for _, reason in skipped)

        models, skipped = llm._chain_for_call("default", llm._approx_tokens("sys", small))
        assert "groq/openai/gpt-oss-120b" in models
        assert not skipped


def test_classification_separates_the_three_rate_limit_kinds():
    assert llm.classify_rate_limit(GROQ_TOO_LARGE)[0] == "request-too-large"
    assert llm.classify_rate_limit(GROQ_TPM)[0] == "per-minute"
    assert llm.classify_rate_limit("Quota exceeded for PerDay")[0] == "daily"
    assert llm.oversized_request_size(GROQ_TOO_LARGE) == 9838
    assert llm.oversized_request_size(GROQ_TPM) is None


def test_advance_notice_names_the_fallback_and_its_position():
    chain = ["groq/a", "groq/b", "groq/c"]
    notice = llm._advance_notice("groq/a", 0, 3, chain, "rate limited")
    assert notice == "groq/a -> rate limited -> trying fallback 2/3: groq/b"
    assert "no fallback models left" in llm._advance_notice("groq/c", 2, 3, chain, "x")


# --- malformed model JSON ----------------------------------------------------
#
# Models routinely return prose-wrapped, trailing-comma, or truncated JSON.
# ask() salvages what it can for free, moves to the next model on failure,
# and surfaces LLMChainError (never a raw decoder traceback) when all fail.


def test_prose_wrapped_json_is_salvaged():
    def fake_completion(**kwargs):
        return FakeResponse('Here is the plan:\n{"subquestions": []}\nDone.')

    _chain("gemini/gemini-3.6-flash")

    with completing(fake_completion):
        assert llm.ask("sys", "user", json_mode=True) == {"subquestions": []}


def test_trailing_comma_json_is_salvaged():
    def fake_completion(**kwargs):
        return FakeResponse('{"a": 1, "b": [2, 3,],}')

    _chain("gemini/gemini-3.6-flash")

    with completing(fake_completion):
        assert llm.ask("sys", "user", json_mode=True) == {"a": 1, "b": [2, 3]}


def test_garbage_json_moves_to_next_model():
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "gemini/gemini-3.6-flash":
            return FakeResponse("not json at all {{{")
        return FakeResponse('{"ok": true}')

    _chain("gemini/gemini-3.6-flash", "gemini/gemini-3.7-flash")

    with completing(fake_completion):
        assert llm.ask("sys", "user", json_mode=True) == {"ok": True}

    assert calls == ["gemini/gemini-3.6-flash", "gemini/gemini-3.7-flash"]


def test_all_garbage_json_raises_chain_error():
    def fake_completion(**kwargs):
        return FakeResponse("{{{ definitely not json")

    _chain("gemini/gemini-3.6-flash")

    with completing(fake_completion):
        with pytest.raises(llm.LLMChainError):
            llm.ask("sys", "user", json_mode=True)


def test_think_blocks_are_stripped_before_parsing():
    def fake_completion(**kwargs):
        return FakeResponse("<think>planning the answer…</think>\n"
                            '{"ok": true}')

    _chain("gemini/gemini-3.6-flash")

    with completing(fake_completion):
        assert llm.ask("sys", "user", json_mode=True) == {"ok": True}


def test_think_blocks_are_stripped_from_prose():
    def fake_completion(**kwargs):
        return FakeResponse("<THINK>draft</THINK>final answer")

    _chain("gemini/gemini-3.6-flash")

    with completing(fake_completion):
        assert llm.ask("sys", "user") == "final answer"


# --- chain-level retry after temporary rate limits ---------------------------
#
# When every model in the chain failed only with temporary quotas, one bounded
# wait plus a single second pass beats collapsing the whole research stage.
# Anything else (auth, oversized prompts) still fails fast.


def test_all_temp_failures_earn_one_bounded_retry(world, monkeypatch):
    clock, slept = world
    monkeypatch.setenv("LLM_CHAIN_RETRY_WAIT_S", "90")
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _rate_limited(GROQ_TPM)
        return FakeResponse("recovered on second pass")

    _chain("groq/openai/gpt-oss-120b", "openrouter/google/gemma-3-27b-it")

    with completing(fake_completion):
        assert llm.ask("sys", "user") == "recovered on second pass"

    assert calls["n"] == 3, "two failures, one wait, one successful retry"
    assert 60 in slept, "the per-minute park must be waited out exactly once"


def test_auth_failure_fails_fast_without_chain_retry(world, monkeypatch):
    clock, slept = world
    monkeypatch.setenv("LLM_CHAIN_RETRY_WAIT_S", "900")

    def fake_completion(**kwargs):
        if kwargs["model"] == "gemini/gemini-3.6-flash":
            raise _unauthorized()
        raise _rate_limited(GROQ_TPM)

    _chain("gemini/gemini-3.6-flash", "gemini/gemini-3.7-flash")

    with completing(fake_completion):
        with pytest.raises(llm.LLMChainError):
            llm.ask("sys", "user")

    assert slept == [], "a hard rejection must never trigger the retry wait"


def test_retry_cap_exceeded_raises_instead_of_waiting(world, monkeypatch):
    clock, slept = world
    monkeypatch.setenv("LLM_CHAIN_RETRY_WAIT_S", "5")

    def fake_completion(**kwargs):
        raise _rate_limited(GROQ_TPM)

    _chain("groq/openai/gpt-oss-120b", "openrouter/google/gemma-3-27b-it")

    with completing(fake_completion):
        with pytest.raises(llm.LLMChainError):
            llm.ask("sys", "user")

    assert slept == [], "a wait past the cap must not be paid"


def test_auth_failure_parks_nothing(world):
    clock, _ = world

    def fake_completion(**kwargs):
        raise _unauthorized()

    _chain("gemini/gemini-3.6-flash")

    with completing(fake_completion):
        with pytest.raises(llm.LLMChainError):
            llm.ask("sys", "user")

    assert llm._remaining("gemini/gemini-3.6-flash") is None, \
        "a config error must not consume cooldown state"
