"""Which paid services this run depends on, and what is left of each.

Asked for at the top of a run: before spending anything, say what is wired up
and what budget remains. Two rules shape this module.

**It never affects the run.** Every provider call is wrapped, timed out, and
allowed to fail. A dashboard that can break a research pipeline is worse than
no dashboard, so an unreachable provider shows "unknown" rather than raising.

**It reports once, not continuously.** The remote lookups are cached for
``CREDIT_TTL_S`` so repeated runs do not hammer five provider APIs (and do not
risk tripping their own rate limits just to print a table). The terminal sees
one block per run.

What can and cannot be reported is worth being precise about:

* SerpApi, Tavily and Firecrawl expose real balance endpoints, verified
  against their live APIs.
* OpenRouter exposes account credits.
* Most free-tier model providers (Gemini, Groq, Cohere, HuggingFace) publish no
  balance API at all -- they are governed by per-minute rate limits, not
  credits. Those are listed as configured, with their plan-usage cap we enforce
  locally, and are explicitly marked as having no balance API. Inventing a
  number there would be worse than saying "unknown".
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger(__name__)

# Remote balance lookups are cached this long. Long enough that a handful of
# runs in a row does not re-query every provider, short enough that topping up
# credits mid-session shows up without a restart.
CREDIT_TTL_S = 300.0

# Per-provider wall-clock ceiling. The whole point of this is to be fast; a
# provider that hangs must not delay the run that asked about it.
PROBE_TIMEOUT_S = 6.0

_lock = threading.Lock()
_cache: dict[str, tuple[float, dict]] = {}


class ServiceStatus:
    """One row of the dashboard."""

    __slots__ = ("name", "purpose", "configured", "balance", "limit",
                 "detail", "error", "details")

    def __init__(self, name: str, purpose: str, configured: bool = False,
                 balance: str = "", limit: str = "", detail: str = "",
                 error: str = "", details: list | None = None):
        self.name = name
        self.purpose = purpose
        self.configured = configured
        self.balance = balance
        self.limit = limit
        self.detail = detail
        self.error = error
        self.details = list(details or [])

    def as_dict(self) -> dict:
        return {"name": self.name, "purpose": self.purpose,
                "configured": self.configured, "balance": self.balance,
                "limit": self.limit, "detail": self.detail,
                "details": list(self.details), "error": self.error}

    def line(self) -> str:
        if not self.configured:
            state = "not configured"
        elif self.error:
            state = "balance unknown"
        elif self.balance:
            state = self.balance
        else:
            state = "no balance API"
        limit = f"   limit {self.limit}" if self.limit else ""
        return f"{state}{limit}"


def _key_hint(env_var: str) -> str:
    """A masked key, so it is visible which credential is loaded without leaking it.

    Last four characters only: enough to tell two keys apart when the wrong one
    is in the environment, useless to anyone reading over your shoulder.
    """
    raw = (os.environ.get(env_var) or "").strip()
    return f"...{raw[-4:]}" if len(raw) > 4 else "set"


# --- provider probes -------------------------------------------------------

def _cached(name: str, probe):
    """Run ``probe`` at most once per TTL. Never raises."""
    now = time.monotonic()
    with _lock:
        hit = _cache.get(name)
        if hit and now - hit[0] < CREDIT_TTL_S:
            return dict(hit[1])
    try:
        value = probe()
    except Exception as exc:
        value = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    with _lock:
        _cache[name] = (now, dict(value))
    return dict(value)


def _probe_serpapi() -> dict:
    key = os.environ.get("SERPAPI_KEY", "")
    if not key:
        return {}
    import httpx
    r = httpx.get("https://serpapi.com/account",
                 params={"api_key": key}, timeout=PROBE_TIMEOUT_S)
    d = r.json()
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}"}
    left = d.get("total_searches_left")
    total = d.get("searches_per_month")
    status = str(d.get("account_status") or "")
    out = {"balance": f"{left}/{total} searches left" if left is not None else "",
           "detail": status}
    # "ran out of searches" is the single most useful thing on this screen, and
    # it is not visible from the numbers alone.
    if "out of searches" in status.lower() or (left == 0):
        out["detail"] = status or "exhausted"
    return out


def _probe_tavily() -> dict:
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        return {}
    import httpx
    r = httpx.get("https://api.tavily.com/usage",
                 headers={"Authorization": f"Bearer {key}"},
                 timeout=PROBE_TIMEOUT_S)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}"}
    acct = (r.json() or {}).get("account") or {}
    used = acct.get("plan_usage")
    limit = acct.get("plan_limit")
    return {"balance": f"{limit - used}/{limit} credits left"
            if isinstance(used, (int, float)) and isinstance(limit, (int, float))
            else f"{used} used",
            "detail": str(acct.get("current_plan") or "")}


def _probe_firecrawl() -> dict:
    key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not key:
        return {}
    import httpx
    # The v1 prefix is part of the path; the unversioned host 404s.
    r = httpx.get("https://api.firecrawl.dev/v1/team/credit-usage",
                 headers={"Authorization": f"Bearer {key}"},
                 timeout=PROBE_TIMEOUT_S)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}"}
    d = (r.json() or {}).get("data") or {}
    remaining = d.get("remaining_credits")
    total = d.get("plan_credits")
    return {"balance": f"{remaining}/{total} credits left"
            if remaining is not None else ""}


def _probe_openrouter() -> dict:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        return {}
    import httpx
    r = httpx.get("https://openrouter.ai/api/v1/credits",
                 headers={"Authorization": f"Bearer {key}"},
                 timeout=PROBE_TIMEOUT_S)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}"}
    d = (r.json() or {}).get("data") or {}
    total = d.get("total_credits")
    used = d.get("total_usage") or 0
    if not total:
        # A key with no purchased credits still reports usage. Subtracting usage
        # from a zero balance produces a negative number that reads as a bug, so
        # free-tier is stated plainly instead.
        return {"balance": f"{used:.4f} USD used" if used else "free tier",
                "detail": "no purchased credits"}
    left = max(0.0, total - used)
    return {"balance": f"{left:.2f}/{total:.2f} USD left"}


# --- the LLM chain ---------------------------------------------------------

_PROVIDER_ENV = {
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "cohere": "COHERE_API_KEY",
    "huggingface": "HF_TOKEN",
}


def _model_chain() -> list[str]:
    """The default model chain, read straight from the environment.

    Deliberately does not import ``llm``: that pulls in litellm, which costs
    ~7s cold. This dashboard runs before the run does and has no business
    making the first request wait for a heavyweight import just to print model
    names. Mirrors llm._models(role) for the default role.
    """
    from .config import DEFAULT_MODEL

    default = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    chain = [default]
    chain += [m.strip() for m in os.getenv("LLM_FALLBACKS", "").split(",")
              if m.strip()]
    out: list[str] = []
    for model in chain:                       # dedupe, keep order
        if model not in out:
            out.append(model)
    return out


def _llm_chain_status() -> ServiceStatus:
    """Summarise the configured model chain.

    Most free-tier providers publish no balance endpoint -- they are governed by
    per-minute rate limits, which is what THROTTLE_LLM_* exists to respect. So
    this reports which providers are actually wired up and how hard we are
    allowed to push them, and says plainly that no balance is knowable.
    """
    from . import config

    providers: dict[str, list[str]] = {}
    chain = _model_chain()
    for model in chain:
        provider = str(model).split("/", 1)[0]
        providers.setdefault(provider, []).append(model.split("/", 1)[-1])

    wired = [p for p, env in _PROVIDER_ENV.items()
             if os.environ.get(env) and p in providers]
    unwired = [p for p in providers if p not in wired]

    bits = []
    provider_rows: list[tuple[str, str, list[str]]] = []
    for name in sorted(providers):
        env = _PROVIDER_ENV.get(name, "")
        mark = "key set" if os.environ.get(env) else "NO KEY"
        provider_rows.append((name, mark, providers[name]))

    return ServiceStatus(
        name="LLM", purpose="planning, extraction, synthesis",
        configured=bool(chain),
        balance="",                       # no provider in the chain has one
        limit=f"{config.throttle_llm_per_minute():g}/min, "
              f"{config.throttle_llm_max_concurrent()} concurrent",
        detail=("NOT USABLE, missing key: " + ", ".join(sorted(unwired)))
               if unwired else "free-tier: governed by rate limits, not credits",
        details=[f"{name} [{mark}]: {', '.join(models)}"
                 for name, mark, models in provider_rows],
    )


# --- assembling the dashboard ---------------------------------------------

# Probes are dispatched together rather than one after another: a slow or
# unreachable provider must cost PROBE_TIMEOUT_S of wall-clock, not one
# timeout per provider, because this runs before the user's run does.
_PROBES = (
    ("serpapi", "SerpApi", "web / news / scholar / patent search",
     "SERPAPI_KEY", _probe_serpapi),
    ("tavily", "Tavily", "paid fetch fallback",
     "TAVILY_API_KEY", _probe_tavily),
    ("firecrawl", "Firecrawl", "paid fetch fallback",
     "FIRECRAWL_API_KEY", _probe_firecrawl),
    ("openrouter", "OpenRouter", "fallback model provider",
     "OPENROUTER_API_KEY", _probe_openrouter),
)


def _mock_mode() -> bool:
    """Mock runs must not query paid providers.

    In mock mode no real provider is being used, so asking them for balances
    would spend real calls to describe a run that never touched them -- and it
    would make an offline test depend on the network.
    """
    return (os.environ.get("LLM_MOCK") == "1"
            or os.environ.get("SEARCH_MOCK") == "1")


def _registry_rows() -> list[ServiceStatus]:
    """One dashboard row per registered provider, built from the registry.

    A provider with both a search and an extract role gets one row, not two:
    they share a key and a balance, and two rows would imply two accounts.

    These rows report the *per-run cap we impose* rather than a vendor balance.
    None of the registered providers publish a documented credit-read endpoint,
    and inventing one by probing an undocumented URL would produce a row that
    reads "HTTP 404" at the start of every run. Saying what we meter locally
    and how the vendor meters us is the honest version of the same information.
    """
    from . import config
    from . import providers as registry

    grouped: dict[str, list] = {}
    order: list[str] = []
    for provider in registry.for_role(registry.SEARCH) + registry.for_role(registry.EXTRACT):
        if provider.name not in grouped:
            grouped[provider.name] = []
            order.append(provider.name)
        grouped[provider.name].append(provider)

    rows: list[ServiceStatus] = []
    for name in order:
        entries = grouped[name]
        head = entries[0]
        extractors = [p for p in entries if p.role == registry.EXTRACT]
        purpose = "; ".join(dict.fromkeys(p.purpose for p in entries))
        limits = " · ".join(f"{config.provider_max_per_run(p.name)}/run"
                            for p in extractors)
        meter = " · ".join(dict.fromkeys(p.meter for p in entries))
        paused = registry.quarantine_detail(name)
        detail = f"key {_key_hint(head.env)} · {meter}"
        if paused:
            # Deliberately not buried: a provider the run has stopped using
            # looks identical to one that was never configured, and the whole
            # point of this panel is that the difference is visible.
            detail = f"PAUSED: {paused} · {detail}"
        rows.append(ServiceStatus(
            name=head.name, purpose=purpose,
            configured=head.has_key and not paused,
            balance="", limit=limits, detail=detail,
            error=paused))
    return rows


def _rows_from(results: dict) -> list[ServiceStatus]:
    from . import config

    limits = {
        "serpapi": f"per-run cap {config.max_sources()} sources",
        "tavily": f"{config.tavily_max_per_run()}/run",
        "firecrawl": f"{config.firecrawl_max_per_run()}/run",
        "openrouter": "",
    }
    rows: list[ServiceStatus] = []
    for key, name, purpose, env, _probe in _PROBES:
        data = results.get(key) or {}
        limit = limits.get(key, "")
        detail = data.get("detail") or ""
        rows.append(ServiceStatus(
            name=name, purpose=purpose,
            configured=bool(os.environ.get(env)),
            balance=data.get("balance", ""), limit=limit,
            detail=((detail + f" · key {_key_hint(env)}") if detail
                    else f"key {_key_hint(env)}"),
            error=data.get("error", "")))

    # Tavily and Firecrawl are already listed above with live balances, so the
    # registry does not repeat them; only providers the panel cannot probe are
    # added from it. Compared case-insensitively because the probe table
    # spells them "Tavily"/"Firecrawl" while the registry uses its canonical
    # lower-case name.
    already = {name.lower() for _k, name, _p, _e, _pr in _PROBES}
    rows.extend(row for row in _registry_rows() if row.name.lower() not in already)

    rows.append(_llm_chain_status())
    rows.append(ServiceStatus(
        name="Unpaywall", purpose="open-access PDF lookup (free)",
        configured=bool(os.environ.get("UNPAYWALL_EMAIL")),
        balance="unmetered", detail="email-identified, no quota reported"))
    return rows


async def snapshot() -> list[ServiceStatus]:
    """The dashboard. Probes run concurrently, off the event loop, and bounded.

    A provider that hangs or errors yields "balance unknown" for its row and
    costs at most PROBE_TIMEOUT_S; nothing here can raise into a research run.

    Futures are *polled* against a deadline rather than awaited. That matters:
    ``asyncio.wait_for`` cannot cancel a ``to_thread`` future once its thread
    has started, so it ends up waiting for the very thing it was meant to bound
    (measured: a 6s ceiling taking 60s). Polling gives a ceiling that holds no
    matter what the probe does, and abandoning the thread is safe because the
    probe only reads a provider balance.
    """
    wanted = [(key, probe) for key, _n, _p, env, probe in _PROBES
              if os.environ.get(env)]
    if _mock_mode() or not wanted:
        return _rows_from({})

    executor = ThreadPoolExecutor(max_workers=len(wanted),
                                  thread_name_prefix="svc-probe")
    pending: dict = {}
    results: dict = {}
    try:
        for key, probe in wanted:
            pending[key] = executor.submit(_cached, key, probe)

        deadline = time.monotonic() + PROBE_TIMEOUT_S
        while pending and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
            for key in [k for k, f in pending.items() if f.done()]:
                results[key] = _safe_result(pending.pop(key))

        for key, future in pending.items():
            future.cancel()               # abandon; the thread winds itself up
            results[key] = {"error": f"timed out after {PROBE_TIMEOUT_S:g}s"}
    except Exception as exc:
        log.debug("services probe dispatch failed: %s", exc)
    finally:
        executor.shutdown(wait=False)

    return _rows_from(results)


def _safe_result(future) -> dict:
    try:
        value = future.result()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:100]}"}
    return value if isinstance(value, dict) else {}


def render(rows: list[ServiceStatus], question: str = "") -> str:
    """The block printed to the terminal at the start of a run."""
    title = "services in use"
    if question:
        title += f"  ·  {question[:70]}"
    lines = ["", "=" * 78, f"  {title.upper()}", "=" * 78]
    if not rows:
        lines.append("  (service status unavailable)")
        return "\n".join(lines) + "\n"
    for row in rows:
        lines.append(f"  {row.name}")
        lines.append(f"      {row.purpose}")
        lines.append(f"      {row.line()}")
        for extra in row.details:
            lines.append(f"      - {extra}")
        if row.detail:
            lines.append(f"      {row.detail}")
    lines.append("=" * 78)
    return "\n".join(lines) + "\n"


_last_announce = 0.0


async def announce(question: str = "", *, min_gap_s: float = 30.0) -> list[ServiceStatus]:
    """Print the dashboard once at the start of a run.

    Re-announcing on every node would drown the run log; a short minimum gap
    also stops a burst of runs from turning into a burst of provider lookups.
    """
    global _last_announce
    now = time.monotonic()
    with _lock:
        due = now - _last_announce >= min_gap_s
        if due:
            _last_announce = now
    rows = await snapshot()
    if due and rows:
        log.info("\n%s", render(rows, question).rstrip())
    return rows


def reset_cache() -> None:
    """Forget cached balances and the announce gate. For tests."""
    global _last_announce
    with _lock:
        _cache.clear()
        _last_announce = 0.0