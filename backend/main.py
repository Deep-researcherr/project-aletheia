import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import logging as _logging


def _configure_logging() -> None:
    """Send the pipeline's own INFO lines to the terminal.

    Without this the interesting progress lines -- "extraction batch 3: 12
    passages -> 5 records", "retrieval: 40/62 sources readable" -- are emitted
    and then dropped, because nothing configures a handler for them. Set
    LOG_LEVEL=DEBUG for the full trace.
    """
    level = os.environ.get("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    _logging.basicConfig(
        level=getattr(_logging, level, _logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # uvicorn installs its own handlers; leave its format alone.
    _logging.getLogger("uvicorn.access").setLevel(_logging.WARNING)


_configure_logging()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.research.config import max_rounds as max_rounds_default
from backend.research.config import rounds_ceiling as max_rounds_ceiling
from backend.research.researcher import deep_research

log = logging.getLogger(__name__)

app = FastAPI()


def _cors_origins() -> list[str]:
    """Extra origins from CORS_ORIGINS, comma separated.

    Needed as soon as anyone reaches the app by anything other than
    localhost:8000 -- a LAN address, a container port, or a separate dev server
    on a different port. None of those are guessable, so they are configurable.
    """
    raw = os.environ.get("CORS_ORIGINS", "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _cors_origin_regex() -> str | None:
    """Pattern for loopback origins on any port.

    On by default so a teammate running a dev server on 5173/3000/8080 is not
    blocked by a hardcoded port list. Loopback only, so this grants nothing to a
    remote host; set CORS_ORIGINS for anything else.
    """
    configured = os.environ.get("CORS_ORIGIN_REGEX")
    if configured is not None:
        return configured.strip() or None
    return r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=_cors_origin_regex(),
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


@app.middleware("http")
async def no_cache_frontend(request, call_next):
    """Force revalidation of the frontend shell on every load.

    Without an explicit Cache-Control, browsers fall back to heuristic
    caching and can serve days-old JS/CSS without revalidating — exactly
    how a fresh index.html ends up running against a stale app.js.
    API routes (including the SSE stream) are untouched.
    """
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/health")
async def health():
    """Liveness probe for uptime monitors, load balancers and containers."""
    return {"status": "ok"}


@app.get("/api/health")
async def api_health():
    """Same probe under /api for clients that prefix everything."""
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    """Minimal OpenAI-compatible model list.

    Nothing in this app calls it; external tooling (LLM gateways, monitors)
    probes /v1/models on any localhost server, which used to spam 404s in the
    log right after each /health check. Answer with the configured chain.
    """
    from backend.research.llm import _models

    seen: list[str] = []
    for role in ("default", "judge", "synth"):
        try:
            chain = _models(role)
        except Exception:
            chain = []
        for name in chain:
            if name and name not in seen:
                seen.append(name)
    return {
        "object": "list",
        "data": [{"id": name, "object": "model"} for name in seen],
    }


@app.get("/api/research")
async def research(q: str, rounds: int = 2, throttle: int = -1):
    # Clamped against the configured ceiling rather than a hard-coded number, so
    # raising RESEARCH_MAX_ROUNDS_LIMIT is enough to allow deeper runs.
    ceiling = max_rounds_ceiling()
    depth = max(1, min(rounds, ceiling))
    if depth != rounds:
        log.warning("rounds=%s clamped to %d (ceiling %d)", rounds, depth, ceiling)

    # -1 means "use the configured default"; 0/1 override it for this run.
    throttled = None if throttle < 0 else bool(throttle)

    async def gen():
        from backend.research import throttle as throttle_module

        if throttled is not None:
            throttle_module.apply_config(enabled=throttled)
        try:
            async for ev in deep_research(q, depth):
                if ev.get("type") in ("report", "status"):
                    ev["throttle"] = throttle_module.snapshot()
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            # Restore the configured default so one unthrottled request does not
            # silently disable pacing for every later run.
            if throttled is not None:
                throttle_module.apply_config()

        yield 'data: {"type":"done"}\n\n'
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/throttle")
async def throttle_status():
    """Live throttle state, so the UI can show what pacing is in effect."""
    from backend.research import throttle as throttle_module
    from backend.research.config import (throttle_llm, throttle_llm_max_concurrent,
                                         throttle_llm_per_minute)

    throttle_module.apply_config()
    return {
        "configured": {
            "enabled": throttle_llm(),
            "per_minute": throttle_llm_per_minute(),
            "max_concurrent": throttle_llm_max_concurrent(),
        },
        "live": throttle_module.snapshot(),
    }


@app.get("/api/config")
async def public_config():
    """Limits the browser needs in order to offer valid choices."""
    from backend.research.config import budget_estimate, total_llm_budget

    ceiling = max_rounds_ceiling()
    from backend.research.config import (throttle_llm, throttle_llm_max_concurrent,
                                         throttle_llm_per_minute,
                                         throttle_search, throttle_search_per_minute)
    return {
        "throttle": {
            "enabled": throttle_llm(),
            "per_minute": throttle_llm_per_minute(),
            "max_concurrent": throttle_llm_max_concurrent(),
            "search_enabled": throttle_search(),
            "search_per_minute": throttle_search_per_minute(),
        },
        "rounds_ceiling": ceiling,
        "rounds_default": min(max_rounds_default(), ceiling),
        "llm_budget": total_llm_budget(),
        "budget_estimate": {str(r): budget_estimate(r) for r in range(1, ceiling + 1)},
    }


@app.get("/api/report/download")
async def download(q: str):
    """Serve the markdown export for a question as a file download.

    The browser cannot write to the user's disk, so the file the server already
    wrote is served back with a download disposition rather than pretending the
    page generated the file locally.
    """
    from backend.research.export import read, slugify

    content = read(q)
    if content is None:
        raise HTTPException(status_code=404,
                            detail="no exported report found for this question")
    filename = f"{slugify(q)}.md"
    return StreamingResponse(
        iter([content.encode("utf-8")]),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/report/markdown")
async def report_markdown(q: str):
    """Raw markdown text, for clients that want to render it themselves."""
    from backend.research.export import read

    content = read(q)
    if content is None:
        raise HTTPException(status_code=404,
                            detail="no exported report found for this question")
    return {"markdown": content}


@app.get("/api/report/pdf")
async def report_pdf(q: str, rounds: int = 2, throttle: int = -1):
    """Styled PDF of the most recent completed run for a question.

    Built deterministically from the recorded run bundle (the same payloads
    the frontend received), so the PDF mirrors what was displayed. 404 when
    no completed run exists for the question. ``rounds``/``throttle`` are
    display-only labels for the run-overview table.
    """
    import asyncio as _asyncio

    from backend.research import runlog
    from backend.research.export import slugify
    from backend.research.pdf_report import build_pdf

    bundle = runlog.get(q)
    if bundle is None:
        raise HTTPException(status_code=404,
                            detail="no completed report found for this question; "
                                   "run research first")
    rounds_label = f"{max(1, rounds)} round{'s' if max(1, rounds) != 1 else ''}"
    throttle_label = ("throttled" if throttle else "unthrottled") \
        if throttle >= 0 else ""
    try:
        pdf = await _asyncio.to_thread(build_pdf, bundle,
                                       rounds_label, throttle_label)
    except Exception as exc:
        log.warning("PDF build failed for %r: %s", q[:80], exc)
        raise HTTPException(status_code=500,
                            detail="could not build the PDF for this report")
    filename = f"{slugify(q)}.pdf"
    return StreamingResponse(
        iter([pdf]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _exports_dir() -> Path:
    from backend.research.export import output_dir
    return output_dir()


def _frontend_dir() -> Path:
    """Absolute path to the bundled frontend.

    A relative "frontend" resolves against the *process* working directory, so
    it works when uvicorn is started from the repo root and fails with
    "Directory 'frontend' does not exist" the moment anything starts the app
    from anywhere else -- a different shell, an IDE runner, or pytest.
    """
    return Path(__file__).resolve().parents[1] / "frontend"


# Only mount when the directory exists, so a fresh checkout still boots.
try:
    _exports_dir().mkdir(parents=True, exist_ok=True)
    app.mount("/exports", StaticFiles(directory=str(_exports_dir())), name="exports")
except OSError:
    pass

_frontend = _frontend_dir()
if _frontend.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
else:                                            # pragma: no cover
    raise RuntimeError(
        f"frontend directory not found at {_frontend}. "
        "The repository looks incomplete -- re-clone it."
    )