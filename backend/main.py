import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.research.researcher import deep_research

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/research")
async def research(q: str, rounds: int = 2):
    async def gen():
        try:
            async for ev in deep_research(q, max(1, min(rounds, 4))):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield 'data: {"type":"done"}\n\n'
    return StreamingResponse(gen(), media_type="text/event-stream")


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


def _exports_dir() -> Path:
    from backend.research.export import output_dir
    return output_dir()


# Only mount when the directory exists, so a fresh checkout still boots.
try:
    _exports_dir().mkdir(parents=True, exist_ok=True)
    app.mount("/exports", StaticFiles(directory=str(_exports_dir())), name="exports")
except OSError:
    pass

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")