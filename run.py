#!/usr/bin/env python3
"""Check the environment is ready, then serve Project Aletheia.

There is one process, not two. backend/main.py mounts frontend/ as static
files at "/", so the same uvicorn that answers /health also serves the UI,
the SSE stream and the export downloads. A second "frontend terminal" would
have nothing to do.

What this checks, and why each one:

  * Python packages, parsed from requirements.txt rather than a list kept
    here -- so a dependency added to requirements.txt is checked without
    editing this file. Presence is tested with importlib.util.find_spec,
    which does not execute the module: importing litellm costs seconds and
    would defeat the point of a fast preflight.
  * .env exists, and carries the keys a run actually needs. A missing key is
    the failure this project has been bitten by most: the run does not crash,
    it quietly searches with fewer engines and says nothing.
  * The bundled fonts the stylesheet references, because a 404 there degrades
    the wordmark to a fallback face with no error anywhere.

Usage:
    python run.py              check, then serve
    python run.py --check      check only, exit non-zero if something is missing
    python run.py --install    install anything missing first
    python run.py --port 9000  serve somewhere else
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"
DEV_REQUIREMENTS = ROOT / "requirements-dev.txt"
ENV_FILE = ROOT / ".env"
STYLESHEET = ROOT / "frontend" / "style.css"

#: Distribution name -> import name. They differ often enough that guessing
#: from the distribution name fails for a third of them: the import for
#: "beautifulsoup4" is bs4, for "python-dotenv" is dotenv, and for
#: "google-search-results" it is serpapi.
IMPORT_NAMES = {
    "beautifulsoup4": "bs4",
    "python-dotenv": "dotenv",
    "google-search-results": "serpapi",
    "lxml": "lxml",
    "pypdf": "pypdf",
    "fonttools": "fontTools",
    "reportlab": "reportlab",
    "requests": "requests",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "litellm": "litellm",
    "langgraph": "langgraph",
    "pytest": "pytest",
    "dukpy": "dukpy",
}

#: Keys without which a run still "succeeds" while quietly doing less.
REQUIRED_KEYS = ("SERPAPI_KEY", "LLM_MODEL")
#: Warned about, never fatal. Each one removes a source or a fallback tier.
OPTIONAL_KEYS = ("EXA_API_KEY", "JINA_API_KEY", "TINYFISH_API_KEY",
                 "SCRAPINGBEE_API_KEY", "TAVILY_API_KEY", "FIRECRAWL_API_KEY")

GREEN, RED, YELLOW, DIM, BOLD, OFF = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m")


def _colour(text: str, code: str) -> str:
    return f"{code}{text}{OFF}" if sys.stdout.isatty() else text


def parse_requirements(path: Path, include_dev: bool = False) -> list[str]:
    """Distribution names from a requirements file.

    Skips comments, blank lines, option lines and `-r` includes, which is
    what stops this from trying to import the string "-r".
    """
    names: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith(("-r", "--", "-e", "-c")):
            continue
        match = re.match(r"^([A-Za-z0-9._-]+)", line)
        if match:
            names.append(match.group(1))
    return names


def missing_packages(names: list[str]) -> list[tuple[str, str]]:
    """(distribution, import name) for every one that will not import."""
    missing = []
    for dist in names:
        module = IMPORT_NAMES.get(dist.lower(), dist.replace("-", "_"))
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append((dist, module))
    return missing


def check_packages(include_dev: bool) -> list[tuple[str, str]]:
    if not REQUIREMENTS.is_file():
        print(_colour(f"  {RED}x{OFF} requirements.txt is missing", RED))
        return []
    names = parse_requirements(REQUIREMENTS, include_dev)
    if include_dev and DEV_REQUIREMENTS.is_file():
        names += parse_requirements(DEV_REQUIREMENTS)
    missing = missing_packages(names)
    if missing:
        for dist, module in missing:
            print(f"  {RED}x{OFF} {dist:<24} {DIM}(imports as '{module}'){OFF}")
    else:
        print(f"  {GREEN}ok{OFF} {len(names)} packages present")
    return missing


def read_env() -> dict[str, str]:
    """Parse .env without importing dotenv, so this stays dependency-free."""
    values: dict[str, str] = {}
    if not ENV_FILE.is_file():
        return values
    for raw in ENV_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def check_env() -> bool:
    if not ENV_FILE.is_file():
        print(f"  {RED}x{OFF} .env is missing{OFF}")
        print(f"      {DIM}cp .env.example .env  then fill in your keys{OFF}")
        return False
    values = read_env()
    ok = True
    for key in REQUIRED_KEYS:
        if values.get(key):
            print(f"  {GREEN}ok{OFF} {key}")
        else:
            ok = False
            print(f"  {RED}x{OFF} {key:<20} {DIM}required{OFF}")
    absent = [k for k in OPTIONAL_KEYS if not values.get(k)]
    present = len(OPTIONAL_KEYS) - len(absent)
    if absent:
        print(f"  {YELLOW}!{OFF} {present}/{len(OPTIONAL_KEYS)} optional provider keys")
        print(f"      {DIM}absent: {', '.join(absent)}{OFF}")
        print(f"      {DIM}each one removes a search engine or a fetch fallback.{OFF}")
        print(f"      {DIM}a run without them still succeeds -- and says nothing.{OFF}")
    else:
        print(f"  {GREEN}ok{OFF} all {len(OPTIONAL_KEYS)} provider keys")
    return ok


def check_fonts() -> bool:
    """The stylesheet's bundled fonts must exist on disk.

    A missing font is a 404 the browser absorbs by substituting a fallback
    face, so it looks like a design decision rather than a broken file.
    """
    if not STYLESHEET.is_file():
        return True
    urls = re.findall(r'src:\s*url\("([^"]+)"\)', STYLESHEET.read_text())
    ok = True
    for url in urls:
        if url.startswith(("http://", "https://", "data:")):
            continue
        path = (STYLESHEET.parent / url.lstrip("./")).resolve()
        if path.is_file():
            print(f"  {GREEN}ok{OFF} {url.lstrip('./')}")
        else:
            ok = False
            print(f"  {RED}x{OFF} {url.lstrip('./')} {DIM}referenced but missing{OFF}")
    return ok


def preflight(include_dev: bool) -> tuple[bool, bool]:
    print(_colour("\nProject Aletheia — preflight\n", BOLD))
    packages_ok = not check_packages(include_dev)
    print()
    env_ok = check_env()
    print()
    fonts_ok = check_fonts()
    print()
    return packages_ok, env_ok and fonts_ok


def install(deps_file: Path) -> int:
    print(_colour(f"\ninstalling from {deps_file.name}...\n", BOLD))
    return subprocess.call([sys.executable, "-m", "pip", "install", "-r",
                            str(deps_file)])


def health_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def serve(port: int, host: str, reload: bool) -> int:
    """Start uvicorn, confirm it is actually answering, then wait on it.

    Popen rather than call, so /health can be polled while it boots. Saying
    "serving on ..." before the port is open is the kind of claim that sends
    you to a browser too early and finds nothing there.
    """
    cmd = [sys.executable, "-m", "uvicorn", "backend.main:app",
           "--host", host, "--port", str(port)]
    if reload:
        cmd.append("--reload")
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"

    print(_colour(f"\nstarting {url}\n", BOLD))
    process = subprocess.Popen(cmd, cwd=str(ROOT))

    deadline = time.time() + 45.0
    while process.poll() is None and time.time() < deadline:
        if health_ok(url):
            print(_colour(f"  ready  {url}", GREEN))
            print(_colour("  Ctrl+C to stop\n", DIM))
            break
        time.sleep(0.4)

    if process.poll() is not None:
        print(_colour(f"\nuvicorn exited immediately (code {process.returncode})"
                      f"\n", RED))
        return process.returncode or 1

    try:
        return process.wait()
    except KeyboardInterrupt:
        print()
        return _stop(process)
    finally:
        # Whatever ends this function -- Ctrl+C, an exception, anything --
        # uvicorn must not be left holding the port as an orphan. Without this
        # a crash in run.py leaves a server running that nothing supervises.
        if process.poll() is None:
            _stop(process)


def _stop(process) -> int:
    """Shut uvicorn down, escalating if it ignores the polite request."""
    process.terminate()
    try:
        return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the environment, then serve Project Aletheia.")
    parser.add_argument("--check", action="store_true",
                        help="check only; exit 1 if anything required is missing")
    parser.add_argument("--install", action="store_true",
                        help="install missing packages first")
    parser.add_argument("--dev", action="store_true",
                        help="include dev requirements (pytest, dukpy)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="uvicorn --reload, for editing")
    args = parser.parse_args()

    os.chdir(ROOT)

    if args.install:
        install(REQUIREMENTS)
        if args.dev and DEV_REQUIREMENTS.is_file():
            install(DEV_REQUIREMENTS)

    packages_ok, rest_ok = preflight(args.dev)

    if args.check:
        if packages_ok and rest_ok:
            print(_colour("ready\n", GREEN))
            return 0
        print(_colour("\nnot ready\n", RED))
        return 1

    if not packages_ok:
        print(_colour("\nInstall the missing packages first:\n", RED))
        print(f"    {sys.executable} -m pip install -r requirements.txt\n")
        return 1

    if not serve(args.port, args.host, args.reload):
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())