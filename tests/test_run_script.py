"""Tests for run.py, the preflight and launcher.

A launcher that reports "ready" when it is not is worse than no launcher:
it moves the failure somewhere with a worse error message. So these tests
check that it actually detects each failure it claims to detect, rather than
just that it runs.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run  # noqa: E402


# --- it must be syntactically whole ---------------------------------------

def test_every_colour_call_is_well_formed():
    """A single-argument _colour() raises at runtime, not at import.

    It shipped that way once: the preflight printed "ready", the server came
    up, and then the script died on the next line and orphaned a listening
    uvicorn. Nothing in the test suite noticed, because nothing covered this.
    """
    tree = ast.parse((ROOT / "run.py").read_text())
    bad = [n.lineno for n in ast.walk(tree)
           if isinstance(n, ast.Call)
           and getattr(n.func, "id", "") == "_colour"
           and len(n.args) != 2]
    assert not bad, f"_colour() called with the wrong arity on lines {bad}"


# --- dependency checking ---------------------------------------------------

def test_requirements_parsing_skips_noise():
    names = run.parse_requirements(ROOT / "requirements.txt")
    assert "-r" not in names
    assert not any(n.startswith("-") for n in names)
    assert "fastapi" in names and "litellm" in names


def test_dev_requirements_are_the_extra_only():
    """It must not re-list runtime packages; the -r include means pip merges
    them, and duplicating them here would make two sources of truth."""
    names = run.parse_requirements(ROOT / "requirements-dev.txt")
    assert "pytest" in names
    assert "fastapi" not in names, "runtime deps must not be repeated in dev"


def test_a_missing_package_is_detected():
    missing = run.missing_packages(["fastapi", "definitely-not-a-real-pkg"])
    assert len(missing) == 1
    dist, module = missing[0]
    assert dist == "definitely-not-a-real-pkg"
    assert module == "definitely_not_a_real_pkg"


def test_present_packages_are_not_reported_missing():
    missing = dict(run.missing_packages(["fastapi", "pytest"]))
    assert "fastapi" not in missing
    assert "pytest" not in missing


@pytest.mark.parametrize("dist,expected", [
    ("beautifulsoup4", "bs4"),
    ("python-dotenv", "dotenv"),
    ("google-search-results", "serpapi"),
    ("fonttools", "fontTools"),
])
def test_distribution_names_map_to_their_real_import_names(dist, expected):
    """Guessing the import from the distribution name fails for a third of
    them. beautifulsoup4 does not provide a 'beautifulsoup4' module."""
    assert run.IMPORT_NAMES[dist] == expected


def test_every_requirement_resolves_to_an_installable_module():
    """Catches a dependency added to requirements.txt whose import name is not
    mapped here -- it would be reported missing forever."""
    for path in (ROOT / "requirements.txt", ROOT / "requirements-dev.txt"):
        for dist in run.parse_requirements(path):
            assert run.IMPORT_NAMES.get(dist.lower(), dist.replace("-", "_")), \
                f"{dist} in {path.name} has no resolvable import name"


def test_runtime_dependencies_are_all_present_in_this_environment():
    """run.py's first job. If this fails, the launcher cannot work here."""
    assert run.missing_packages(run.parse_requirements(ROOT / "requirements.txt")) == []


# --- environment checking --------------------------------------------------

def test_env_parsing_reads_real_keys_without_importing_dotenv():
    values = run.read_env()
    assert "SERPAPI_KEY" in values, "the repo's .env should parse"
    assert all(" " not in v for v in values.values())


def test_env_parsing_ignores_comments_and_blank_lines(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# a comment\n\nKEY=value\n  SPACED = padded  \n")
    values = {k: v for k, v in
              ((k.strip(), v.strip()) for k, _, v in
               (l.partition("=") for l in path.read_text().splitlines()
                if "=" in l and not l.strip().startswith("#")))}
    assert values == {"KEY": "value", "SPACED": "padded"}


def test_missing_env_file_is_reported_not_ignored(tmp_path, monkeypatch):
    """A run with no .env searches with zero engines and says nothing. That
    is the failure this preflight exists to catch."""
    monkeypatch.setattr(run, "ENV_FILE", tmp_path / "nope.env")
    assert run.read_env() == {}
    assert run.check_env() is False


def test_optional_keys_are_a_warning_not_a_failure(monkeypatch, tmp_path):
    """Nobody runs a personal project with six provider keys wired. A missing
    optional key must not block the launcher."""
    env = tmp_path / ".env"
    env.write_text("SERPAPI_KEY=k\nLLM_MODEL=gemini/x\n")
    monkeypatch.setattr(run, "ENV_FILE", env)
    assert run.check_env() is True, "absent optional keys must not fail the check"


# --- font checking ---------------------------------------------------------

def test_bundled_fonts_are_present():
    """A missing font is a 404 the browser absorbs by substituting a fallback
    face, so it looks like a design choice rather than a broken file."""
    assert run.check_fonts() is True


def test_a_missing_font_is_detected(monkeypatch, tmp_path):
    css = tmp_path / "style.css"
    css.write_text('@font-face { src: url("./Nope_font/Nope.otf") format("opentype"); }')
    monkeypatch.setattr(run, "STYLESHEET", css)
    assert run.check_fonts() is False


def test_data_uris_and_remote_urls_are_not_checked_on_disk(monkeypatch, tmp_path):
    css = tmp_path / "style.css"
    css.write_text('@font-face { src: url("https://example.com/f.woff2"); }\n'
                   '@font-face { src: url("data:font/woff2;base64,AAA"); }')
    monkeypatch.setattr(run, "STYLESHEET", css)
    assert run.check_fonts() is True


# --- end to end ------------------------------------------------------------

def test_check_mode_succeeds_and_exits_zero_in_this_repo():
    result = subprocess.run([sys.executable, "run.py", "--check"],
                            cwd=str(ROOT), capture_output=True, text=True,
                            timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ready" in result.stdout


def test_check_mode_exits_non_zero_when_a_package_is_missing(monkeypatch,
                                                                capsys):
    """The exit code is the part a script or CI actually reads.

    main() *returns* its code rather than raising SystemExit -- only the
    __main__ guard turns it into an exit -- so the contract under test is the
    return value.
    """
    monkeypatch.setattr(run, "check_packages",
                        lambda include_dev=False: [("some-pkg", "some_pkg")])
    monkeypatch.setattr(sys, "argv", ["run.py", "--check"])
    assert run.main() == 1
    capsys.readouterr()


def test_check_mode_exits_non_zero_when_a_required_key_is_absent(monkeypatch,
                                                                  tmp_path,
                                                                  capsys):
    env = tmp_path / ".env"
    env.write_text("SOME_OTHER_KEY=x\n")          # neither required key present
    monkeypatch.setattr(run, "ENV_FILE", env)
    monkeypatch.setattr(sys, "argv", ["run.py", "--check"])
    assert run.main() == 1
    capsys.readouterr()


def test_check_mode_exits_zero_when_everything_is_present(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run.py", "--check"])
    assert run.main() == 0
    assert "ready" in capsys.readouterr().out


def test_a_missing_package_stops_the_launch_before_serving(monkeypatch, capsys):
    """--check is opt-in; without it, the default path must still refuse to
    start uvicorn when a dependency is absent."""
    served = []
    monkeypatch.setattr(run, "check_packages",
                        lambda include_dev=False: [("some-pkg", "some_pkg")])
    monkeypatch.setattr(run, "serve",
                        lambda *a, **k: served.append(a) or 0)
    monkeypatch.setattr(sys, "argv", ["run.py"])
    assert run.main() == 1
    assert served == [], "it must not launch a server it knows cannot work"
    capsys.readouterr()
