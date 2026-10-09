"""Static assertions for the frontend stream guards (audit fixes 5-6).

The SSE consumer must never let one malformed frame or one malformed source
row abort a healthy run. Structural assertions on app.js, mirroring the
style of the other frontend tests.
"""
from pathlib import Path
import re

APP = Path(__file__).resolve().parents[1] / "frontend" / "app.js"
SOURCE = APP.read_text()


def test_json_parse_is_wrapped_per_frame():
    """A bare JSON.parse in the stream loop aborted the whole run on one
    non-JSON keepalive frame."""
    idx = SOURCE.index("JSON.parse(line.slice(6))")
    fenced = SOURCE[idx - 400: idx + 120]
    assert "try {" in fenced and "catch" in fenced, \
        "the frame parser must catch and skip malformed frames"
    assert "const data = JSON.parse(line.slice(6));" not in SOURCE, \
        "the bare assignment must not come back"


def test_source_status_is_guarded_everywhere():
    """A missing status field must render as 'unknown', not crash."""
    assert "(source.status || \"unknown\").replace" in SOURCE
    assert "status-${source.status || \"unknown\"}" in SOURCE
    assert "source.status.replace" not in SOURCE


def test_article_links_are_scheme_checked():
    """Scraped URLs must not reach href unchecked (javascript: risk)."""
    assert "function safeExternalUrl" in SOURCE
    assert "title.href = source.url || \"#\"" not in SOURCE
    line = next(l for l in SOURCE.splitlines() if "title.href = " in l)
    assert "safeExternalUrl" in line, f"unguarded href assignment: {line}"


def test_stat_tile_call_signature_matches_definition():
    """statTile(value, label, hint): the big text is the value, not the label."""
    assert "function statTile(value, label, hint)" in SOURCE


# --- brand -----------------------------------------------------------------

STYLES = (APP.parent / "style.css").read_text()
MARKUP = (APP.parent / "index.html").read_text()


def test_wordmark_is_present_and_names_the_project():
    assert 'class="brand"' in MARKUP
    assert "Aletheia" in MARKUP
    assert "<title>Aletheia</title>" in MARKUP


def test_wordmark_uses_the_display_serif():
    """The hero is set in a different face, so a missing font-family on the
    brand would silently fall back and nobody would notice."""
    rule = re.search(r"\.brand-name \{.*?\}", STYLES, re.S)
    assert rule, ".brand-name rule is missing"
    assert "AnthropicSerif" in rule.group(0)


def test_logo_is_inline_svg_so_it_inherits_colour():
    mark = re.search(r'<span class="brand-mark".*?</span>', MARKUP, re.S)
    assert mark, "brand mark is missing"
    assert "<svg" in mark.group(0)
    assert 'stroke="currentColor"' in mark.group(0)
    assert "<img" not in mark.group(0), "an <img> cannot inherit theme colour"


def test_favicon_carries_the_same_mark():
    """A mismatched favicon is the first thing anyone sees in a tab."""
    icon = re.search(r'<link rel="icon" href="([^"]+)"', MARKUP).group(1)
    assert "%3Csvg" in icon, "favicon should be an inline SVG"
    assert "circle" in icon
    assert "15.5" in icon, "favicon geometry has drifted from the header mark"


def test_wordmark_lives_in_the_top_right_bar():
    """.settings is the fixed top-right strip; the brand shares it with the
    menu, so the wordmark cannot end up somewhere else on the page."""
    assert re.search(r"\.settings \{[^}]*position: fixed;", STYLES, re.S)
    assert re.search(r"\.settings \{[^}]*display: flex;", STYLES, re.S)
    settings_at = MARKUP.index('<div class="settings">')
    brand_at = MARKUP.index('class="brand"')
    assert brand_at > settings_at, "brand must live inside the settings bar"
