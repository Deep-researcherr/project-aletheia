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
    assert "text-transform: uppercase" in rule.group(0), \
        "the wordmark is set in caps with wide tracking, as in the artwork"


def test_logo_is_inline_svg_so_it_inherits_colour():
    mark = re.search(r'<span class="brand-mark".*?</span>', MARKUP, re.S)
    assert mark, "brand mark is missing"
    assert "<svg" in mark.group(0)
    assert 'stroke="currentColor"' in mark.group(0)
    assert "<img" not in mark.group(0), "an <img> cannot inherit theme colour"


def test_mark_geometry_matches_the_supplied_artwork():
    """Open circle, lambda inside, amber point at the centre.

    Traced from the artwork: circle r=36.5 with a ~124 degree gap at the
    bottom, lambda from (32.6,58.5) to (50,30.5) to (67.4,58.5), and the amber
    point sitting on the circle's exact centre.
    """
    mark = re.search(r'<span class="brand-mark".*?</span>', MARKUP, re.S).group(0)
    assert "M82.2 67.2 A36.5 36.5 0 1 0 17.8 67.2" in mark, \
        "circle arc (r=36.5, gap at the bottom) has drifted"
    assert "M32.6 58.5 L50 30.5 L67.4 58.5" in mark, "lambda has drifted"
    assert 'cx="50" cy="50" r="2.8"' in mark, "amber point is off-centre"
    assert "#E9B031" in mark, "the amber accent is missing"
    # Butt caps and a miter join: the artwork's lambda has flat-cut feet and a
    # crisp apex, which round caps would blunt.
    assert 'stroke-linecap="butt"' in mark


def test_favicon_carries_the_same_mark():
    """A mismatched favicon is the first thing anyone sees in a tab."""
    icon = re.search(r'<link rel="icon" href="([^"]+)"', MARKUP).group(1)
    assert "%3Csvg" in icon, "favicon should be an inline SVG"
    assert "36.5" in icon, "favicon geometry has drifted from the header mark"
    assert "32.6 58.5" in icon or "32.6%2058.5" in icon, \
        "favicon lambda has drifted from the header mark"
    # Thicker strokes so the mark survives being scaled into a tab.
    assert "stroke-width='9'" in icon or "stroke-width='4'" in icon


def test_wordmark_is_top_left_and_settings_top_right():
    """.topbar is the full-width strip; the brand must not be re-centred.

    A dead legacy `.brand { margin: auto }` rule sat further down this file
    and silently re-centred the wordmark, because a later rule of equal
    specificity wins. That is the exact failure this guards against.
    """
    assert re.search(r"\.topbar \{[^}]*justify-content: space-between;", STYLES, re.S)
    assert not re.search(r"^\.brand \{[^}]*margin: auto;", STYLES, re.M | re.S), \
        "margin:auto on .brand will centre the wordmark again"
    topbar_at = MARKUP.index('<div class="topbar">')
    brand_at = MARKUP.index('class="brand"')
    assert brand_at > topbar_at, "brand must live inside the topbar"
    assert MARKUP.index('class="settings"') > brand_at, \
        "settings must follow the brand in the bar"
