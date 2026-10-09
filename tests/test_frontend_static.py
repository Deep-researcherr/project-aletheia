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
    assert "Project&nbsp;Aletheia" in MARKUP, \
        "the wordmark is 'Project Aletheia', not a bare 'Aletheia'"
    assert "<title>Project Aletheia</title>" in MARKUP


def test_wordmark_uses_the_display_serif():
    """The hero is set in a different face, so a missing font-family on the
    brand would silently fall back and nobody would notice."""
    rule = re.search(r"\.brand-name \{.*?\}", STYLES, re.S)
    assert rule, ".brand-name rule is missing"
    assert "AnthropicSerif" in rule.group(0)
    assert "text-transform: uppercase" in rule.group(0), \
        "the wordmark is set in caps, as in the artwork"
    assert "font-weight: 700" in rule.group(0), \
        "the wordmark is set bold -- it is now the only thing identifying the project"


def test_header_carries_no_logo_mark():
    """The header is text only.

    The mark lives in the favicon, where it has room to breathe. At header size
    a circle-and-lambda beside a 32px wordmark read as a competing element
    rather than a lockup, and the artwork's mark is small relative to its name
    anyway.
    """
    assert "brand-mark" not in MARKUP, "the header logo was removed on purpose"
    assert not re.search(r"\.brand-mark \{", STYLES), \
        "the removed .brand-mark rule should be deleted, not left behind"
    brand = re.search(r'<a class="brand".*?</a>', MARKUP, re.S)
    assert brand, "brand link is missing"
    assert "<svg" not in brand.group(0), "the header wordmark is text only"


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
    # The menu must be pinned to the viewport, not to the 1100px bar, or it
    # drifts inwards on a wide screen instead of sitting at the far right.
    settings_rule = re.search(r"^\.settings \{[^}]*\}", STYLES, re.M | re.S)
    assert settings_rule, ".settings rule is missing"
    assert "position: fixed" in settings_rule.group(0)
    topbar_at = MARKUP.index('<div class="topbar">')
    brand_at = MARKUP.index('class="brand"')
    assert brand_at > topbar_at, "brand must live inside the topbar"
    assert MARKUP.index('class="settings"') > brand_at, \
        "settings must follow the brand in the bar"


def test_mark_and_wordmark_sizes():
    """The wordmark is the whole header now, so it is sized like a masthead:
    32px, bold, with the topbar padding pulled left so it sits near the edge."""
    name = re.search(r"\.brand-name \{[^}]*\}", STYLES, re.S)
    assert name, ".brand-name rule is missing"
    size = float(re.search(r"font-size: (\d+)px", name.group(0)).group(1))
    assert size >= 30, f"wordmark is only {size}px for a text-only masthead"
    assert float(re.search(r"font-weight: (\d+)", name.group(0)).group(1)) >= 600, \
        "the wordmark should be bold"
    topbar = re.search(r"\.topbar \{[^}]*\}", STYLES, re.S).group(0)
    assert "padding: 18px 16px 6px;" in topbar, \
        "topbar padding should hold the wordmark close to the left edge"
