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
    assert "Project Aletheia" in MARKUP
    assert "<title>Project Aletheia</title>" in MARKUP


def test_wordmark_is_a_two_line_lockup_with_the_name_under_project():
    """"PROJECT" small above "ALETHEIA" -- it reads in that order, so the page
    says it is a project before the name has finished landing."""
    assert 'class="brand-kicker"' in MARKUP
    assert 'class="brand-word"' in MARKUP
    kicker_at = MARKUP.index('class="brand-kicker"')
    word_at = MARKUP.index('class="brand-word"')
    assert kicker_at < word_at, "'Project' must come before 'Aletheia' in the DOM"

    name = re.search(r"\.brand-name \{[^}]*\}", STYLES, re.S).group(0)
    assert "flex-direction: column" in name, \
        "the two lines must stack, not sit side by side"

    kicker = re.search(r"\.brand-kicker \{[^}]*\}", STYLES, re.S).group(0)
    word = re.search(r"\.brand-word \{[^}]*\}", STYLES, re.S).group(0)
    kicker_px = float(re.search(r"font-size: (\d+)px", kicker).group(1))
    word_px = float(re.search(r"font-size: (\d+)px", word).group(1))
    assert kicker_px < word_px, \
        "the kicker must be smaller than the name it introduces"


def test_wordmark_uses_the_code_font():
    """JetBrains Mono first in the stack, with a monospace fallback behind it.

    The fallback matters: the font file could fail to load on any given
    machine, and the stack has to land on something monospace rather than
    quietly reverting to the display serif and reading as a different design.
    """
    rule = re.search(r"\.brand-name \{[^}]*\}", STYLES, re.S).group(0)
    assert "JetBrains Mono" in rule
    families = re.search(r"font-family:([^;]+);", rule, re.S).group(1)
    assert "monospace" in families, "the stack must end in a generic monospace"
    assert "serif" not in families, "a serif fallback would undo the lockup"
    for line in (".brand-kicker", ".brand-word"):
        assert re.search(rf"{re.escape(line)} \{{[^}}]*\}}", STYLES, re.S)


def test_the_wordmark_font_is_bundled_and_licensable():
    """Bundled, and under a licence that permits redistribution.

    Two separate failures this guards against:

    * the font silently becoming a system one again -- works here, nowhere else,
      and invisible in the source;
    * a font file being bundled without checking what its licence says. Comic
      Code was briefly committed for exactly that reason and had to be pulled:
      "Copyright 2019 Toshi Omagari. All rights reserved" is not a licence to
      redistribute, and the repo is public.
    """
    face = re.search(r'@font-face\s*\{\s*font-family:\s*"JetBrains Mono";.*?\}',
                     STYLES, re.S)
    assert face, "the wordmark font has no @font-face; it is resolving from the system"
    src = re.search(r"src:\s*url\(\"([^\"]+)\"\)", face.group(0))
    assert src, "@font-face has no src"
    font_file = APP.parent / src.group(1).lstrip("./")
    assert font_file.is_file(), (
        f"{src.group(1)} is referenced by the stylesheet but is not in the repo")
    assert font_file.stat().st_size > 10_000, "the font file looks truncated"
    assert "font-weight: 400" in face.group(0), \
        "only one weight is bundled, so only that weight may be declared"

    licence = font_file.parent / "OFL.txt"
    assert licence.is_file(), (
        "SIL OFL requires the licence to travel with the font it covers")
    assert "SIL Open Font License" in licence.read_text(errors="ignore")

    # Comic Code is not redistributable, so it must not come back. Checked
    # against the declarations rather than the whole file, because the CSS
    # deliberately names it in a comment recording why it was replaced.
    for declaration in re.findall(r"font-family:\s*([^;{}]+)[;}]", STYLES):
        assert "Comic Code" not in declaration, \
            "Comic Code is not redistributable; do not let it back into a stack"
    assert "ComicCode_font" not in STYLES, \
        "no src may point back at the removed font directory"
    assert not (APP.parent / "ComicCode_font").exists(), \
        "the Comic Code files are gone from the working tree"


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
    """.topbar spans the viewport; the brand must not be re-centred.

    Two ways this has gone wrong, both guarded:

    a dead legacy `.brand { margin: auto }` rule further down this file, which
    silently re-centred the wordmark because a later rule of equal specificity
    wins; and `max-width: 1100px; margin: 0 auto` on the bar itself, which put
    the wordmark 186px in from the left edge on a 1440px screen -- the middle
    of the page wearing a "top left" hat.
    """
    assert re.search(r"\.topbar \{[^}]*justify-content: space-between;", STYLES, re.S)
    assert not re.search(r"^\.brand \{[^}]*margin: auto;", STYLES, re.M | re.S), \
        "margin:auto on .brand will centre the wordmark again"

    topbar = re.search(r"^\.topbar \{[^}]*\}", STYLES, re.M | re.S).group(0)
    # The wordmark is chrome, not content: it has no reason to track the 1100px
    # column, and a cap here is what pushed it inwards.
    assert not re.search(r"max-width:\s*\d+px", topbar), \
        "a width-capped topbar centres the bar and moves the wordmark inwards"
    assert re.search(r"margin:\s*0\s*;", topbar), \
        "the topbar must not be auto-margined into the middle of the page"
    assert "padding: 14px 20px 0;" in topbar, \
        "left padding is what actually places the wordmark near the edge"

    # And out of normal flow. In flow, the bar took ~64px of height above the
    # hero and pushed the whole page down -- the settings menu used to be
    # position:fixed and cost nothing, and that is what it should cost again.
    assert re.search(r"position:\s*absolute;", topbar), \
        "an in-flow topbar pushes the page content down"
    assert re.search(r"top:\s*0;", topbar)

    # The menu must be pinned to the viewport, not to the bar, or it drifts
    # inwards on a wide screen instead of sitting at the far right.
    settings_rule = re.search(r"^\.settings \{[^}]*\}", STYLES, re.M | re.S)
    assert settings_rule, ".settings rule is missing"
    assert "position: fixed" in settings_rule.group(0)
    topbar_at = MARKUP.index('<div class="topbar">')
    brand_at = MARKUP.index('class="brand"')
    assert brand_at > topbar_at, "brand must live inside the topbar"
    assert MARKUP.index('class="settings"') > brand_at, \
        "settings must follow the brand in the bar"


def test_wordmark_is_not_set_like_a_headline():
    """The lockup is chrome, not a masthead.

    It was 32px on a single line, which shouted over the hero. It is now 28px
    stacked over a 16px kicker, raised in 2px steps on request.

    There used to be a flat `size <= 26` cap here. It is gone, deliberately:
    it only ever existed to catch the one-line 32px treatment coming back, and
    the ratio assertions below do that job properly. At 28px the cap was also
    contradicting them -- the word passed both ratio bounds while failing the
    flat one, which is a guard that disagrees with itself and gets deleted
    rather than bumped.

    The headroom is now nearly gone by design. The wordmark sits just under the
    hero's smallest rendered size, so the next +2px fails here rather than
    silently making the header louder than the headline.
    """
    word = re.search(r"\.brand-word \{[^}]*\}", STYLES, re.S).group(0)
    size = float(re.search(r"font-size: (\d+)px", word).group(1))
    assert float(re.search(r"font-weight: (\d+)", word).group(1)) >= 600, \
        "the name should still read as the strong line of the lockup"

    hero = re.search(r"\.hero-title \{[^}]*\}", STYLES, re.S)
    assert hero, ".hero-title rule is missing"
    clamp = re.search(r"font-size:\s*clamp\(\s*([\d.]+)px\s*,[^,]+,\s*([\d.]+)px\s*\)",
                      hero.group(0))
    assert clamp, "the hero no longer uses clamp(), so the ratio check is stale"
    smallest, largest = float(clamp.group(1)), float(clamp.group(2))
    assert size < smallest, (
        f"the wordmark ({size}px) must stay under the hero's smallest size "
        f"({smallest}px), which is what a narrow viewport renders")
    assert size < largest * 0.6, (
        f"the wordmark ({size}px) should stay well under the hero at desktop "
        f"({largest}px)")
