"""Dynamic "Save PDF" report builder: run bundle -> styled PDF bytes.

Same visual format as the hand-built run PDFs (navy cover band, stat tiles,
quartile tables, evidence table, regrouped sources, reference cards with
tappable in-text citations), but driven entirely by a recorded run bundle --
the same event payloads the frontend receives -- instead of a hand-written
markdown file. No per-run hardcoding: titles, tiles, tables and references
all derive from the bundle.

No LLM is used. Everything is deterministic reportlab Platypus plus regex
text transforms. Internal ``#ref-n`` links are real GoTo annotations
(verified in this environment); no post-processing pass is needed.
"""
from __future__ import annotations

import datetime
import logging
import re

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ palette

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, HRFlowable, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle, Frame)

NAVY = colors.HexColor("#1B2A4A")
TEAL = colors.HexColor("#0E7C7B")

#: Product name, used for the header band, the per-page footer and the
#: document metadata. One constant so the three cannot drift apart -- a
#: PDF whose cover says one thing and whose footer says another reads as
#: a bug in the export rather than as branding.
PROJECT_NAME = "Project Aletheia"
LIGHT_BG = colors.HexColor("#F3F5F9")
CARD_BORDER = colors.HexColor("#D9E2EC")
MUTED = colors.HexColor("#5B6B7F")
REF_BG = colors.HexColor("#EEF4FF")
REF_LINE = colors.HexColor("#B9CCF0")
# Used only for the "this run was incomplete" band on the cover.
AMBER_BG = colors.HexColor("#FFF7ED")
AMBER_LINE = colors.HexColor("#E8A33D")
INK = colors.HexColor("#232A35")

# ------------------------------------------------------------ text transforms


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.S)


def strip_html_comments(markdown: str) -> str:
    """Remove HTML comments before anything downstream reads the markdown.

    ``report.py`` deliberately hides non-critical pipeline advisories in an
    HTML comment -- correct for the ``.md`` export, where a renderer ignores
    it. The PDF has no such luxury: the body renderer skips comment blocks,
    but ``parse_references`` does not, so a comment that lands inside the
    References section falls through to the catch-all branch and is absorbed
    into the reference's ``meta`` field.

    That field is unbounded, so a 4KB advisory becomes a ~4300-character meta,
    which becomes a single-cell Table taller than the page frame, which
    ReportLab cannot split -- and one oversized card took the *entire* PDF
    down with it. Observed on the topological-qubits report, where reference
    15 rendered 718pt tall against a 703.89pt frame.

    Stripped once, here, so the body renderer, the reference parser, the
    note builder and the word/citation counters all see the same text.
    """
    return _HTML_COMMENT_RE.sub("", markdown or "")


_URL_RE = re.compile(r"https?://\S+|www\.\S+")


# A reference card is a single-cell Table, and ReportLab cannot place a row
# taller than the page frame -- it raises and the whole PDF is lost. These caps
# keep any one field from ever producing a card that tall. The longest meta seen
# in practice was ~4300 chars (an unstripped HTML comment); a 300-char cap puts
# the worst realistic card near 130pt against a 703.89pt frame.
META_MAX_CHARS = 300
NOTE_MAX_CHARS = 600


def _cap(text: str, limit: int) -> str:
    """Truncate on a word boundary, marking that something was dropped."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut or text[:limit]) + " …"


def fix_maths(raw):
    """Computer maths -> human scientific typesetting.

    ASCII-only super/sub so the built-in Helvetica always has the glyphs.
    URLs are shielded first: patterns like ``a/b`` must never rewrite inside
    a link.
    """
    urls = []

    def _hold(match):
        urls.append(match.group(0))
        return "@@URL%d@@" % (len(urls) - 1)

    t = _URL_RE.sub(_hold, raw)
    t = t.replace("$", "")
    t = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", t)
    t = re.sub(r"\\log\s*S", r"log <i>S</i>", t)
    t = re.sub(r"\bR\^2\b", r"R<super>2</super>", t)
    t = re.sub(r"\bN\s*=\s*(\d+)", r"<i>N</i> = \1", t)
    t = re.sub(r"pK_a", r"p<i>K</i><sub>a</sub>", t)
    t = t.replace("\\pm", "\u00b1")
    for d in ("MolWt", "MolLogP", "MolMR", "TPSA", "LabuteASA"):
        t = re.sub(rf"\b{d}\b", rf"<i>{d}</i>", t)
    t = re.sub(r"E_\{a,\s*dif\}", r"E<sub>a,dif</sub>", t)
    t = re.sub(r"10\^\{(-?\d+)\}", lambda m: f"10<super>{m.group(1)}</super>", t)
    t = re.sub(r"10\^(-?\d+)", lambda m: f"10<super>{m.group(1)}</super>", t)
    t = re.sub(r"cm\^2\s*s\^\{-1\}", r"cm<super>2</super>·s<super>-1</super>", t)
    t = re.sub(r"cm\^2\s*s\^-1", r"cm<super>2</super>·s<super>-1</super>", t)
    t = re.sub(r"kJ\s*mol\^\{-1\}", r"kJ·mol<super>-1</super>", t)
    t = re.sub(r"kJ\s*mol\^-1", r"kJ·mol<super>-1</super>", t)
    t = re.sub(r"kJ\s*mol-1", r"kJ·mol<super>-1</super>", t)
    t = re.sub(r"g\s*mol\^\{-1\}", r"g·mol<super>-1</super>", t)
    t = re.sub(r"g\s*mol\^-1", r"g·mol<super>-1</super>", t)
    t = re.sub(r"g\s*mol-1", r"g·mol<super>-1</super>", t)
    t = re.sub(r"g\s*cm\^\{-3\}", r"g·cm<super>-3</super>", t)
    t = re.sub(r"g\s*cm\^-3", r"g·cm<super>-3</super>", t)
    t = re.sub(r"g\s*cm-3", r"g·cm<super>-3</super>", t)
    t = re.sub(r"cm\^2", r"cm<super>2</super>", t)
    t = re.sub(r"s\^\{-1\}", r"s<super>-1</super>", t)
    t = re.sub(r"s\^-1", r"s<super>-1</super>", t)
    t = re.sub(r"(?<![A-Za-z])mol-1", r"mol<super>-1</super>", t)
    t = re.sub(r"(?<![A-Za-z])cm-3", r"cm<super>-3</super>", t)
    # Divisions read on-paper style (compact superscript-over-subscript)
    # anywhere except inside the shielded URLs restored below.
    t = re.sub(r"(\d[\d.,]*)\s*/\s*(\d[\d.,]*)",
               "<super>\\1</super>/\u2009<sub>\\2</sub>", t)
    for i, u in enumerate(urls):
        t = t.replace("@@URL%d@@" % i, u)
    return t


def _cite(n: str, sep: str = "", known=None) -> str:
    """One superscript citation, optionally carrying a trailing separator.

    The separator has to live *inside* the ``<super>`` element. ReportLab
    decides word boundaries from the tag structure, so a space or ``&nbsp;``
    placed between two ``<a>`` elements does not split them and adjacent
    citations fuse: ``[19][10][20]`` came out as the single number ``191020``,
    which is not just ugly but wrong -- it reads as one citation to a document
    numbered 191020. Zero-width characters are worse: this font has no glyph
    for them and they render as tofu boxes.

    Only the gap between citations needs the separator, so the caller passes it
    for every element except the last and no trailing space is left behind.
    """
    inner = f'<font size="8"><b>{n}</b></font>{sep}'
    if known is None or n in known:
        return (f'<a href="#ref-{n}" color="#1D4ED8">'
                f"<super>{inner}</super></a>")
    return f"<super>{inner}</super>"


# One or more adjacent bracketed groups: [3], [19][10][20], [2, 5, 9].
# Matching the whole run at once is what lets the separator be placed *between*
# its members; substituting each [n] separately makes that impossible, and the
# numbers fuse into "191020".
_BRACKET_RUN = re.compile(r"(?:\[[\d,\s]*\d[\d,\s]*\])+")


def fix_cites(raw, known=None):
    """[n], [n][m] and [a, b, c] -> tappable superscript links to #ref-n.

    Only numbers present in ``known`` become links (None means all): a link
    without a matching ``<a name>`` anchor crashes the whole build, so a
    citation with no reference entry stays plain superscript instead.
    """
    def run(m):
        nums = re.findall(r"\d+", m.group(0))
        return "".join(_cite(n, "&#160;" if i < len(nums) - 1 else "", known)
                       for i, n in enumerate(nums))

    return _BRACKET_RUN.sub(run, raw)


def para(raw, known=None):
    return fix_cites(fix_maths(esc(raw)), known)


def citation_count(markdown: str) -> int:
    return len(re.findall(r"\[\d+(?:\s*,\s*\d+)*\]|\]\s*\[", markdown or ""))


def build_ref_notes(refs: dict, markdown: str, retrieval_sources: list) -> dict:
    """Deterministic "what [n] gives you" notes: no model call.

    Each note combines what the pipeline already knows — findings count and
    strength from the evidence table, read grade from retrieval, and cited
    locations from the references section. Pure Python, same input same note.
    """
    table, _ = parse_evidence_table(markdown)
    by_ref = {}
    if table:
        for row in _evidence_rows(markdown):
            by_ref.update(row)
    status_by_url = {}
    for src in retrieval_sources or []:
        url = _norm_url(src.get("url", ""))
        if url:
            status_by_url[url] = src
    notes = {}
    for num, ref in (refs or {}).items():
        bits = []
        info = by_ref.get(num, {})
        if info.get("findings"):
            bits.append("%s finding(s)%s" % (
                info["findings"],
                f", {info['strength']} evidence" if info.get("strength") else ""))
        url = _norm_url(ref.get("url", ""))
        grade = ""
        for known, src in status_by_url.items():
            if known and (known in url or url in known):
                grade = src.get("status", "")
                break
        if grade and grade != "full":
            bits.append(f"read grade: {grade}")
        cited = "; ".join(ref.get("cited") or [])
        if cited:
            bits.append("Cited from: " + cited)
        notes[num] = "; ".join(bits) or "Cited in the report body."
    return notes


def _evidence_rows(markdown: str) -> list:
    """Yield {refnum: {findings, strength}} from the evidence table."""
    header, rows = parse_evidence_table(markdown)
    if not header:
        return []
    try:
        ref_i = next(i for i, h in enumerate(header) if h.strip().lower() == "ref")
        fnd_i = next(i for i, h in enumerate(header) if "finding" in h.strip().lower())
        str_i = next(i for i, h in enumerate(header) if "strength" in h.strip().lower())
    except StopIteration:
        return []
    out = []
    for row in rows:
        if max(ref_i, fnd_i, str_i) >= len(row):
            continue
        num = re.search(r"\[(\d+)\]", re.sub(r"<[^>]+>", "", row[ref_i]))
        if not num:
            continue
        out.append({num.group(1): {
            "findings": re.sub(r"<[^>]+>", "", row[fnd_i]).strip(),
            "strength": re.sub(r"<[^>]+>", "", row[str_i]).strip()}})
    return out


# -------------------------------------------------------------------- styles

def styles():
    return {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=21,
                                leading=25, textColor=NAVY, spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=9.5,
                                   leading=13, textColor=MUTED, spaceAfter=6),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13,
                             leading=16, textColor=NAVY, spaceBefore=14, spaceAfter=4,
                             keepWithNext=True),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10,
                               leading=14.5, textColor=INK,
                               alignment=4, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=10,
                                 leading=14.5, textColor=INK,
                                 leftIndent=16, bulletIndent=6, spaceAfter=3),
        "caption": ParagraphStyle("caption", fontName="Helvetica", fontSize=8.5,
                                  leading=12, textColor=MUTED, spaceAfter=2),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.5,
                                leading=12, textColor=colors.HexColor("#33404F"),
                                spaceAfter=3),
        "ref": ParagraphStyle("ref", fontName="Helvetica", fontSize=9.5,
                              leading=13.5, textColor=INK, spaceAfter=2),
        "tablecell": ParagraphStyle("tc", fontName="Helvetica", fontSize=8.5,
                                    leading=11.5, textColor=INK),
        "tablehead": ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=8.5,
                                    leading=11.5, textColor=colors.white),
    }


def make_header_footer(band_title, band_sub, footer_tag):
    def fn(canvas, doc):
        canvas.saveState()
        if doc.page == 1:
            canvas.setFillColor(NAVY)
            canvas.rect(0, A4[1] - 64, A4[0], 64, stroke=0, fill=1)
            canvas.setFillColor(colors.white)
            canvas.setFont("Helvetica-Bold", 11)
            canvas.drawString(54, A4[1] - 30, band_title)
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#B9C4D6"))
            canvas.drawString(54, A4[1] - 44, band_sub)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(54, 28, footer_tag)
        canvas.drawRightString(A4[0] - 54, 28, f"Page {doc.page}")
        canvas.setStrokeColor(CARD_BORDER)
        canvas.setLineWidth(0.6)
        canvas.line(54, 36, A4[0] - 54, 36)
        canvas.restoreState()
    return fn


def _duration(seconds) -> str:
    """Human duration for the cover: 4m 12s, 1h 03m 09s."""
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return "unknown"
    if total < 0:
        return "unknown"
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def stat_tile(S, value, label):
    inner = [[Paragraph(f'<font size="13"><b>{esc(value)}</b></font>', S["small"])],
             [Paragraph(f'<font color="#5B6B7F">{esc(label)}</font>', S["caption"])]]
    t = Table(inner, colWidths=[95])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ("BOX", (0, 0), (-1, -1), 0.7, CARD_BORDER),
        ("INNERPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def overview_table(S, rows):
    ot = Table([[P("<b>" + esc(k) + "</b>", S, "tablecell"),
                 P(esc(v), S, "tablecell")] for k, v in rows],
               colWidths=[72, 338])
    ot.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("ROUNDEDCORNERS", [5, 5, 5, 5]),
        ("BOX", (0, 0), (-1, -1), 0.7, CARD_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, CARD_BORDER),
        ("INNERPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return ot


def P(txt, S, style="body"):
    return Paragraph(txt, S[style])


def C(n, sep=""):
    return _cite(str(n), sep)


def simple_table(S, header, rows, widths):
    data = [[P(f"<b>{esc(h)}</b>", S, "tablehead") for h in header]]
    for row in rows:
        data.append([P(cell, S, "tablecell") for cell in row])
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, CARD_BORDER),
        ("INNERPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def section_rule():
    return HRFlowable(width="100%", thickness=0.7, color=CARD_BORDER,
                      spaceAfter=4, spaceBefore=0)


_DISCLOSURE_RE = re.compile(r"^#{2,3}\s+(.+)$", re.M)


def trailing_disclosures(markdown: str) -> list[tuple[str, str]]:
    """Sections that sit *after* the References block, as (heading, body).

    ``report.py`` appends the run's integrity disclosure last, so everything
    between the References heading and end-of-document is machine-generated
    commentary about the run rather than report body.

    This has to be extracted explicitly. ``build_pdf`` cuts the report body at
    "## Evidence by Source", which sits *before* References, and ``
    render_refs`` only draws the reference cards. So the disclosure landed in
    neither renderer and was dropped from the PDF entirely -- on a run that
    was PARTIAL, the evidence-integrity warning appeared in the .md and
    nowhere in the document a reader actually receives.

    That is the exact failure the disclosure exists to prevent ("a pipeline
    problem hidden in markup is a pipeline problem nobody reads"), so it is
    rendered here rather than inferred from the bundle.
    """
    lines = (markdown or "").split("\n")
    try:
        start = next(i for i, ln in enumerate(lines)
                     if ln.strip() == "## References")
    except StopIteration:
        return []
    tail = "\n".join(lines[start + 1:])
    if not tail.strip():
        return []
    chunks = _DISCLOSURE_RE.split(tail)
    out = []
    for i in range(1, len(chunks), 2):
        heading = chunks[i].strip()
        body = chunks[i + 1] if i + 1 < len(chunks) else ""
        if heading:
            out.append((heading, body.strip()))
    return out


def render_disclosures(story, S, markdown, known=None):
    """Draw the post-References run disclosure, e.g. "How this run went"."""
    blocks = trailing_disclosures(markdown)
    for heading, body in blocks:
        story.append(Spacer(1, 10))
        story.append(P(esc(heading), S, "h2"))
        story.append(section_rule())
        for block in re.split(r"\n\s*\n", body.strip()):
            block = block.strip()
            if not block or block.startswith("<!--"):
                continue
            if block.startswith("- ") or block.startswith("* "):
                for item in re.split(r"\n(?=\s*[-*] )", block):
                    item = item.strip().lstrip("-* ").strip()
                    if item:
                        render_item(story, S, item, known)
            else:
                story.append(P(para(block, known), S))


# ------------------------------------------------------- section body parser

def render_sections(story, S, body_md, known=None):
    """Split markdown on ## headings; bullets (incl. embedded single-newline
    lists) become items; everything else becomes justified body text."""
    chunks = re.split(r"^## (.+)$", body_md, flags=re.M)
    for i in range(1, len(chunks), 2):
        heading = chunks[i].strip()
        content = chunks[i + 1] if i + 1 < len(chunks) else ""
        story.append(P(esc(heading), S, "h2"))
        story.append(section_rule())
        for block in re.split(r"\n\s*\n", content.strip()):
            block = block.strip()
            if not block or block.startswith("<!--"):
                continue
            marker = None
            if block.startswith("* "):
                marker = "* "
            elif block.startswith("- "):
                marker = "- "
            if marker:
                render_item(story, S, block[len(marker):].strip(), known)
            elif "\n- " in block or "\n* " in block:
                lines = block.split("\n")
                intro, items, buf = [], [], []
                for ln in lines:
                    s = ln.strip()
                    if s.startswith("- ") or s.startswith("* "):
                        if buf and not items:
                            intro, buf = buf, []
                        items.append(s[2:].strip())
                    else:
                        buf.append(ln)
                if not items:
                    story.append(P(para(block), S))
                    continue
                rest = "\n".join(intro or buf).strip()
                if rest:
                    story.append(P(para(rest, known), S))
                for core in items:
                    render_item(story, S, core, known)
            else:
                story.append(P(para(block, known), S))


def render_item(story, S, core, known=None):
    m = re.match(r"^(.{3,90}?):(.*)$", core, flags=re.S)
    if m and "\n" not in m.group(1):
        story.append(P(f"<b>{para(m.group(1).strip(), known)}:</b>{para(m.group(2), known)}", S))
    else:
        story.append(P("\u2022&nbsp;&nbsp;" + para(core, known), S, "bullet"))


def render_refs(story, S, refs):
    story.append(P("References \u2014 tap a number anywhere to land here", S, "h2"))
    story.append(section_rule())
    story.append(P("Each card explains what the source contributed, so a tap on a superscript "
                   "number in the text resolves to meaning, not just a URL.", S, "caption"))
    for n, title, link, meta, note in refs:
        url_text = (f'<link href="{esc(link)}" color="#1D4ED8">{esc(link)}</link>'
                    if link else "")
        html = (f'<a name="ref-{n}"/><font size="13" color="#1B2A4A"><b>[{n}]</b></font>'
                f"&nbsp;&nbsp;<b>{esc(title)}</b><br/>"
                f'{url_text}'
                f' &nbsp;<font color="#5B6B7F">{esc(meta)}</font><br/>'
                f'<b>Note \u2014 what [{n}] gives you:</b> {esc(note)}')
        # splitInRow lets an over-long card break across pages instead of
        # raising LayoutError and taking the entire document with it. Same
        # design rule as the rest of the project: one bad element degrades,
        # it does not take the run down.
        card = Table([[Paragraph(html, S["ref"])]], colWidths=[410],
                     splitInRow=1)
        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), REF_BG),
            ("ROUNDEDCORNERS", [6, 6, 6, 6]),
            ("BOX", (0, 0), (-1, -1), 0.8, REF_LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(card)
        story.append(Spacer(1, 6))


# ------------------------------------------------- bundle parsing (generic)

REF_START = re.compile(r"^(\d+)\.\s+\*\*(.+?)\*\*\s*\u2014\s*(.*)$")


def parse_references(markdown: str) -> dict:
    """Reference number -> {title, url, meta, cited, notes}.

    Tolerates titles wrapped across lines, exactly as the generator emits.
    """
    refs, current = {}, None
    lines = (markdown or "").split("\n")
    i, n = 0, len(lines)
    try:
        start = next(j for j, ln in enumerate(lines)
                     if ln.strip() == "## References")
    except StopIteration:
        return refs
    i = start + 1
    while i < n:
        line = lines[i]
        m = REF_START.match(line)
        if m:
            current = {"title": m.group(2).strip(), "meta": m.group(3).strip(),
                       "url": "", "cited": [], "notes": []}
            refs[m.group(1)] = current
            i += 1
            continue
        m2 = re.match(r"^(\d+)\.\s+\*\*(.*)$", line)
        if m2:
            num, buf = m2.group(1), [m2.group(2)]
            i += 1
            while i < n and "**" not in buf[-1]:
                buf.append(lines[i].strip())
                i += 1
            joined = " ".join(buf)
            title, sep, meta = joined.partition("**")
            current = {"title": title.strip(),
                       "meta": meta.lstrip("\u2014- ").strip() if sep else "",
                       "url": "", "cited": [], "notes": []}
            refs[num] = current
            continue
        if current is None:
            i += 1
            continue
        s = line.strip()
        if re.match(r"^#+\s", line) or (not s and i + 1 < n
                                        and re.match(r"^#+\s", lines[i + 1])):
            break
        if not s:
            i += 1
            continue
        url = re.search(r"<(https?://[^>]+|10\.\S+?)>", s)
        if url:
            current["url"] = url.group(1)
        elif s.lower().startswith("cited from:"):
            current["cited"].append(s[len("cited from:"):].strip())
        elif s.startswith("*") and s.endswith("*") and len(s) > 4:
            current["notes"].append(s.strip("*").strip())
        else:
            current["meta"] += " " + s
        i += 1
    return refs


def parse_evidence_table(markdown: str):
    """(header, rows) from the ## Evidence by Source pipe table, if present."""
    lines = (markdown or "").split("\n")
    try:
        start = next(j for j, ln in enumerate(lines)
                     if ln.strip() == "## Evidence by Source")
    except StopIteration:
        return None, []
    header, rows, current = None, [], None
    width = 0
    i = start + 1
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            if current is not None:
                break
            i += 1
            continue
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.match(r"^:?-{2,}:?$", c) for c in cells if c):
                i += 1
                continue
            if header is None:
                header = cells
                width = len(header)
            else:
                current = cells
                rows.append(current)
            i += 1
            continue
        if current is not None and not re.match(r"^#+", line):
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) > 1:
                current[-1] += " " + parts[0]
                current.extend(parts[1:])
            else:
                current[-1] += " " + line.strip()
            i += 1
            continue
        break
    fixed = []
    for row in rows:
        if width and len(row) > width:
            surplus = len(row) - width
            row = row[:2] + [" | ".join(row[2:2 + surplus + 1])] + row[2 + surplus + 1:]
        fixed.append(row)
    return header, fixed


_QROW = re.compile(r"^(?:Quartile\s*)?\(([^)\]]+)\]")
_QRMSE = re.compile(r"RMSE\s+([\d.]+)")
_QN = re.compile(r"\(\$?N=(\d+)\$?\)")


def quartile_tables(body_md):
    """Consecutive quartile bullet runs -> [(caption, rows)] for real tables."""
    found = []
    lines = (body_md or "").split("\n")
    i = 0
    caption = None
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("- "):
            run = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                run.append(lines[i].strip()[2:])
                i += 1
            rows = []
            for item in run:
                m, rmse, nval = (_QROW.search(item), _QRMSE.search(item),
                                 _QN.search(item))
                if not (m and rmse and nval):
                    break
                rows.append((m.group(1).strip(), rmse.group(1), nval.group(1)))
            if len(rows) >= 2 and len(rows) == len(run):
                found.append((caption, rows))
            caption = None
            continue
        caption = s if s.endswith(":") else None
        i += 1
    return found


# ------------------------------------------------------------------- builder

def _words(text: str) -> int:
    return len((text or "").split())


def _norm_url(url: str) -> str:
    url = (url or "").lower().strip()
    if url.startswith("<") and url.endswith(">"):
        url = url[1:-1]
    return url.removeprefix("www.").rstrip("/")


def _ref_urls(refs: dict) -> set:
    urls = set()
    for ref in refs.values():
        url = (ref.get("url") or "").strip()
        if not url:
            continue
        urls.add(_norm_url(url))
        if not url.startswith("http"):
            urls.add(_norm_url("https://doi.org/" + url))
    return urls


def build_pdf(bundle: dict, rounds_label: str = "",
              throttle_label: str = "") -> bytes:
    """Render a recorded run bundle to PDF bytes. Raises on empty reports."""
    import io as _io
    from .export import slugify as _slugify

    question = bundle.get("question", "")
    markdown = strip_html_comments(bundle.get("report", ""))
    if not markdown.strip():
        raise ValueError("no report content to render")
    today = datetime.date.today().isoformat()

    stats = bundle.get("stats") or {}
    retrieval = bundle.get("retrieval") or {}
    analysis = bundle.get("analysis") or {}
    status = bundle.get("status") or {}
    plan = bundle.get("plan") or {}
    searches = bundle.get("searches") or []
    gaps = bundle.get("gaps") or []
    sources = (retrieval.get("sources") or [])

    records = stats.get("records", analysis.get("records", 0))
    distinct_sources = stats.get("sources", 0)
    report_words = _words(markdown)
    cites = citation_count(markdown)

    subqs = (plan.get("subquestions") or [])
    query_count = len(searches)
    result_count = sum(int(s.get("count") or 0) for s in searches)
    errors = [s for s in searches if s.get("error")]

    gap = gaps[-1] if gaps else {}
    if gap.get("sufficient"):
        gap_verdict, gap_note = "sufficient", "evidence looks sufficient"
    elif (gap.get("missing") or "").lower().find("round limit") >= 0:
        gap_verdict, gap_note = "round limit", "round limit reached"
    elif gap.get("follow_ups"):
        follows = gap["follow_ups"]
        gap_verdict = f"{len(follows)} follow-up{'s' if len(follows) != 1 else ''}"
        gap_note = gap.get("missing", "")
    else:
        gap_verdict, gap_note = "checked", gap.get("missing", "")

    headline = status.get("headline") or status.get("status", "REPORT").upper()
    # A stat tile is for a number or a single word. The full headline -- a
    # PARTIAL verdict plus a sentence explaining the degraded evidence base --
    # was being crammed into a 130pt column and wrapped every two or three
    # words. Keep the verdict in the tile and give the reason its own full-width
    # band underneath, where it can be read.
    state = (status.get("status") or "completed").upper()
    short_verdict = {"COMPLETED": "COMPLETE", "PARTIAL": "PARTIAL",
                     "FAILED": "FAILED"}.get(state, state)
    reason = ""
    if "—" in headline:
        reason = headline.split("—", 1)[1].strip()
    missing_sections = [str(m) for m in (status.get("missing_sections") or [])]

    S = styles()
    story = []

    # ---- cover ---------------------------------------------------------
    story.append(P(question.strip().rstrip("?") or "Research Report", S, "title"))
    subtitle_bits = [
        f"{distinct_sources} distinct sources",
        f"{records} traceable findings",
        f"{report_words:,} words",
        f"{cites} citations",
    ]
    # How long the run actually took, stamped when the report landed. Without
    # it a reader has no idea whether this was a two-minute lookup or a
    # twenty-minute investigation, which is much of what makes the artefact
    # believable.
    elapsed = bundle.get("elapsed_s")
    if elapsed:
        subtitle_bits.insert(0, f"generated in {_duration(elapsed)}")
    story.append(P(
        f"Research synthesis generated {today} &nbsp;\u2022&nbsp; "
        + " &nbsp;\u2022&nbsp; ".join(subtitle_bits),
        S, "subtitle"))
    story.append(HRFlowable(width="100%", thickness=1.2, color=TEAL,
                            spaceAfter=8, spaceBefore=2))

    # ---- run overview --------------------------------------------------
    story.append(P("Run overview \u2014 as shown live in the app", S, "h2"))
    story.append(P("What the frontend displayed while this run executed: plan, searches, "
                   "retrieval grades, gap verdict and the completion banner.", S, "caption"))
    tiles = Table([
        [stat_tile(S, str(len(subqs)), "sub-questions planned"),
         stat_tile(S, str(query_count), f"queries  \u2022  {result_count} results"),
         stat_tile(S, f"{retrieval.get('retrieved', 0)} / {retrieval.get('attempted', 0)}",
                   f"docs read ({retrieval.get('full_text', 0)} full)")],
        [stat_tile(S, gap_verdict, f"gap check{', ' + gap_note if gap_note else ''}"[:60]),
         stat_tile(S, short_verdict,
                   "all sections generated" if state == "COMPLETED"
                   else "report is incomplete — see below"),
         stat_tile(S, str(cites), "citations in report")],
        [stat_tile(S, _duration(elapsed) if elapsed else "—", "run duration"),
         stat_tile(S, str(len(gaps)) or "—", "gap checks run"),
         stat_tile(S, rounds_label or "—", "rounds requested")],
    ], colWidths=[130, 130, 130], spaceBefore=4, spaceAfter=4)
    tiles.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 4),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    story.append(tiles)
    story.append(Spacer(1, 2))

    # The reason a run came out partial, given room to be read. Without this
    # band the only trace of a degraded evidence base was a column of text
    # wrapping two words per line inside a stat tile.
    if state != "COMPLETED":
        band_rows = []
        if reason:
            band_rows.append(P(f"<b>Why:</b> {esc(reason)}", S, "body"))
        if missing_sections:
            # An integrity failure is both the "why" and a missing entry, so
            # listing it twice reads as two separate problems. Show only what
            # the reason does not already cover.
            extra = [m for m in missing_sections
                     if not (reason and reason[:40] in m)]
            if extra:
                shown = extra[:6]
                listed = "<br/>".join(f"• {esc(m)}" for m in shown)
                if len(extra) > len(shown):
                    listed += f"<br/>• …and {len(extra) - len(shown)} more"
                band_rows.append(P(f"<b>Outstanding:</b><br/>{listed}", S, "body"))
        band = Table([[cell] for cell in band_rows], colWidths=[400])
        band.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), AMBER_BG),
            ("BOX", (0, 0), (-1, -1), 0.7, AMBER_LINE),
            ("INNERPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(band)
        story.append(Spacer(1, 6))

    used_sources = sorted({(s.get("source") or "") for s in searches if s.get("source")})
    search_line = (f"{query_count} queries across {', '.join(used_sources) or 'no sources'}"
                   + (f"; {len(errors)} failed ({errors[0].get('error', '')[:80]})"
                      if errors else ""))
    # Join only the parts that exist: throttle_label is empty when the caller
    # did not pin a throttle mode, which used to render "(2 rounds, )".
    run_label = ", ".join(p for p in (rounds_label, throttle_label) if p)
    story.append(overview_table(S, [
        ("Question", f"{question.strip()}"
         + (f" ({run_label})" if run_label else "")),
        ("Search", search_line),
        ("Retrieval", f"{retrieval.get('retrieved', 0)} full-text reads, "
                      f"{retrieval.get('partial', 0)} partial, "
                      f"{retrieval.get('metadata_only', 0)} snippet-only; "
                      f"{retrieval.get('total_words', retrieval.get('words', 0)):,} words in "
                      f"{retrieval.get('chunks', 0)} chunks; "
                      f"{retrieval.get('references_followed', 0)} reference links followed."),
        ("Gap check", f"{gap_verdict}" + (f" \u2014 {gap_note}" if gap_note else "")),
    ]))
    story.append(P(f"Reading tip \u2014 every superscript number is tappable: it jumps "
                   "to the source card in References, where a short note says what that source "
                   "contributed.", S, "small"))

    # ---- key numbers ---------------------------------------------------
    body_md = markdown
    try:
        cut = body_md.index("## Evidence by Source")
        body_md, evidence_md = body_md[:cut], body_md[cut:]
    except ValueError:
        evidence_md = ""
    try:
        first_section = next(i for i, ln in enumerate(body_md.split("\n"))
                             if ln.startswith("## "))
        body_md = "\n".join(body_md.split("\n")[first_section:])
    except StopIteration:
        pass
    by_quality = stats.get("by_quality", {}) or {}
    dims = analysis.get("dimensions") or []
    story.append(P("Key numbers at a glance", S, "h2"))
    story.append(P("Counts straight from the run bundle \u2014 findings by quality tier, "
                   "dimensions covered, and words on both sides of the pipeline.", S, "caption"))
    quality_line = ", ".join(f"{v} {k}" for k, v in sorted(by_quality.items())) or "ungraded"
    krows = [
        [P(f"<b>Findings</b><br/>{records} extracted and verified<br/>{esc(quality_line)}",
           S, "tablecell"),
         P(f"<b>Dimensions</b><br/>{esc(', '.join(dims) or 'none listed')}<br/>"
           f"{analysis.get('sources_summarised', 0)} sources individually assessed",
           S, "tablecell")],
        [P(f"<b>Retrieved</b><br/>{retrieval.get('total_words', retrieval.get('words', 0)):,} words "
           f"in {retrieval.get('chunks', 0)} chunks", S, "tablecell"),
         P(f"<b>Report</b><br/>{report_words:,} words \u2022 {cites} citations<br/>{esc(headline)}",
           S, "tablecell")],
    ]
    kt = Table(krows, colWidths=[205, 205])
    kt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), REF_BG),
        ("ROUNDEDCORNERS", [5, 5, 5, 5]),
        ("BOX", (0, 0), (-1, -1), 0.7, REF_LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, REF_LINE),
        ("INNERPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(kt)
    story.append(Spacer(1, 4))
    for _caption, rows in quartile_tables(body_md):
        story.append(simple_table(S, ["Range", "RMSE (log units)", "N"],
                                  [[c, r, n] for c, r, n in rows],
                                  [230, 90, 90]))
        story.append(Spacer(1, 2))

    # ---- report body ---------------------------------------------------
    story.append(P("Report", S, "h2"))
    story.append(section_rule())
    known_refs = set(parse_references(markdown))
    render_sections(story, S, body_md, known_refs)

    # ---- evidence by source --------------------------------------------
    header, rows = parse_evidence_table(evidence_md or markdown)
    if header and rows:
        story.append(P("Evidence by source", S, "h2"))
        story.append(section_rule())
        total_w = 410
        if len(header) >= 7:
            widths = [30, 115, 60, 52, 55, 40, 48]
        else:
            each = total_w // max(1, len(header))
            widths = [each] * len(header)
        data = [[P(f"<b>{esc(h)}</b>", S, "tablehead") for h in header]]
        for row in rows:
            cells = []
            for j, cell in enumerate(row):
                text = re.sub(r"\s+", " ", cell).strip()
                if j == 0:
                    num = re.search(r"\[(\d+)\]", text)
                    if num and num.group(1) in known_refs:
                        text = (f'<a href="#ref-{num.group(1)}" color="#1D4ED8">'
                                f"<b>[{num.group(1)}]</b></a>")
                    else:
                        text = esc(text)
                else:
                    text = para(text, known_refs)
                cells.append(P(text or "\u2014", S, "tablecell"))
            while len(cells) < len(header):
                cells.append(P("\u2014", S, "tablecell"))
            data.append(cells[:len(header)])
        evt = Table(data, colWidths=widths[:len(header)], repeatRows=1)
        evt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ("GRID", (0, 0), (-1, -1), 0.5, CARD_BORDER),
            ("INNERPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(evt)

    # ---- sources regrouped ----------------------------------------------
    ref_urls = _ref_urls(parse_references(markdown))
    cited, background, snippets = [], [], []
    for src in sources:
        url = _norm_url(src.get("url", ""))
        title = src.get("title") or src.get("url", "")
        method = src.get("method", "") or src.get("type", "")
        words = src.get("words", 0)
        grade = (src.get("status") or "").lower()
        label = f"{title} ({method}, {words:,} words)"
        if grade == "metadata_only" or method == "search-metadata":
            snippets.append(title)
        elif url and any(url in r or r in url for r in ref_urls if r):
            cited.append((label, src))
        else:
            background.append((label, src))
    story.append(P("Sources retrieved \u2014 what was actually read", S, "h2"))
    story.append(section_rule())
    story.append(P("Cited evidence first (full reads behind the references), then background "
                   "reading, then snippet-only leads. Tap a reference number anywhere above to jump "
                   "to its card below.", S, "caption"))
    if cited:
        story.append(simple_table(S, ["Cited document", "Size", "Grade"],
                                  [[esc(t), f"{s.get('words', 0):,} words",
                                    s.get("status", "")] for t, s in cited],
                                  [250, 90, 70]))
        story.append(Spacer(1, 4))
    if background:
        story.append(P("Background reading (retrieved in full, informs context, not directly cited):",
                       S, "caption"))
        brows = [[P("\u2022 " + esc(background[j][0]), S, "small"),
                  P("\u2022 " + esc(background[j + 1][0])
                    if j + 1 < len(background) else "", S, "small")]
                 for j in range(0, len(background), 2)]
        bt = Table(brows, colWidths=[205, 205])
        bt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
            ("ROUNDEDCORNERS", [5, 5, 5, 5]),
            ("BOX", (0, 0), (-1, -1), 0.7, CARD_BORDER),
            ("INNERPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(bt)
        story.append(Spacer(1, 4))
    if snippets:
        story.append(P("Search-only leads (snippet only \u2014 not used as evidence, "
                       f"{len(snippets)} items): "
                       + esc("; ".join(snippets[:12]))
                       + ("; \u2026plus %d further snippets (see run log)"
                          % (len(snippets) - 12) if len(snippets) > 12 else ""),
                       S, "small"))

    # ---- references -------------------------------------------------------
    refs = parse_references(markdown)
    ref_notes = build_ref_notes(refs, markdown,
                                (retrieval.get("sources") or []))
    cards = []
    for num in sorted(refs, key=int):
        ref = refs[num]
        link = (ref.get("url") or "").strip()
        shown = link if link.startswith("http") else ("https://doi.org/" + link if link else "")
        meta_bits = [b for b in [ref.get("meta", ""), ] if b]
        meta = " ".join(meta_bits)
        # build_ref_notes() already prefixes "Cited from:" for the same list.
        # Appending it again here duplicated the whole thing on every card that
        # had any locations, which is most of them.
        note = ref_notes.get(num, "")
        meta = _cap(meta, META_MAX_CHARS)
        note = _cap(note, NOTE_MAX_CHARS)
        cards.append((num, ref.get("title", ""), shown or link,
                      meta, note or "Cited in the report body."))
    render_refs(story, S, cards)
    # Anything the generator wrote after References -- currently the run's
    # integrity disclosure. Rendered last so it sits with the sources rather
    # than being folded silently into the body.
    render_disclosures(story, S, markdown)

    notes = analysis.get("notes") or []
    if notes:
        story.append(P("Pipeline notes: " + "; ".join(str(x) for x in notes), S, "small"))
    # Missing sections are not repeated at the end of the document: the cover
    # carries them in a readable "Why / Outstanding" band, and the report body
    # carries the partial banner and its own "How this run went" section. A
    # fourth copy of the same sentence added nothing except length.

    slug = _slugify(question)
    today_str = datetime.date.today().isoformat()
    band_title = f"{PROJECT_NAME.upper()}  \u2014  RESEARCH REPORT"
    band_sub = f"{question.strip()[:90]}  \u2022  {today_str}"
    footer_tag = f"{PROJECT_NAME}  \u2022  {slug}"

    buf = _io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=A4,
                          leftMargin=54, rightMargin=54, topMargin=78, bottomMargin=48,
                          title=question.strip()[:120], author=PROJECT_NAME)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame],
                                       onPage=make_header_footer(band_title, band_sub,
                                                                 footer_tag))])
    doc.build(story)
    return buf.getvalue()
