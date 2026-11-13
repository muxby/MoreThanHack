"""Render TECHNICAL.md as the submission PDF.

    pip install reportlab matplotlib      # documentation only; not needed for the pipeline
    python scripts/build_pdf.py           # -> submissions/Team Claude.pdf

A small Markdown subset is supported -- exactly what TECHNICAL.md uses: #/## headings, paragraphs,
"-" bullets (with indented continuation lines), pipe tables, **bold**, *italic*, `code` and links.
DejaVu fonts (shipped with matplotlib) are used because the built-in PDF fonts lack glyphs such as
≈, ↔, ≤ and Σ.
"""

import argparse
import os
import re

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (ListFlowable, ListItem, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")

ACCENT = colors.HexColor("#1f4e79")
RULE = colors.HexColor("#c8d3df")
SHADE = colors.HexColor("#eef3f8")


def register_fonts():
    for name, file in (("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf"),
                       ("Sans-Italic", "DejaVuSans-Oblique.ttf"),
                       ("Sans-BoldItalic", "DejaVuSans-BoldOblique.ttf"),
                       ("Mono", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name, os.path.join(FONTS, file)))
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic",
                                  boldItalic="Sans-BoldItalic")


def styles():
    base = ParagraphStyle("body", fontName="Sans", fontSize=9.5, leading=13.5, alignment=TA_LEFT,
                          spaceAfter=5)
    return {
        "title": ParagraphStyle("title", parent=base, fontName="Sans-Bold", fontSize=17,
                                leading=21, textColor=ACCENT, spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle", parent=base, fontSize=10, textColor=colors.grey,
                                   spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base, fontName="Sans-Bold", fontSize=12.5, leading=16,
                             textColor=ACCENT, spaceBefore=10, spaceAfter=4, keepWithNext=1),
        "body": base,
        "code": ParagraphStyle("code", parent=base, fontName="Mono", fontSize=8.3, leading=11,
                               backColor=SHADE, borderPadding=5, leftIndent=5, spaceBefore=3,
                               spaceAfter=9),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8.5, leading=11, spaceAfter=0),
        "cellhead": ParagraphStyle("cellhead", parent=base, fontName="Sans-Bold", fontSize=8.5,
                                   leading=11, spaceAfter=0, textColor=colors.white),
    }


def inline(text):
    """Markdown inline markup -> reportlab paragraph markup."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r'<font face="Mono" size="8.5">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<font color="#1f4e79">\1</font>', text)
    return text


def table(lines, st, width):
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        rows.append(cells)
    head, body = rows[0], rows[1:]
    data = [[Paragraph(inline(c), st["cellhead"]) for c in head]]
    data += [[Paragraph(inline(c), st["cell"]) for c in r] for r in body]
    n = len(head)
    # column widths proportional to the longest cell (clamped), so short columns stay compact
    longest = [max(len(re.sub(r"[*`]", "", r[k])) for r in rows) for k in range(n)]
    weights = [min(max(x, 12), 60) for x in longest]
    widths = [w / float(sum(weights)) for w in weights]
    t = Table(data, colWidths=[w * width for w in widths], repeatRows=1, hAlign="LEFT")
    style = [("BACKGROUND", (0, 0), (-1, 0), ACCENT),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), SHADE))
    t.setStyle(TableStyle(style))
    return t


def build(md_path, pdf_path, team):
    register_fonts()
    st = styles()
    with open(md_path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="{} - Technical document".format(team), author=team,
                            subject="More than a Hack 2027, Phase 1: ball possession")
    width = doc.width
    story = []
    i = 0
    paragraph, bullets, numbered = [], [], []

    def flush_paragraph():
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), st["body"]))
            paragraph.clear()

    def flush_bullets():
        if bullets:
            items = [ListItem(Paragraph(inline(b), st["body"]), leftIndent=12) for b in bullets]
            story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=12,
                                      bulletFontName="Sans", bulletFontSize=8))
            bullets.clear()

    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            flush_paragraph(), flush_bullets()
            story.append(Paragraph(inline(line[2:]), st["title"]))
            story.append(Paragraph("{} · More than a Hack 2027, Phase 1: ball possession".format(
                team), st["subtitle"]))
        elif line.startswith("## "):
            flush_paragraph(), flush_bullets()
            story.append(Paragraph(inline(line[3:]), st["h2"]))
        elif line.lstrip().startswith("|"):
            flush_paragraph(), flush_bullets()
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            story.append(table(block, st, width))
            story.append(Spacer(1, 6))
            continue
        elif line.startswith("```"):
            flush_paragraph(), flush_bullets()
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            story.append(Preformatted("\n".join(code), st["code"]))
        elif re.match(r"\d+\. ", line):
            flush_paragraph(), flush_bullets()
            numbered.append(re.sub(r"^\d+\. ", "", line).strip())
            while i + 1 < len(lines) and lines[i + 1].startswith("   ") and lines[i + 1].strip():
                i += 1
                numbered[-1] += " " + lines[i].strip()
            if i + 1 >= len(lines) or not re.match(r"\d+\. ", lines[i + 1]):
                items = [ListItem(Paragraph(inline(b), st["body"]), leftIndent=14)
                         for b in numbered]
                story.append(ListFlowable(items, bulletType="1", leftIndent=14,
                                          bulletFontName="Sans-Bold", bulletFontSize=9))
                numbered.clear()
        elif line.startswith("- "):
            flush_paragraph()
            bullets.append(line[2:].strip())
        elif line.startswith("  ") and bullets and line.strip():
            bullets[-1] += " " + line.strip()
        elif not line.strip():
            flush_paragraph()
            flush_bullets()
        else:
            flush_bullets()
            paragraph.append(line.strip())
        i += 1
    flush_paragraph(), flush_bullets()

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Sans", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(doc_.leftMargin, 9 * mm, "{} · github.com/muxby/MoreThanHack".format(team))
        canvas.drawRightString(A4[0] - doc_.rightMargin, 9 * mm, "page {}".format(doc_.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default="Team Claude")
    parser.add_argument("--md", default=os.path.join(HERE, "TECHNICAL.md"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = args.out or os.path.join(HERE, "submissions", args.team + ".pdf")
    build(args.md, out, args.team)
    print("wrote", os.path.relpath(out, HERE))


if __name__ == "__main__":
    main()
