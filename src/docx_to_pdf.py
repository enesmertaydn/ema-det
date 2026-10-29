# -*- coding: utf-8 -*-
"""
docx_to_pdf.py <in.docx> <out.pdf> -- plain-paragraph docx (runs with bold/italic, no tables or images) to PDF with reportlab and
Times New Roman, for portals that accept PDF only. Keeps size, alignment, spacing and indents of each paragraph; adds page numbers.
"""
import sys
from xml.sax.saxutils import escape

from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate

F = r"C:\Windows\Fonts"
for name, f in (("TNR", "times.ttf"), ("TNR-B", "timesbd.ttf"), ("TNR-I", "timesi.ttf"), ("TNR-BI", "timesbi.ttf")):
    pdfmetrics.registerFont(TTFont(name, F + "\\" + f))
pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")
ALIGN = {None: 4, 0: 0, 1: 1, 2: 2, 3: 4}


def markup(p):
    out = []
    for r in p.runs:
        t = escape(r.text)
        if not t:
            continue
        if r.bold:
            t = "<b>%s</b>" % t
        if r.italic:
            t = "<i>%s</i>" % t
        out.append(t)
    return "".join(out)


def main(src, dst):
    d = Document(src)
    sec = d.sections[0]
    flow = []
    for p in d.paragraphs:
        m = markup(p)
        if not m.strip():
            flow.append(Paragraph("&nbsp;", ParagraphStyle("e", fontName="TNR", fontSize=6, leading=6)))
            continue
        size = max((r.font.size.pt for r in p.runs if r.font.size), default=11)
        pf = p.paragraph_format
        st = ParagraphStyle(
            "p", fontName="TNR", fontSize=size, leading=size * 1.22,
            alignment=ALIGN.get(int(p.alignment) if p.alignment is not None else None, 4),
            spaceBefore=pf.space_before.pt if pf.space_before else 0, spaceAfter=pf.space_after.pt if pf.space_after else 0,
            leftIndent=pf.left_indent.cm * cm if pf.left_indent else 0,
            firstLineIndent=pf.first_line_indent.cm * cm if pf.first_line_indent else 0,
            keepWithNext=bool(pf.keep_with_next),
        )
        flow.append(Paragraph(m, st))

    def footer(c, doc):
        c.saveState()
        c.setFont("TNR", 9)
        c.drawCentredString(A4[0] / 2, 1.2 * cm, str(doc.page))
        c.restoreState()

    SimpleDocTemplate(dst, pagesize=A4, leftMargin=sec.left_margin.cm * cm, rightMargin=sec.right_margin.cm * cm,
                      topMargin=sec.top_margin.cm * cm, bottomMargin=sec.bottom_margin.cm * cm,
                      title=d.core_properties.title or "Response to reviewers", author="").build(flow, onFirstPage=footer, onLaterPages=footer)
    print("saved", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
