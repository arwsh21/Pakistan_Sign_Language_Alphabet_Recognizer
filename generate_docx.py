"""Generates docs/project_report.docx with professional typography, styled formula boxes, and embedded figures."""

import os
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls


def set_cell_background(cell, fill_hex):
  """Sets the background color of a table cell."""
  tcPr = cell._element.get_or_add_tcPr()
  shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
  tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
  """Sets internal padding for a table cell."""
  tcPr = cell._element.get_or_add_tcPr()
  tcMar = parse_xml(
      f'<w:tcMar {nsdecls("w")}>'
      f'<w:top w:w="{top}" w:type="dxa"/>'
      f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
      f'<w:left w:w="{left}" w:type="dxa"/>'
      f'<w:right w:w="{right}" w:type="dxa"/>'
      f'</w:tcMar>'
  )
  tcPr.append(tcMar)


def create_report_docx():
  doc = Document()

  # Set 1-inch margins
  for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

  # Title
  p_title = doc.add_paragraph()
  run_title = p_title.add_run(
      "PROJECT REPORT:\nPAKISTAN SIGN LANGUAGE ALPHABET RECOGNIZER"
  )
  run_title.font.name = "Calibri"
  run_title.font.size = Pt(22)
  run_title.font.bold = True
  run_title.font.color.rgb = RGBColor(24, 76, 120)
  p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
  p_title.paragraph_format.space_after = Pt(4)

  # Subtitle
  p_sub = doc.add_paragraph()
  p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
  run_sub = p_sub.add_run(
      "AL-2002 Artificial Intelligence (Laboratory) | Fall 2026\n"
      "FAST National University of Computer and Emerging Sciences, Karachi Campus\n"
      "Department of Software Engineering\n"
  )
  run_sub.font.name = "Calibri"
  run_sub.font.size = Pt(11)
  run_sub.font.italic = True
  run_sub.font.color.rgb = RGBColor(100, 100, 100)
  p_sub.paragraph_format.space_after = Pt(14)

  # Author info table
  info_table = doc.add_table(rows=3, cols=2)
  info_table.alignment = WD_TABLE_ALIGNMENT.CENTER
  info_table.autofit = False

  author_data = [
      ("Student Name", "Areesha Ahmad"),
      ("Roll Number & Section", "24K-3043 (BSE-5B)"),
      ("Project Repository", "https://github.com/arwsh21/urdu-sign-lang"),
  ]

  for row_idx, (k, v) in enumerate(author_data):
    cell_k = info_table.cell(row_idx, 0)
    cell_v = info_table.cell(row_idx, 1)
    cell_k.text = k
    cell_v.text = v
    cell_k.paragraphs[0].runs[0].font.bold = True
    cell_k.paragraphs[0].runs[0].font.size = Pt(10)
    cell_k.paragraphs[0].runs[0].font.name = "Calibri"
    cell_v.paragraphs[0].runs[0].font.size = Pt(10)
    cell_v.paragraphs[0].runs[0].font.name = "Calibri"
    cell_k.width = Inches(2.2)
    cell_v.width = Inches(4.3)
    set_cell_background(cell_k, "F0F4F8")
    set_cell_background(cell_v, "FAFAFA")
    set_cell_margins(cell_k, top=80, bottom=80, left=120, right=120)
    set_cell_margins(cell_v, top=80, bottom=80, left=120, right=120)

  doc.add_paragraph()

  # Read Markdown source
  md_path = Path("docs/project_report.md")
  with open(md_path, encoding="utf-8") as f:
    lines = f.readlines()

  i = 0
  while i < len(lines):
    raw_line = lines[i]
    line = raw_line.strip()

    # Skip top headers already rendered in title block
    if (
        line.startswith("# ")
        or line.startswith("## Pakistan Sign Language")
        or line.startswith("**Course**")
        or line.startswith("**Institution**")
        or line.startswith("**Department**")
        or line.startswith("**Student Name**")
        or line.startswith("**Roll Number**")
        or line.startswith("**Project Category**")
        or line.startswith("**Repository URL**")
        or line == "---"
    ):
      i += 1
      continue

    # Heading 2 (e.g. ## 1. Project Motivation)
    if line.startswith("## "):
      heading_text = line[3:].strip()
      h = doc.add_heading(heading_text, level=1)
      h.style.font.name = "Calibri"
      h.style.font.color.rgb = RGBColor(24, 76, 120)
      h.paragraph_format.space_before = Pt(14)
      h.paragraph_format.space_after = Pt(4)
      i += 1
      continue

    # Heading 3 (e.g. ### 3.1 PEAS Description)
    if line.startswith("### "):
      heading_text = line[4:].strip()
      h = doc.add_heading(heading_text, level=2)
      h.style.font.name = "Calibri"
      h.style.font.color.rgb = RGBColor(41, 128, 185)
      h.paragraph_format.space_before = Pt(10)
      h.paragraph_format.space_after = Pt(2)
      i += 1
      continue

    # Formula code block (indented with 4 spaces)
    if raw_line.startswith("    ") and line:
      formula_lines = [line]
      i += 1
      while (
          i < len(lines)
          and lines[i].startswith("    ")
          and lines[i].strip() != ""
      ):
        formula_lines.append(lines[i].strip())
        i += 1

      # Render Formula inside an elegant single-cell callout box
      box = doc.add_table(rows=1, cols=1)
      box.alignment = WD_TABLE_ALIGNMENT.CENTER
      box.autofit = False
      cell = box.cell(0, 0)
      cell.width = Inches(6.0)
      set_cell_background(cell, "F2F5F8")
      set_cell_margins(cell, top=100, bottom=100, left=150, right=150)

      # Left accent border
      tcPr = cell._element.get_or_add_tcPr()
      tcBorders = parse_xml(
          f'<w:tcBorders {nsdecls("w")}>'
          f'<w:top w:val="none"/>'
          f'<w:left w:val="single" w:sz="18" w:space="0" w:color="4C6399"/>'
          f'<w:bottom w:val="none"/>'
          f'<w:right w:val="none"/>'
          f'</w:tcBorders>'
      )
      tcPr.append(tcBorders)

      p = cell.paragraphs[0]
      p.paragraph_format.space_before = Pt(4)
      p.paragraph_format.space_after = Pt(4)
      p.alignment = WD_ALIGN_PARAGRAPH.CENTER

      for f_idx, f_line in enumerate(formula_lines):
        if f_idx > 0:
          run = p.add_run("\n" + f_line)
        else:
          run = p.add_run(f_line)
        run.font.name = "Cambria Math"
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = RGBColor(20, 60, 100)

      doc.add_paragraph()  # Spacing
      continue

    # Tables
    if line.startswith("|") and line.endswith("|"):
      table_lines = [line]
      i += 1
      while i < len(lines) and lines[i].strip().startswith("|"):
        table_lines.append(lines[i].strip())
        i += 1

      header_row = [
          c.strip() for c in table_lines[0].split("|")[1:-1] if c.strip()
      ]
      data_rows = []
      for t_line in table_lines[2:]:
        cols = [c.strip() for c in t_line.split("|")[1:-1]]
        if cols:
          data_rows.append(cols)

      t = doc.add_table(rows=len(data_rows) + 1, cols=len(header_row))
      t.alignment = WD_TABLE_ALIGNMENT.CENTER
      t.style = "Table Grid"

      for col_idx, col_name in enumerate(header_row):
        c = t.cell(0, col_idx)
        c.text = col_name.replace("`", "").replace("**", "")
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.bold = True
        p.runs[0].font.name = "Calibri"
        p.runs[0].font.size = Pt(10)
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(c, "1F4E79")
        set_cell_margins(c, top=80, bottom=80, left=100, right=100)

      for r_idx, r_data in enumerate(data_rows):
        bg = "F9FBFD" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(r_data):
          c = t.cell(r_idx + 1, c_idx)
          clean_val = val.replace("`", "")
          c.text = clean_val
          p = c.paragraphs[0]
          p.paragraph_format.space_before = Pt(2)
          p.paragraph_format.space_after = Pt(2)
          if clean_val.startswith("**") and clean_val.endswith("**"):
            p.runs[0].font.bold = True
            c.text = clean_val.replace("**", "")
          p.runs[0].font.name = "Calibri"
          p.runs[0].font.size = Pt(9.5)
          set_cell_background(c, bg)
          set_cell_margins(c, top=60, bottom=60, left=100, right=100)

      doc.add_paragraph()  # Spacing
      continue

    # Bullet lists
    if line.startswith("* ") or line.startswith("- "):
      bullet_text = line[2:].strip()
      p = doc.add_paragraph(style="List Bullet")
      p.paragraph_format.space_before = Pt(1)
      p.paragraph_format.space_after = Pt(2)
      _add_styled_runs(p, bullet_text)
      i += 1
      continue

    # Numbered lists
    if len(line) > 2 and line[0].isdigit() and line[1:3] in (". ", ") "):
      num_text = line[3:].strip()
      p = doc.add_paragraph(style="List Number")
      p.paragraph_format.space_before = Pt(1)
      p.paragraph_format.space_after = Pt(2)
      _add_styled_runs(p, num_text)
      i += 1
      continue

    # Normal body paragraphs
    if line:
      p = doc.add_paragraph()
      p.paragraph_format.space_before = Pt(2)
      p.paragraph_format.space_after = Pt(4)
      p.paragraph_format.line_spacing = 1.15
      _add_styled_runs(p, line)

      # Automatically embed figures after Section 7.4
      if "Three figures were automatically generated" in line:
        fig1_path = Path("outputs/figures/model_accuracy_comparison.png")
        fig2_path = Path("outputs/figures/confusion_matrix.png")
        fig3_path = Path("outputs/figures/learning_curve.png")

        if fig1_path.exists():
          p_f1 = doc.add_paragraph()
          p_f1.alignment = WD_ALIGN_PARAGRAPH.CENTER
          p_f1.paragraph_format.space_before = Pt(8)
          p_f1.paragraph_format.space_after = Pt(4)
          doc.add_picture(str(fig1_path), width=Inches(5.6))
          p_cap1 = doc.add_paragraph()
          p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
          r_cap1 = p_cap1.add_run("Figure 1: Cross-Validation Accuracy Comparison Across All 4 Models")
          r_cap1.font.italic = True
          r_cap1.font.size = Pt(9.5)
          r_cap1.font.color.rgb = RGBColor(100, 100, 100)

        if fig2_path.exists():
          p_f2 = doc.add_paragraph()
          p_f2.alignment = WD_ALIGN_PARAGRAPH.CENTER
          p_f2.paragraph_format.space_before = Pt(8)
          p_f2.paragraph_format.space_after = Pt(4)
          doc.add_picture(str(fig2_path), width=Inches(5.8))
          p_cap2 = doc.add_paragraph()
          p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
          r_cap2 = p_cap2.add_run("Figure 2: Multi-Class Confusion Matrix for SVM (31 Alphabet Signs)")
          r_cap2.font.italic = True
          r_cap2.font.size = Pt(9.5)
          r_cap2.font.color.rgb = RGBColor(100, 100, 100)

        if fig3_path.exists():
          p_f3 = doc.add_paragraph()
          p_f3.alignment = WD_ALIGN_PARAGRAPH.CENTER
          p_f3.paragraph_format.space_before = Pt(8)
          p_f3.paragraph_format.space_after = Pt(4)
          doc.add_picture(str(fig3_path), width=Inches(5.6))
          p_cap3 = doc.add_paragraph()
          p_cap3.alignment = WD_ALIGN_PARAGRAPH.CENTER
          r_cap3 = p_cap3.add_run("Figure 3: Training vs. Validation Learning Curve for SVM")
          r_cap3.font.italic = True
          r_cap3.font.size = Pt(9.5)
          r_cap3.font.color.rgb = RGBColor(100, 100, 100)

    i += 1

  output_docx = Path("docs/project_report.docx")
  doc.save(output_docx)
  print(f"[SUCCESS] Report saved to: {output_docx.resolve()}")


def _add_styled_runs(paragraph, text):
  """Parses bold, code, and regular markdown spans into styled docx runs."""
  parts = text.split("**")
  for idx, part in enumerate(parts):
    if not part:
      continue
    run = paragraph.add_run(part)
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    if idx % 2 == 1:
      run.font.bold = True
      run.font.color.rgb = RGBColor(20, 20, 20)


if __name__ == "__main__":
  create_report_docx()

