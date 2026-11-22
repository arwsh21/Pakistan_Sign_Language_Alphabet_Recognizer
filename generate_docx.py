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
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'''<w:shd {nsdecls('w')} w:fill="{fill_hex}"/>''')
    tcPr.append(shd)

def set_cell_margins(cell, top=60, bottom=60, left=80, right=80):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'''<w:tcMar {nsdecls('w')}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>''')
    tcPr.append(tcMar)

def create_report_docx():
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    p_title = doc.add_paragraph()
    run_title = p_title.add_run('PROJECT REPORT:\nPAKISTAN SIGN LANGUAGE ALPHABET RECOGNIZER')
    run_title.font.name = 'Calibri'
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(24, 76, 120)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run('AL-2002 Artificial Intelligence (Laboratory) | Fall 2026\nFAST National University of Computer and Emerging Sciences, Karachi Campus\nDepartment of Software Engineering')
    run_sub.font.name = 'Calibri'
    run_sub.font.size = Pt(10)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(100, 100, 100)
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(6)

    info_table = doc.add_table(rows=3, cols=2)
    info_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    info_table.autofit = False
    author_data = [
        ('Student Name', 'Areesha Ahmad'),
        ('Roll Number & Section', '24K-3043 (BSE-5B)'),
        ('Project Repository', 'https://github.com/arwsh21/Pakistan_Sign_Language_Alphabet_Recognizer')
    ]
    for row_idx, (k, v) in enumerate(author_data):
        cell_k = info_table.cell(row_idx, 0)
        cell_v = info_table.cell(row_idx, 1)
        cell_k.text = k
        cell_v.text = v
        cell_k.paragraphs[0].runs[0].font.bold = True
        cell_k.paragraphs[0].runs[0].font.size = Pt(9.5)
        cell_k.paragraphs[0].runs[0].font.name = 'Calibri'
        cell_k.paragraphs[0].paragraph_format.space_before = Pt(1)
        cell_k.paragraphs[0].paragraph_format.space_after = Pt(1)
        cell_v.paragraphs[0].runs[0].font.size = Pt(9.5)
        cell_v.paragraphs[0].runs[0].font.name = 'Calibri'
        cell_v.paragraphs[0].paragraph_format.space_before = Pt(1)
        cell_v.paragraphs[0].paragraph_format.space_after = Pt(1)
        cell_k.width = Inches(2.0)
        cell_v.width = Inches(5.0)
        set_cell_background(cell_k, 'F0F4F8')
        set_cell_background(cell_v, 'FAFAFA')
        set_cell_margins(cell_k, top=40, bottom=40, left=80, right=80)
        set_cell_margins(cell_v, top=40, bottom=40, left=80, right=80)

    md_path = Path('docs/project_report.md')
    with open(md_path, encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.strip()

        if line.startswith('# ') or line.startswith('## Pakistan Sign Language') or line.startswith('**Course**') or line.startswith('**Institution**') or line.startswith('**Department**') or line.startswith('**Student Name**') or line.startswith('**Roll Number**') or line.startswith('**Project Category**') or line.startswith('**Repository URL**') or (line == '---'):
            i += 1
            continue

        if line.startswith('## '):
            heading_text = line[3:].strip()
            h = doc.add_heading(heading_text, level=1)
            h.style.font.name = 'Calibri'
            h.style.font.color.rgb = RGBColor(24, 76, 120)
            h.paragraph_format.space_before = Pt(7)
            h.paragraph_format.space_after = Pt(2)
            i += 1
            continue

        if line.startswith('### '):
            heading_text = line[4:].strip()
            h = doc.add_heading(heading_text, level=2)
            h.style.font.name = 'Calibri'
            h.style.font.color.rgb = RGBColor(41, 128, 185)
            h.paragraph_format.space_before = Pt(5)
            h.paragraph_format.space_after = Pt(1.5)
            i += 1
            continue

        if raw_line.startswith('    ') and line:
            formula_lines = [line]
            i += 1
            while i < len(lines) and lines[i].startswith('    ') and (lines[i].strip() != ''):
                formula_lines.append(lines[i].strip())
                i += 1
            box = doc.add_table(rows=1, cols=1)
            box.alignment = WD_TABLE_ALIGNMENT.CENTER
            box.autofit = False
            cell = box.cell(0, 0)
            cell.width = Inches(7.0)
            set_cell_background(cell, 'F2F5F8')
            set_cell_margins(cell, top=50, bottom=50, left=100, right=100)
            tcPr = cell._element.get_or_add_tcPr()
            tcBorders = parse_xml(f"""<w:tcBorders {nsdecls('w')}><w:top w:val="none"/><w:left w:val="single" w:sz="18" w:space="0" w:color="4C6399"/><w:bottom w:val="none"/><w:right w:val="none"/></w:tcBorders>""")
            tcPr.append(tcBorders)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for f_idx, f_line in enumerate(formula_lines):
                if f_idx > 0:
                    run = p.add_run('\n' + f_line)
                else:
                    run = p.add_run(f_line)
                run.font.name = 'Cambria Math'
                run.font.size = Pt(10)
                run.font.bold = True
                run.font.color.rgb = RGBColor(20, 60, 100)
            continue

        if line.startswith('|') and line.endswith('|'):
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            header_row = [c.strip() for c in table_lines[0].split('|')[1:-1] if c.strip()]
            data_rows = []
            for t_line in table_lines[2:]:
                cols = [c.strip() for c in t_line.split('|')[1:-1]]
                if cols:
                    data_rows.append(cols)
            num_cols = len(header_row)
            t = doc.add_table(rows=len(data_rows) + 1, cols=num_cols)
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            t.style = 'Table Grid'
            t.autofit = False

            if num_cols == 6:
                col_widths = [Inches(0.85), Inches(0.45), Inches(2.2), Inches(0.85), Inches(0.45), Inches(2.2)]
            elif num_cols == 5:
                col_widths = [Inches(1.8), Inches(1.4), Inches(1.4), Inches(1.1), Inches(1.3)]
            else:
                col_widths = [Inches(7.0 / num_cols)] * num_cols

            for col_idx, col_name in enumerate(header_row):
                c = t.cell(0, col_idx)
                c.width = col_widths[col_idx]
                c.text = col_name.replace('`', '').replace('**', '')
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(2)
                p.runs[0].font.bold = True
                p.runs[0].font.name = 'Calibri'
                p.runs[0].font.size = Pt(9)
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
                set_cell_background(c, '1F4E79')
                set_cell_margins(c, top=40, bottom=40, left=60, right=60)

            for r_idx, r_data in enumerate(data_rows):
                bg = 'F9FBFD' if r_idx % 2 == 1 else 'FFFFFF'
                for c_idx, val in enumerate(r_data):
                    c = t.cell(r_idx + 1, c_idx)
                    c.width = col_widths[c_idx]
                    clean_val = val.replace('`', '')
                    c.text = clean_val
                    p = c.paragraphs[0]
                    p.paragraph_format.space_before = Pt(1)
                    p.paragraph_format.space_after = Pt(1)
                    if clean_val.startswith('**') and clean_val.endswith('**'):
                        p.runs[0].font.bold = True
                        c.text = clean_val.replace('**', '')
                    p.runs[0].font.name = 'Calibri'
                    p.runs[0].font.size = Pt(8.5 if num_cols == 6 else 9)
                    if c_idx in (0, 1, 3, 4) and num_cols == 6:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    set_cell_background(c, bg)
                    set_cell_margins(c, top=30, bottom=30, left=50, right=50)
            continue

        if line.startswith('* ') or line.startswith('- '):
            bullet_text = line[2:].strip()
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before = Pt(0.5)
            p.paragraph_format.space_after = Pt(1.5)
            p.paragraph_format.line_spacing = 1.12
            _add_styled_runs(p, bullet_text)
            i += 1
            continue

        if len(line) > 2 and line[0].isdigit() and (line[1:3] in ('. ', ') ')):
            num_text = line[3:].strip()
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_before = Pt(0.5)
            p.paragraph_format.space_after = Pt(1.5)
            p.paragraph_format.line_spacing = 1.12
            _add_styled_runs(p, num_text)
            i += 1
            continue

        if line:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2.5)
            p.paragraph_format.line_spacing = 1.12
            _add_styled_runs(p, line)

            if 'Three figures were automatically generated' in line:
                fig1_path = Path('outputs/figures/model_accuracy_comparison.png')
                fig2_path = Path('outputs/figures/confusion_matrix.png')
                fig3_path = Path('outputs/figures/learning_curve.png')

                if fig1_path.exists() and fig3_path.exists():
                    fig_table = doc.add_table(rows=1, cols=2)
                    fig_table.alignment = WD_TABLE_ALIGNMENT.CENTER
                    fig_table.autofit = False
                    
                    cell_f1 = fig_table.cell(0, 0)
                    cell_f3 = fig_table.cell(0, 1)
                    cell_f1.width = Inches(3.45)
                    cell_f3.width = Inches(3.45)
                    set_cell_margins(cell_f1, top=20, bottom=20, left=30, right=30)
                    set_cell_margins(cell_f3, top=20, bottom=20, left=30, right=30)

                    p1 = cell_f1.paragraphs[0]
                    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p1.paragraph_format.space_before = Pt(2)
                    p1.paragraph_format.space_after = Pt(1)
                    r1 = p1.add_run()
                    r1.add_picture(str(fig1_path), width=Inches(3.3))
                    p1_cap = cell_f1.add_paragraph()
                    p1_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p1_cap.paragraph_format.space_before = Pt(1)
                    p1_cap.paragraph_format.space_after = Pt(2)
                    r1_cap = p1_cap.add_run('Figure 1: Cross-Validation Accuracy Comparison')
                    r1_cap.font.italic = True
                    r1_cap.font.size = Pt(8.5)
                    r1_cap.font.color.rgb = RGBColor(100, 100, 100)

                    p3 = cell_f3.paragraphs[0]
                    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p3.paragraph_format.space_before = Pt(2)
                    p3.paragraph_format.space_after = Pt(1)
                    r3 = p3.add_run()
                    r3.add_picture(str(fig3_path), width=Inches(3.3))
                    p3_cap = cell_f3.add_paragraph()
                    p3_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p3_cap.paragraph_format.space_before = Pt(1)
                    p3_cap.paragraph_format.space_after = Pt(2)
                    r3_cap = p3_cap.add_run('Figure 3: Training vs. Validation Learning Curve')
                    r3_cap.font.italic = True
                    r3_cap.font.size = Pt(8.5)
                    r3_cap.font.color.rgb = RGBColor(100, 100, 100)

                if fig2_path.exists():
                    p_f2 = doc.add_paragraph()
                    p_f2.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_f2.paragraph_format.space_before = Pt(4)
                    p_f2.paragraph_format.space_after = Pt(1)
                    r2 = p_f2.add_run()
                    r2.add_picture(str(fig2_path), width=Inches(3.5))

                    p_cap2 = doc.add_paragraph()
                    p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_cap2.paragraph_format.space_before = Pt(1)
                    p_cap2.paragraph_format.space_after = Pt(3)
                    r_cap2 = p_cap2.add_run('Figure 2: Multi-Class Confusion Matrix for SVM')
                    r_cap2.font.italic = True
                    r_cap2.font.size = Pt(8.5)
                    r_cap2.font.color.rgb = RGBColor(100, 100, 100)
        i += 1

    output_docx = Path('docs/project_report.docx')
    doc.save(output_docx)
    print(f'[SUCCESS] Report saved to: {output_docx.resolve()}')

def _add_styled_runs(paragraph, text):
    parts = text.split('**')
    for idx, part in enumerate(parts):
        if not part:
            continue
        run = paragraph.add_run(part)
        run.font.name = 'Calibri'
        run.font.size = Pt(10)
        if idx % 2 == 1:
            run.font.bold = True
            run.font.color.rgb = RGBColor(20, 20, 20)

if __name__ == '__main__':
    create_report_docx()
