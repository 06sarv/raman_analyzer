"""
utils/report.py
---------------
PDF report generation using ReportLab.

Produces a professional multi-section report containing:
  - Analysis summary header
  - Spectrum plot image
  - Peak table with functional group assignments
  - Compound match table with confidence scores
  - AI-generated compound summary
  - PubChem compound details
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, KeepTogether,
)

_STYLES = getSampleStyleSheet()

_HEADING1 = ParagraphStyle(
    "Heading1Custom",
    parent=_STYLES["Heading1"],
    fontSize=18,
    spaceAfter=6,
    textColor=colors.HexColor("#1a3c6e"),
)
_HEADING2 = ParagraphStyle(
    "Heading2Custom",
    parent=_STYLES["Heading2"],
    fontSize=13,
    spaceAfter=4,
    textColor=colors.HexColor("#1a3c6e"),
)
_BODY = ParagraphStyle(
    "BodyCustom",
    parent=_STYLES["Normal"],
    fontSize=10,
    leading=14,
)
_CAPTION = ParagraphStyle(
    "Caption",
    parent=_STYLES["Normal"],
    fontSize=9,
    textColor=colors.grey,
    italic=True,
)


def _table_style(header_color=colors.HexColor("#1a3c6e")) -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4fa")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ])


def _fig_to_image(fig: plt.Figure, width=5.5 * inch, height=2.9 * inch) -> RLImage:
    """Convert a matplotlib figure to a ReportLab Image via in-memory PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    img = RLImage(buf, width=width, height=height)
    img.hAlign = "CENTER"
    return img


def _df_to_table(df: pd.DataFrame, col_widths=None) -> Table:
    headers = [[Paragraph(f"<b>{c}</b>", _CAPTION) for c in df.columns]]
    rows = [
        [Paragraph(str(v), _CAPTION) for v in row]
        for row in df.values.tolist()
    ]
    data = headers + rows
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(_table_style())
    return tbl


def generate_report(
    output_path: str,
    spectrum_label: str,
    fig: plt.Figure,
    peak_df: pd.DataFrame,
    match_df: pd.DataFrame,
    ai_summary: str,
    pubchem_info: Optional[Dict],
    metadata: Dict,
    ai_predictions: Optional[List[Dict]] = None,
) -> str:
    """
    Generate a PDF report and write it to output_path.
    Returns output_path on success.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    elements = []

    # ── Title ──────────────────────────────────────────────────
    elements.append(Paragraph("🔬 Raman Spectrum Analysis Report", _HEADING1))
    elements.append(Paragraph(f"<b>Sample:</b> {spectrum_label}", _BODY))
    meta_line = " | ".join(
        f"<b>{k.replace('_', ' ').title()}:</b> {v}"
        for k, v in metadata.items()
        if v
    )
    elements.append(Paragraph(meta_line, _BODY))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a3c6e")))
    elements.append(Spacer(1, 10))

    # ── Spectrum plot ───────────────────────────────────────────
    elements.append(Paragraph("Spectrum Visualisation", _HEADING2))
    elements.append(_fig_to_image(fig))
    elements.append(Spacer(1, 8))

    # ── Peak table ──────────────────────────────────────────────
    if not peak_df.empty:
        elements.append(Paragraph("Detected Peaks & Functional Group Assignments", _HEADING2))
        elements.append(_df_to_table(peak_df))
        elements.append(Spacer(1, 8))

    # ── Match table ─────────────────────────────────────────────
    if not match_df.empty:
        elements.append(Paragraph("Compound Match Results", _HEADING2))
        elements.append(_df_to_table(match_df))
        elements.append(Spacer(1, 8))

    # ── AI predictions ──────────────────────────────────────────
    if ai_predictions:
        elements.append(Paragraph("AI-Predicted Compounds (Gemini)", _HEADING2))
        pred_data = [["Compound", "Confidence", "Reasoning"]] + [
            [p.get("compound", ""), p.get("confidence", ""), p.get("reasoning", "")]
            for p in ai_predictions
        ]
        pred_tbl = Table(
            [[Paragraph(str(c), _CAPTION) for c in row] for row in pred_data],
            colWidths=[1.5 * inch, 0.9 * inch, 4.1 * inch],
            repeatRows=1,
        )
        pred_tbl.setStyle(_table_style())
        elements.append(pred_tbl)
        elements.append(Spacer(1, 8))

    # ── AI summary ──────────────────────────────────────────────
    if ai_summary:
        elements.append(Paragraph("AI-Generated Compound Summary (Top Match)", _HEADING2))
        for line in ai_summary.splitlines():
            line = line.strip()
            if line:
                elements.append(Paragraph(line, _BODY))
        elements.append(Spacer(1, 8))

    # ── PubChem ─────────────────────────────────────────────────
    if pubchem_info:
        elements.append(Paragraph("PubChem Details", _HEADING2))
        pc_rows = [
            ["CID", str(pubchem_info.get("cid", ""))],
            ["IUPAC Name", pubchem_info.get("iupac_name", "")],
            ["Molecular Formula", pubchem_info.get("molecular_formula", "")],
            ["Molecular Weight", pubchem_info.get("molecular_weight", "")],
            ["SMILES", pubchem_info.get("canonical_smiles", "")],
        ]
        pc_tbl = Table(
            [[Paragraph(f"<b>{r[0]}</b>", _CAPTION), Paragraph(r[1], _CAPTION)] for r in pc_rows],
            colWidths=[1.8 * inch, 4.7 * inch],
        )
        pc_tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f0f4fa")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(pc_tbl)
        if pubchem_info.get("description"):
            elements.append(Spacer(1, 4))
            elements.append(Paragraph(pubchem_info["description"], _BODY))

    doc.build(elements)
    return output_path
