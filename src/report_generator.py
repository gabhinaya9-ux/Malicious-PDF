"""
report_generator.py
-------------------
Assembles a complete PDF security-analysis report from all pipeline
outputs and saves it as a formatted PDF document using fpdf2.

Report sections
---------------
  1. Header – branding, file name, timestamp
  2. File Overview – path, size, SHA-256, PDF version
  3. Validation – pass / fail with details
  4. Risk Assessment – score gauge, classification, confidence
  5. ML Prediction – probability, model name
  6. Rule Analysis – table of triggered rules
  7. Feature Highlights – top static-analysis findings
  8. SHAP Explanation – embedded chart image
  9. URL List – first 20 extracted URLs
  10. Recommendations – actionable guidance
  11. Footer – disclaimer
"""

from __future__ import annotations

import datetime
import io
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# fpdf2 import (graceful fallback to text report)
# ---------------------------------------------------------------------------
try:
    from fpdf import FPDF, XPos, YPos
    _HAVE_FPDF = True
except ImportError:
    _HAVE_FPDF = False

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_C_MALICIOUS  = (231, 76,  60)    # red
_C_SUSPICIOUS = (243, 156,  18)   # amber
_C_BENIGN     = (39,  174,  96)   # green
_C_DARK       = (44,   62,  80)   # dark blue-grey
_C_LIGHT      = (236, 240, 241)   # very light grey
_C_WHITE      = (255, 255, 255)
_C_ACCENT     = (52, 152, 219)    # blue


def _class_colour(classification: str):
    c = classification.lower()
    if c == "malicious":  return _C_MALICIOUS
    if c == "suspicious": return _C_SUSPICIOUS
    return _C_BENIGN


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_report(
    validation_result,          # ValidationResult from file_validator
    analysis:    Dict,          # raw analysis dict from pdf_analyzer
    features:    Dict,          # feature dict from pdf_features
    risk_result,                # RiskResult from risk_scorer
    explanation: Optional[Dict] = None,   # from explainer.explain_prediction()
    model_name:  str = "Best Model",
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Generate the full analysis report.

    Returns the path to the saved report file.
    If fpdf2 is not installed, falls back to a plain-text .txt report.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname     = Path(risk_result.file_path).name

    if output_path is None:
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        safe_name  = fname.replace(" ", "_").replace(".pdf", "")
        output_path = reports_dir / f"report_{safe_name}_{datetime.datetime.now():%Y%m%d_%H%M%S}.pdf"

    output_path = Path(output_path)

    if _HAVE_FPDF:
        _build_pdf_report(
            validation_result, analysis, features, risk_result,
            explanation, model_name, timestamp, fname, output_path
        )
    else:
        output_path = output_path.with_suffix(".txt")
        _build_text_report(
            validation_result, analysis, features, risk_result,
            explanation, model_name, timestamp, fname, output_path
        )

    return str(output_path)


# ===========================================================================
# PDF report (fpdf2)
# ===========================================================================
def _build_pdf_report(val, analysis, features, risk, explanation,
                      model_name, timestamp, fname, output_path):

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── 1. Header ─────────────────────────────────────────────────────
    _header_block(pdf, fname, timestamp)

    # ── 2. File Overview ──────────────────────────────────────────────
    _section_title(pdf, "1. File Overview")
    rows = [
        ("File Name",    fname),
        ("File Size",    f"{val.file_size_bytes:,} bytes"),
        ("SHA-256",      val.sha256[:32] + "..."),
        ("PDF Version",  val.pdf_version),
        ("Page Count",   str(analysis.get("page_count", "N/A"))),
        ("Analysis Time", timestamp),
    ]
    _key_value_table(pdf, rows)

    # ── 3. Validation ─────────────────────────────────────────────────
    _section_title(pdf, "2. File Validation")
    status_colour = _C_BENIGN if val.is_valid else _C_MALICIOUS
    _coloured_badge(pdf, "VALID" if val.is_valid else "INVALID", status_colour)
    if val.errors:
        for e in val.errors:
            _bullet(pdf, f"ERROR: {e}")
    if val.warnings:
        for w in val.warnings:
            _bullet(pdf, f"WARNING: {w}", italic=True)

    # ── 4. Risk Assessment ────────────────────────────────────────────
    _section_title(pdf, "3. Risk Assessment")
    _risk_score_block(pdf, risk)

    # ── 5. ML Prediction ─────────────────────────────────────────────
    _section_title(pdf, "4. Machine Learning Prediction")
    ml_rows = [
        ("Model",             model_name),
        ("Malicious Prob.",   f"{risk.ml_probability:.1%}"),
        ("ML Score (/ 70)",   f"{risk.ml_score:.1f}"),
        ("Rule Score (/ 30)", f"{risk.rule_score:.1f}"),
        ("Final Risk Score",  f"{risk.risk_score:.0f} / 100"),
    ]
    _key_value_table(pdf, ml_rows)

    # ── 6. Rule Analysis ──────────────────────────────────────────────
    _section_title(pdf, "5. Rule-Based Analysis")
    if risk.rule_hits:
        _rule_table(pdf, risk.rule_hits)
    else:
        _body_text(pdf, "No rule-based indicators triggered.")

    # ── 7. Feature Highlights ─────────────────────────────────────────
    _section_title(pdf, "6. Static Analysis Feature Highlights")
    highlight_keys = [
        "has_javascript", "javascript_count",
        "has_openaction", "has_aa", "has_launch", "action_count",
        "embedded_file_count", "has_richmedia",
        "url_count", "has_suspicious_url",
        "annotation_count", "uri_annotation_count",
        "has_acroform", "has_xfa",
        "is_encrypted", "has_objstm", "has_xrefstm",
        "max_stream_entropy", "avg_stream_entropy", "high_entropy_stream_count",
    ]
    feat_rows = []
    for k in highlight_keys:
        v = features.get(k, analysis.get(k, "N/A"))
        feat_rows.append((_pretty_key(k), str(v)))
    _key_value_table(pdf, feat_rows)

    # ── 8. SHAP Explanation ───────────────────────────────────────────
    if explanation and explanation.get("figure"):
        _section_title(pdf, "7. SHAP Feature Contribution")
        try:
            from explainer import fig_to_bytes
            img_bytes = fig_to_bytes(explanation["figure"])
            img_path  = _bytes_to_tmp_png(img_bytes)
            pdf.image(str(img_path), w=180)
            img_path.unlink(missing_ok=True)
        except Exception:
            _body_text(pdf, "[SHAP chart could not be embedded]")

        # Top contributors table
        if explanation.get("top_n_pairs"):
            pdf.ln(4)
            _section_title(pdf, "SHAP: Top Feature Contributions")
            shap_rows = [
                (_pretty_key(n), f"{v:+.4f}", f"{fv:.3g}")
                for n, v, fv in explanation["top_n_pairs"][:15]
            ]
            _three_col_table(pdf, ["Feature", "SHAP Value", "Actual Value"], shap_rows)

    # ── 9. URL List ───────────────────────────────────────────────────
    urls = analysis.get("urls", [])
    if urls:
        _section_title(pdf, "8. Extracted URLs (first 20)")
        for url in urls[:20]:
            _bullet(pdf, url[:100] + ("…" if len(url) > 100 else ""))

    # ── 10. Recommendations ───────────────────────────────────────────
    _section_title(pdf, "9. Recommendations")
    _recommendations(pdf, risk)

    # ── 11. Footer ────────────────────────────────────────────────────
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*_C_DARK)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        pdf.epw, 5,
        "DISCLAIMER: This report is generated by an automated static-analysis "
        "system. It is intended as a screening tool only and does not replace "
        "expert forensic analysis. False positives and false negatives are "
        "possible. Do not rely solely on this report for security decisions.",
        align="C",
    )

    pdf.output(str(output_path))


# ===========================================================================
# Text fallback report
# ===========================================================================
def _build_text_report(val, analysis, features, risk, explanation,
                       model_name, timestamp, fname, output_path):
    lines = [
        "=" * 70,
        "  MALICIOUS PDF DETECTION REPORT",
        f"  Generated: {timestamp}",
        "=" * 70,
        "",
        f"File       : {fname}",
        f"SHA-256    : {val.sha256}",
        f"Size       : {val.file_size_bytes:,} bytes",
        f"PDF Ver    : {val.pdf_version}",
        "",
        "── RISK RESULT ──────────────────────────────────────────────────",
        f"Risk Score     : {risk.risk_score:.0f} / 100",
        f"Classification : {risk.classification}",
        f"Confidence     : {risk.confidence}",
        f"ML Probability : {risk.ml_probability:.1%}",
        "",
    ]
    if risk.rule_hits:
        lines.append("── TRIGGERED RULES ──────────────────────────────────────────────")
        for rh in risk.rule_hits:
            lines.append(f"  [{rh.weight:.1f}] {rh.description}")
        lines.append("")

    lines += [
        "── STATIC ANALYSIS ──────────────────────────────────────────────",
        f"  JavaScript     : {features.get('has_javascript', 0)}",
        f"  OpenAction     : {features.get('has_openaction', 0)}",
        f"  Embedded Files : {features.get('embedded_file_count', 0)}",
        f"  URLs           : {features.get('url_count', 0)}",
        f"  Max Entropy    : {features.get('max_stream_entropy', 0.0):.4f}",
        "",
        "=" * 70,
        "DISCLAIMER: Automated screening only. Expert review recommended.",
        "=" * 70,
    ]

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ===========================================================================
# fpdf2 building blocks
# ===========================================================================
def _header_block(pdf: "FPDF", fname: str, timestamp: str):
    pdf.set_fill_color(*_C_DARK)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*_C_WHITE)
    pdf.set_xy(10, 6)
    pdf.cell(0, 10, "Malicious PDF Detection Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 18)
    pdf.cell(0, 6, _safe(f"File: {fname}    |    Generated: {timestamp}"))
    pdf.ln(12)
    pdf.set_text_color(*_C_DARK)


def _section_title(pdf: "FPDF", title: str):
    pdf.ln(4)
    pdf.set_fill_color(*_C_ACCENT)
    pdf.set_text_color(*_C_WHITE)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, _safe(f"  {title}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.set_text_color(*_C_DARK)
    pdf.ln(2)


def _body_text(pdf: "FPDF", text: str, italic: bool = False):
    pdf.set_font("Helvetica", "I" if italic else "", 10)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, 6, _safe(text))


def _bullet(pdf: "FPDF", text: str, italic: bool = False):
    pdf.set_font("Helvetica", "I" if italic else "", 9)
    safe_text = _safe(text[:120])
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, 6, f"  - {safe_text}")


def _key_value_table(pdf: "FPDF", rows: List[tuple]):
    pdf.set_font("Helvetica", "", 9)
    for k, v in rows:
        pdf.set_fill_color(*_C_LIGHT)
        pdf.cell(70, 7, _safe(f"  {k}"), border=0, fill=True)
        pdf.set_fill_color(*_C_WHITE)
        pdf.cell(0, 7, _safe(f"  {v}"), border=0, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)


def _three_col_table(pdf: "FPDF", headers: List[str], rows: List[tuple]):
    col_w = [90, 40, 50]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(*_C_DARK)
    pdf.set_text_color(*_C_WHITE)
    for h, w in zip(headers, col_w):
        pdf.cell(w, 7, _safe(f" {h}"), fill=True)
    pdf.ln()
    pdf.set_text_color(*_C_DARK)
    for i, row in enumerate(rows):
        fill = i % 2 == 0
        pdf.set_fill_color(*(_C_LIGHT if fill else _C_WHITE))
        pdf.set_font("Helvetica", "", 8)
        for val, w in zip(row, col_w):
            pdf.cell(w, 6, _safe(f" {val}"), fill=fill)
        pdf.ln()
    pdf.ln(2)


def _rule_table(pdf: "FPDF", rule_hits):
    headers = ["Rule Description", "Weight", "Value"]
    rows    = [(rh.description, f"{rh.weight:.1f}", str(rh.value)) for rh in rule_hits]
    _three_col_table(pdf, headers, rows)


def _coloured_badge(pdf: "FPDF", text: str, colour: tuple):
    pdf.set_fill_color(*colour)
    pdf.set_text_color(*_C_WHITE)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(50, 10, f"  {text}  ", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*_C_DARK)
    pdf.ln(2)


def _risk_score_block(pdf: "FPDF", risk):
    colour = _class_colour(risk.classification)
    pdf.set_font("Helvetica", "B", 36)
    pdf.set_text_color(*colour)
    pdf.set_x(pdf.l_margin)
    pdf.cell(pdf.epw, 14, f"{risk.risk_score:.0f} / 100", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_x(pdf.l_margin)
    pdf.cell(pdf.epw, 8, risk.classification.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_C_DARK)
    pdf.set_x(pdf.l_margin)
    pdf.cell(pdf.epw, 6, f"Confidence: {risk.confidence}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(4)


def _recommendations(pdf: "FPDF", risk):
    cls = risk.classification.lower()
    if cls == "malicious":
        recs = [
            "Do NOT open this PDF on a production or personal machine.",
            "Quarantine the file immediately and report to your security team.",
            "Submit the file to a sandbox (e.g., Any.run, Cuckoo) for dynamic analysis.",
            "Block the SHA-256 hash in your endpoint protection platform.",
            "If already opened, assume compromise and begin incident response.",
        ]
    elif cls == "suspicious":
        recs = [
            "Treat with caution — do not open if the source is untrusted.",
            "Scan with a second AV engine or upload to VirusTotal.",
            "Consider opening in an isolated sandbox environment.",
            "Inspect embedded URLs and JavaScript if safe to do so.",
        ]
    else:
        recs = [
            "File appears benign based on static analysis.",
            "Ensure the file came from a trusted source before opening.",
            "Static analysis cannot detect all threats — dynamic analysis provides additional confidence.",
        ]

    for rec in recs:
        _bullet(pdf, rec)


def _safe(text: str, max_len: int = 120) -> str:
    """Truncate and strip non-latin-1 characters for core fpdf fonts."""
    text = text[:max_len]
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _pretty_key(k: str) -> str:
    return k.replace("_", " ").title()


def _bytes_to_tmp_png(data: bytes) -> Path:
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".png"))
    tmp.write_bytes(data)
    return tmp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    print("report_generator.py: import and call generate_report() from the pipeline.")
    print("Use app.py (Streamlit) for the interactive interface.")
