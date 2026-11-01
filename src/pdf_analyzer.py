"""
pdf_analyzer.py
---------------
Deep static analysis of a PDF file.

Extracts raw structural information WITHOUT executing anything inside the PDF:
  - Object counts and types
  - JavaScript presence and snippets
  - OpenAction / AA (additional-action) indicators
  - Embedded files / attachments
  - URIs / URLs from annotations and object streams
  - Annotation details
  - AcroForm / XFA form indicators
  - Stream entropy (flags obfuscated/encoded payloads)

All results are returned as a plain dict so they are easy to serialise
and pass to the feature-extraction layer.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional imports — degrade gracefully if a library is missing
# ---------------------------------------------------------------------------
try:
    import pymupdf as fitz  # PyMuPDF >= 1.24 canonical import
    _HAVE_FITZ = True
except ImportError:
    try:
        import fitz  # legacy name still works on older versions
        _HAVE_FITZ = True
    except ImportError:
        _HAVE_FITZ = False

try:
    from pdfminer.high_level import extract_pages
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdftypes import PDFStream, resolve1
    _HAVE_PDFMINER = True
except ImportError:
    _HAVE_PDFMINER = False


# ---------------------------------------------------------------------------
# Regex patterns for raw-byte scanning
# ---------------------------------------------------------------------------
_RE_JS_KEYS = re.compile(
    rb"/JS\b|/JavaScript\b|/S\s*/JavaScript|app\.alert|app\.launchURL"
    rb"|eval\s*\(|unescape\s*\(|String\.fromCharCode",
    re.IGNORECASE,
)
_RE_OPENACTION = re.compile(rb"/OpenAction\b", re.IGNORECASE)
_RE_AA = re.compile(rb"/AA\b", re.IGNORECASE)
_RE_LAUNCH = re.compile(rb"/Launch\b", re.IGNORECASE)
_RE_EMBEDDED = re.compile(rb"/EmbeddedFile\b|/Filespec\b", re.IGNORECASE)
_RE_URI = re.compile(
    rb"(https?://[^\s\x00-\x1f\"'>)\]]{4,256})",
    re.IGNORECASE,
)
_RE_ACROFORM = re.compile(rb"/AcroForm\b", re.IGNORECASE)
_RE_XFA = re.compile(rb"/XFA\b", re.IGNORECASE)
_RE_ENCRYPT = re.compile(rb"/Encrypt\b", re.IGNORECASE)
_RE_OBJSTM = re.compile(rb"/ObjStm\b", re.IGNORECASE)
_RE_XREF_STM = re.compile(rb"/XRefStm\b", re.IGNORECASE)
_RE_RICHMEDIA = re.compile(rb"/RichMedia\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_pdf(file_path: str | Path) -> Dict[str, Any]:
    """
    Perform full static analysis of *file_path*.

    Returns a dict with the keys documented in :data:`FEATURE_KEYS`.
    Missing / uncomputable values default to 0 or empty list.
    """
    path = Path(file_path)
    result: Dict[str, Any] = _empty_result(str(path))

    # ------------------------------------------------------------------
    # Raw-byte pass (fast, works even on corrupt PDFs)
    # ------------------------------------------------------------------
    try:
        raw = path.read_bytes()
        result.update(_raw_scan(raw))
    except OSError as exc:
        result["parse_errors"].append(f"Cannot read file: {exc}")
        return result

    # ------------------------------------------------------------------
    # PyMuPDF pass (structural)
    # ------------------------------------------------------------------
    if _HAVE_FITZ:
        try:
            _fitz_analysis(path, result)
        except Exception as exc:
            result["parse_errors"].append(f"PyMuPDF error: {exc}")
    else:
        result["parse_errors"].append("PyMuPDF (fitz) not available.")

    # ------------------------------------------------------------------
    # pdfminer pass (object-level)
    # ------------------------------------------------------------------
    if _HAVE_PDFMINER:
        try:
            _pdfminer_analysis(path, result)
        except Exception as exc:
            result["parse_errors"].append(f"pdfminer error: {exc}")
    else:
        result["parse_errors"].append("pdfminer.six not available.")

    return result


# ---------------------------------------------------------------------------
# Empty result skeleton
# ---------------------------------------------------------------------------
def _empty_result(file_path: str) -> Dict[str, Any]:
    return {
        "file_path": file_path,
        # Page info
        "page_count": 0,
        # Object counts
        "obj_count": 0,
        "stream_count": 0,
        "endstream_count": 0,
        "xref_count": 0,
        # JavaScript
        "has_javascript": 0,
        "javascript_count": 0,
        # Actions
        "has_openaction": 0,
        "has_aa": 0,
        "has_launch": 0,
        "action_count": 0,
        # Embedded content
        "embedded_file_count": 0,
        "has_richmedia": 0,
        # URLs
        "url_count": 0,
        "urls": [],
        "has_suspicious_url": 0,
        # Annotations
        "annotation_count": 0,
        "uri_annotation_count": 0,
        # Forms
        "has_acroform": 0,
        "has_xfa": 0,
        # Encryption / obfuscation
        "is_encrypted": 0,
        "has_objstm": 0,
        "has_xrefstm": 0,
        # Entropy
        "max_stream_entropy": 0.0,
        "avg_stream_entropy": 0.0,
        "high_entropy_stream_count": 0,
        # Metadata
        "parse_errors": [],
    }


# ---------------------------------------------------------------------------
# Raw-byte scanning
# ---------------------------------------------------------------------------
_SUSPICIOUS_TLD = re.compile(
    rb"https?://[^\s/]*\.(tk|ml|ga|cf|gq|xyz|top|pw|cc|ws)\b",
    re.IGNORECASE,
)

def _raw_scan(raw: bytes) -> Dict[str, Any]:
    d: Dict[str, Any] = {}

    d["has_javascript"]   = 1 if _RE_JS_KEYS.search(raw) else 0
    d["javascript_count"] = len(_RE_JS_KEYS.findall(raw))
    d["has_openaction"]   = 1 if _RE_OPENACTION.search(raw) else 0
    d["has_aa"]           = 1 if _RE_AA.search(raw) else 0
    d["has_launch"]       = 1 if _RE_LAUNCH.search(raw) else 0
    d["has_acroform"]     = 1 if _RE_ACROFORM.search(raw) else 0
    d["has_xfa"]          = 1 if _RE_XFA.search(raw) else 0
    d["is_encrypted"]     = 1 if _RE_ENCRYPT.search(raw) else 0
    d["has_objstm"]       = 1 if _RE_OBJSTM.search(raw) else 0
    d["has_xrefstm"]      = 1 if _RE_XREF_STM.search(raw) else 0
    d["has_richmedia"]    = 1 if _RE_RICHMEDIA.search(raw) else 0

    # Count /obj and stream markers
    d["obj_count"]        = raw.count(b" obj")
    d["stream_count"]     = raw.count(b"stream")
    d["endstream_count"]  = raw.count(b"endstream")
    d["xref_count"]       = raw.count(b"xref")

    # URL extraction
    urls = [u.decode("latin-1", errors="replace")
            for u in _RE_URI.findall(raw)]
    d["urls"]             = urls[:200]   # cap list size
    d["url_count"]        = len(urls)
    d["has_suspicious_url"] = 1 if _SUSPICIOUS_TLD.search(raw) else 0

    return d


# ---------------------------------------------------------------------------
# PyMuPDF analysis
# ---------------------------------------------------------------------------
def _fitz_analysis(path: Path, result: Dict[str, Any]) -> None:
    doc = fitz.open(str(path))
    result["page_count"]        = doc.page_count
    result["is_encrypted"]      = 1 if doc.is_encrypted else result["is_encrypted"]

    annotation_count    = 0
    uri_annotation_count = 0
    action_count        = 0
    entropy_values: List[float] = []

    for page in doc:
        # Annotations
        for annot in page.annots():
            annotation_count += 1
            if annot.uri:
                uri_annotation_count += 1
            info = annot.info
            if info.get("action"):
                action_count += 1

    result["annotation_count"]     = annotation_count
    result["uri_annotation_count"] = uri_annotation_count
    result["action_count"]         = max(result.get("action_count", 0), action_count)

    # Stream entropy via xref
    high_entropy = 0
    # xref_count() was renamed to xref_length() in PyMuPDF 1.28
    try:
        xref_len = doc.xref_count()
    except AttributeError:
        try:
            xref_len = doc.xref_length()
        except AttributeError:
            xref_len = 0

    for xref in range(1, xref_len):
        try:
            stream = doc.xref_stream(xref)
            if stream and len(stream) >= 64:
                ent = _byte_entropy(stream)
                entropy_values.append(ent)
                if ent >= 7.2:
                    high_entropy += 1
        except Exception:
            pass

    if entropy_values:
        result["max_stream_entropy"]        = round(max(entropy_values), 4)
        result["avg_stream_entropy"]        = round(
            sum(entropy_values) / len(entropy_values), 4
        )
        result["high_entropy_stream_count"] = high_entropy

    doc.close()


# ---------------------------------------------------------------------------
# pdfminer analysis
# ---------------------------------------------------------------------------
def _pdfminer_analysis(path: Path, result: Dict[str, Any]) -> None:
    with open(path, "rb") as fh:
        parser   = PDFParser(fh)
        document = PDFDocument(parser)

        embedded = 0
        try:
            catalog = resolve1(document.catalog)
            # Embedded files in the name tree
            names = catalog.get("Names")
            if names:
                names = resolve1(names)
                ef = names.get("EmbeddedFiles")
                if ef:
                    ef = resolve1(ef)
                    kids = ef.get("Names", [])
                    embedded = len(kids) // 2  # pairs: name, filespec
        except Exception:
            pass

        result["embedded_file_count"] = max(
            result.get("embedded_file_count", 0), embedded
        )


# ---------------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------------
def _byte_entropy(data: bytes) -> float:
    """Shannon entropy of a byte string (0–8 bits)."""
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    entropy = 0.0
    for c in counts:
        if c:
            p = c / n
            entropy -= p * math.log2(p)
    return entropy


# ---------------------------------------------------------------------------
# Feature key reference
# ---------------------------------------------------------------------------
FEATURE_KEYS = [
    "page_count",
    "obj_count", "stream_count", "endstream_count", "xref_count",
    "has_javascript", "javascript_count",
    "has_openaction", "has_aa", "has_launch", "action_count",
    "embedded_file_count", "has_richmedia",
    "url_count", "has_suspicious_url",
    "annotation_count", "uri_annotation_count",
    "has_acroform", "has_xfa",
    "is_encrypted", "has_objstm", "has_xrefstm",
    "max_stream_entropy", "avg_stream_entropy", "high_entropy_stream_count",
]


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_analyzer.py <path_to_pdf>")
        sys.exit(1)

    data = analyze_pdf(sys.argv[1])
    # Remove non-serialisable list of URLs for display
    display = {k: v for k, v in data.items() if k != "urls"}
    print(json.dumps(display, indent=2))
