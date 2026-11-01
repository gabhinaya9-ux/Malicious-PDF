"""
file_validator.py
-----------------
Validates an uploaded file to confirm it is a real, well-formed PDF
before any deeper analysis is performed.

Checks performed:
  1. Extension must be .pdf (case-insensitive)
  2. File size must be between MIN_SIZE and MAX_SIZE
  3. PDF magic-bytes header (%PDF-)
  4. PDF EOF marker (%%EOF) present somewhere in the last 1 KB
  5. File is openable by PyMuPDF without a hard crash

Returns a ValidationResult dataclass with pass/fail status and details.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MIN_SIZE_BYTES = 64           # smaller than this is almost certainly not a PDF
MAX_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB upper limit

PDF_HEADER_MAGIC = b"%PDF-"
PDF_EOF_MARKER   = b"%%EOF"
EOF_SEARCH_WINDOW = 1024     # bytes from end to search for %%EOF


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    is_valid: bool
    file_path: str
    file_size_bytes: int
    sha256: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    pdf_version: str = "unknown"

    def summary(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [
            f"[{status}] {self.file_path}",
            f"  Size    : {self.file_size_bytes:,} bytes",
            f"  SHA-256 : {self.sha256}",
            f"  Version : {self.pdf_version}",
        ]
        for e in self.errors:
            lines.append(f"  ERROR   : {e}")
        for w in self.warnings:
            lines.append(f"  WARNING : {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------
def validate_pdf(file_path: str | Path) -> ValidationResult:
    """
    Run all validation checks on *file_path*.

    Parameters
    ----------
    file_path : str or Path
        Path to the file to validate.

    Returns
    -------
    ValidationResult
        Populated result object; ``is_valid`` is True only when every
        mandatory check passes.
    """
    path = Path(file_path)
    errors: List[str] = []
    warnings: List[str] = []
    sha256 = "N/A"
    pdf_version = "unknown"
    file_size = 0

    # ------------------------------------------------------------------
    # 1. Existence
    # ------------------------------------------------------------------
    if not path.exists():
        return ValidationResult(
            is_valid=False,
            file_path=str(path),
            file_size_bytes=0,
            sha256=sha256,
            errors=[f"File does not exist: {path}"],
        )

    if not path.is_file():
        return ValidationResult(
            is_valid=False,
            file_path=str(path),
            file_size_bytes=0,
            sha256=sha256,
            errors=[f"Path is not a regular file: {path}"],
        )

    # ------------------------------------------------------------------
    # 2. Extension check
    # ------------------------------------------------------------------
    if path.suffix.lower() != ".pdf":
        errors.append(
            f"File extension is '{path.suffix}'; expected '.pdf'."
        )

    # ------------------------------------------------------------------
    # 3. Size check
    # ------------------------------------------------------------------
    file_size = path.stat().st_size

    if file_size < MIN_SIZE_BYTES:
        errors.append(
            f"File too small ({file_size} bytes); minimum is {MIN_SIZE_BYTES} bytes."
        )
    elif file_size > MAX_SIZE_BYTES:
        errors.append(
            f"File too large ({file_size:,} bytes); maximum is {MAX_SIZE_BYTES:,} bytes."
        )

    # ------------------------------------------------------------------
    # 4. Read raw bytes (needed for magic bytes + hash)
    # ------------------------------------------------------------------
    try:
        with open(path, "rb") as fh:
            raw_header = fh.read(1024)
            fh.seek(max(0, file_size - EOF_SEARCH_WINDOW))
            raw_tail = fh.read()
    except OSError as exc:
        errors.append(f"Cannot read file: {exc}")
        return ValidationResult(
            is_valid=False,
            file_path=str(path),
            file_size_bytes=file_size,
            sha256=sha256,
            errors=errors,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # 5. Magic-bytes check (%PDF-)
    # ------------------------------------------------------------------
    if not raw_header.startswith(PDF_HEADER_MAGIC):
        # Some PDFs start with a BOM or whitespace; search first 16 bytes
        if PDF_HEADER_MAGIC not in raw_header[:16]:
            errors.append(
                "File does not start with PDF magic bytes (%PDF-)."
            )
        else:
            warnings.append(
                "PDF magic bytes found but not at byte 0 (unusual)."
            )
    else:
        # Extract PDF version string, e.g. "%PDF-1.7"
        try:
            version_line = raw_header[:12].decode("latin-1", errors="replace")
            pdf_version = version_line[1:8].strip()  # e.g. "PDF-1.7"
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 6. EOF marker check
    # ------------------------------------------------------------------
    if PDF_EOF_MARKER not in raw_tail:
        warnings.append(
            "PDF EOF marker (%%%%EOF) not found in the last 1 KB of the file. "
            "The file may be truncated or corrupt."
        )

    # ------------------------------------------------------------------
    # 7. SHA-256 hash
    # ------------------------------------------------------------------
    sha256 = _sha256_file(path)

    # ------------------------------------------------------------------
    # 8. Structural sanity via PyMuPDF (optional, soft check)
    # ------------------------------------------------------------------
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz  # legacy
        doc = fitz.open(str(path))
        page_count = doc.page_count
        doc.close()
        if page_count == 0:
            warnings.append("PDF has 0 pages — may be a stub or heavily corrupted.")
    except ImportError:
        warnings.append("PyMuPDF not installed; structural check skipped.")
    except Exception as exc:
        warnings.append(f"PyMuPDF could not open file cleanly: {exc}")

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        file_path=str(path),
        file_size_bytes=file_size,
        sha256=sha256,
        errors=errors,
        warnings=warnings,
        pdf_version=pdf_version,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file in 64 KB chunks (memory-efficient)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python file_validator.py <path_to_pdf>")
        sys.exit(1)

    result = validate_pdf(sys.argv[1])
    print(result.summary())
    sys.exit(0 if result.is_valid else 1)
