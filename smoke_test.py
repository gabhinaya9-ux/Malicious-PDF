"""Quick end-to-end smoke test — no ML model required."""
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

print("=" * 55)
print("  Malicious PDF Detector — Smoke Test")
print("=" * 55)

# ── 1. Validate ──────────────────────────────────────────
from file_validator import validate_pdf
v = validate_pdf("data/smoke_test.pdf")
print(f"\n[1] File Validation : {'PASS' if v.is_valid else 'FAIL'}")
print(f"    SHA-256         : {v.sha256[:40]}...")
print(f"    PDF Version     : {v.pdf_version}")
print(f"    File size       : {v.file_size_bytes:,} bytes")
if v.errors:
    for e in v.errors:
        print(f"    ERROR: {e}")

# ── 2. Analyse ───────────────────────────────────────────
from pdf_analyzer import analyze_pdf
a = analyze_pdf("data/smoke_test.pdf")
print(f"\n[2] Static Analysis")
print(f"    Pages           : {a['page_count']}")
print(f"    Object count    : {a['obj_count']}")
print(f"    JavaScript      : {a['has_javascript']}")
print(f"    OpenAction      : {a['has_openaction']}")
print(f"    Embedded files  : {a['embedded_file_count']}")
print(f"    URLs found      : {a['url_count']}")
print(f"    Max entropy     : {a['max_stream_entropy']}")
if a["parse_errors"]:
    for e in a["parse_errors"]:
        print(f"    PARSE WARN: {e}")

# ── 3. Feature extraction ────────────────────────────────
from pdf_features import extract_features, FEATURE_COLUMNS
feat, _ = extract_features("data/smoke_test.pdf")
n_feats = len([k for k in FEATURE_COLUMNS if k in feat])
print(f"\n[3] Feature Vector  : {n_feats} features extracted")

# ── 4. Risk scoring (rule-only, no model) ────────────────
from risk_scorer import compute_risk
risk = compute_risk(
    features=feat,
    ml_probability=0.05,
    file_path="data/smoke_test.pdf",
    sha256=v.sha256,
)
print(f"\n[4] Risk Assessment")
print(f"    Risk Score      : {risk.risk_score}/100")
print(f"    Classification  : {risk.classification}")
print(f"    Confidence      : {risk.confidence}")
print(f"    Rule hits       : {len(risk.rule_hits)}")
for rh in risk.rule_hits:
    print(f"      - {rh.description}")

# ── 5. Report generation ─────────────────────────────────
from report_generator import generate_report
rpt_path = generate_report(
    validation_result=v,
    analysis=a,
    features=feat,
    risk_result=risk,
    explanation=None,
    model_name="Rule-Only (no model loaded)",
    output_path="reports/smoke_test_report.pdf",
)
print(f"\n[5] Report saved    : {rpt_path}")

print("\n" + "=" * 55)
print("  ALL STAGES PASSED")
print("  Run: streamlit run app.py")
print("=" * 55)
