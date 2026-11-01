"""
app.py  –  Malicious PDF Detection System
==========================================
Runs ALL trained models on every uploaded PDF, picks the best model
dynamically based on training F1 score, and displays a full comparison.

Run:
  streamlit run app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from file_validator import validate_pdf
from pdf_analyzer   import analyze_pdf
from pdf_features   import extract_features, FEATURE_COLUMNS
from risk_scorer    import compute_risk

try:
    from explainer import explain_prediction, native_importance_chart, fig_to_bytes
    _HAVE_EXPLAINER = True
except Exception:
    _HAVE_EXPLAINER = False

try:
    from report_generator import generate_report
    _HAVE_REPORT = True
except Exception:
    _HAVE_REPORT = False

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Malicious PDF Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title  { font-size:2.2rem; font-weight:800; color:#1a1a2e; margin-bottom:0; }
  .sub-title   { font-size:1rem;   color:#7f8c8d;   margin-top:0; }
  .section-hdr { font-size:1.1rem; font-weight:700; color:#2c3e50;
                 border-bottom:2px solid #3498db; padding-bottom:4px;
                 margin-top:1.4rem; margin-bottom:0.6rem; }
  .best-banner { background:linear-gradient(135deg,#1a1a2e,#16213e);
                 color:white; border-radius:10px; padding:16px 20px; margin:10px 0; }
  .card        { border-radius:10px; padding:14px; text-align:center; }
  .card-mal    { background:#fde8e8; border:2px solid #e74c3c; }
  .card-sus    { background:#fef9e7; border:2px solid #f39c12; }
  .card-ben    { background:#eafaf1; border:2px solid #27ae60; }
  .ev-hit      { background:#fde8e8; border-left:4px solid #e74c3c;
                 padding:7px 12px; margin:4px 0; border-radius:4px; }
  .ev-ok       { background:#eafaf1; border-left:4px solid #27ae60;
                 padding:7px 12px; margin:4px 0; border-radius:4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model registry  (loads once, cached)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FILES = {
    "Random Forest":       "random_forest.pkl",
    "XGBoost":             "xgboost.pkl",
    "Gradient Boosting":   "gradient_boosting.pkl",
    "SVM":                 "svm.pkl",
    "Logistic Regression": "logistic_regression.pkl",
}

@st.cache_resource(show_spinner="Loading models…")
def load_all_models() -> dict:
    """Load every available .pkl from models/. Returns {name: pipeline}."""
    loaded = {}
    for name, fname in MODEL_FILES.items():
        p = ROOT / "models" / fname
        if p.exists():
            try:
                loaded[name] = joblib.load(p)
            except Exception:
                pass
    return loaded

@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    p = ROOT / "models" / "training_meta.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Run all models on a feature vector
# ─────────────────────────────────────────────────────────────────────────────
def run_all_models(feat_dict: dict, meta: dict) -> dict:
    """
    Run every loaded model on feat_dict.
    Returns {model_name: {prob, label, f1, roc_auc, accuracy, recall, precision}}
    """
    models     = load_all_models()
    feat_cols  = meta.get("feature_names", FEATURE_COLUMNS)
    stored_met = meta.get("metrics", {})
    X = np.array([[feat_dict.get(c, 0) for c in feat_cols]], dtype=float)

    results = {}
    for name, pipe in models.items():
        try:
            prob  = float(pipe.predict_proba(X)[0][1])
            label = ("Malicious" if prob >= 0.6
                     else "Suspicious" if prob >= 0.3
                     else "Benign")
            # pull training metrics
            tm = {}
            for mkey, mval in stored_met.items():
                if mkey.lower() in name.lower() or name.lower() in mkey.lower():
                    tm = mval
                    break
            results[name] = {
                "prob":      prob,
                "label":     label,
                "f1":        tm.get("f1",        0.0),
                "roc_auc":   tm.get("roc_auc",   0.0),
                "accuracy":  tm.get("accuracy",  0.0),
                "recall":    tm.get("recall",    0.0),
                "precision": tm.get("precision", 0.0),
                "cv_f1":     tm.get("cv_f1_mean",0.0),
            }
        except Exception:
            pass
    return results


def pick_best(results: dict, meta: dict) -> str:
    """
    Select the best model for THIS specific PDF using majority vote.

    Steps:
    1. Each model votes for its predicted label (Benign/Suspicious/Malicious).
    2. Find the majority label (most votes).
    3. Among models that voted for the majority label, pick the one
       with the highest malicious probability (most decisive on this PDF).
    4. If all models are tied or only one model voted for majority, 
       pick by highest absolute distance from 0.5 (most confident).

    This produces a DIFFERENT winner for different PDFs — which is the
    correct behaviour. Training F1 is NOT used for selection here.
    """
    if not results:
        return ""
    if len(results) == 1:
        return list(results.keys())[0]

    from collections import Counter

    # Step 1 — majority vote
    label_votes  = Counter(d["label"] for d in results.values())
    majority_label = label_votes.most_common(1)[0][0]

    # Step 2 — models that agree with the majority
    majority_pool = {n: d for n, d in results.items()
                     if d["label"] == majority_label}

    # Step 3 — most confident among the majority pool
    # confidence = distance from the 0.5 decision boundary
    return max(majority_pool,
               key=lambda n: abs(majority_pool[n]["prob"] - 0.5))


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def _sidebar(meta: dict) -> tuple[bool, bool]:
    st.sidebar.markdown("## 🛡️ PDF Detector")
    st.sidebar.markdown("---")

    # Show which models are available
    models = load_all_models()
    stored = meta.get("metrics", {})
    st.sidebar.markdown("### 📦 Loaded Models")
    best_name = meta.get("best_model_name", "")
    for name in MODEL_FILES:
        if name in models:
            st.sidebar.markdown(
                f"<div style='font-size:0.82rem; padding:3px 0;'>✅ <b>{name}</b></div>",
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(
                f"<div style='font-size:0.82rem; color:#aaa; padding:3px 0;'>⬜ {name}</div>",
                unsafe_allow_html=True,
            )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Options")
    use_shap   = st.sidebar.checkbox("SHAP Explanations", value=True)
    gen_report = st.sidebar.checkbox("Generate PDF Report", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Score Thresholds")
    st.sidebar.markdown(
        "| Score | Verdict |\n|---|---|\n"
        "| 60–100 | 🔴 Malicious |\n"
        "| 30–59  | 🟡 Suspicious |\n"
        "| 0–29   | 🟢 Benign |"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Static analysis only · No file sent externally")
    return use_shap, gen_report


# ─────────────────────────────────────────────────────────────────────────────
# Risk card
# ─────────────────────────────────────────────────────────────────────────────
def _risk_card(risk, best_model_name: str, best_prob: float):
    cls = risk.classification.lower()
    css = {"malicious": "card-mal", "suspicious": "card-sus"}.get(cls, "card-ben")
    icon = {"malicious": "🔴", "suspicious": "🟡"}.get(cls, "🟢")
    color = {"malicious": "#e74c3c", "suspicious": "#f39c12"}.get(cls, "#27ae60")

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="card {css}">'
        f'<div style="font-size:0.8rem;color:#777;">RISK SCORE</div>'
        f'<div style="font-size:3.5rem;font-weight:900;color:{color};">{risk.risk_score:.0f}</div>'
        f'<div style="font-size:0.85rem;color:#777;">/ 100</div>'
        f'</div>', unsafe_allow_html=True)

    c2.markdown(
        f'<div class="card {css}">'
        f'<div style="font-size:2rem;">{icon}</div>'
        f'<div style="font-size:1.5rem;font-weight:800;color:{color};">'
        f'{risk.classification.upper()}</div>'
        f'<div style="font-size:0.8rem;color:#777;">Confidence: {risk.confidence}</div>'
        f'</div>', unsafe_allow_html=True)

    c3.markdown(
        f'<div class="card {css}">'
        f'<div style="font-size:0.8rem;color:#777;">ML PROBABILITY</div>'
        f'<div style="font-size:1.6rem;font-weight:800;color:{color};margin:6px 0;">{best_prob:.0%}</div>'
        f'<div style="font-size:0.75rem;color:#777;">malicious probability</div>'
        f'</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# All-models comparison
# ─────────────────────────────────────────────────────────────────────────────
def _all_models_panel(results: dict, best_name: str):
    st.markdown('<div class="section-hdr">📊 All Models — Results & Comparison</div>',
                unsafe_allow_html=True)

    # Per-model prediction cards — name, probability, verdict only
    cols = st.columns(len(results))
    for col, (name, d) in zip(cols, results.items()):
        c = ("#e74c3c" if d["label"] == "Malicious"
             else "#f39c12" if d["label"] == "Suspicious"
             else "#27ae60")
        col.markdown(
            f'<div style="border:1px solid {c};border-radius:10px;padding:16px 10px;'
            f'text-align:center;background:#fff;">'
            f'<div style="font-size:0.75rem;color:#888;font-weight:600;margin-bottom:6px;">{name}</div>'
            f'<div style="font-size:2.2rem;font-weight:900;color:{c};line-height:1.1;">'
            f'{d["prob"]:.0%}</div>'
            f'<div style="font-size:0.9rem;font-weight:700;color:{c};margin-top:6px;">'
            f'{d["label"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Agreement banner
    labels    = [d["label"] for d in results.values()]
    agree     = len(set(labels)) == 1
    ac        = "#27ae60" if agree else "#e67e22"
    at        = (f"✅ All {len(results)} models agree: <b>{labels[0]}</b> — high-confidence verdict."
                 if agree else
                 "⚠️ Models disagree. PDF sits near a decision boundary. "
                 "Check Rules and SHAP tabs for manual evidence.")
    st.markdown(
        f'<div style="background:#f8f9fa;border-left:5px solid {ac};'
        f'padding:10px 16px;border-radius:6px;margin-top:10px;font-size:0.9rem;">'
        f'{at}',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Verdict justification
# ─────────────────────────────────────────────────────────────────────────────
def _verdict_justification(risk, feat_dict: dict, best_name: str, best_d: dict):
    cls   = risk.classification.lower()
    score = risk.risk_score
    prob  = risk.ml_probability
    n_hits = len(risk.rule_hits)
    ent    = feat_dict.get("max_stream_entropy", 0.0)

    vc = {"malicious": "#c0392b", "suspicious": "#e67e22"}.get(cls, "#27ae60")
    vi = {"malicious": "🔴", "suspicious": "🟡"}.get(cls, "🟢")
    vt = {"malicious": "Strong malicious indicators detected.",
          "suspicious": "Suspicious characteristics found — treat with caution.",
          }.get(cls, "No significant malicious indicators detected.")

    st.markdown('<div class="section-hdr">🔎 Verdict Justification</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:#f8f9fa;border-left:5px solid {vc};'
        f'padding:12px 18px;border-radius:6px;margin:8px 0;">'
        f'<span style="font-size:1.2rem;">{vi}</span>'
        f'<span style="font-size:1.05rem;font-weight:700;color:{vc};">'
        f'&nbsp;{risk.classification.upper()}</span>'
        f'<span style="color:#555;margin-left:8px;">— {vt}</span></div>',
        unsafe_allow_html=True,
    )

    # Three signal columns
    c1, c2, c3 = st.columns(3)
    ml_c  = "#e74c3c" if prob >= 0.6 else ("#f39c12" if prob >= 0.3 else "#27ae60")
    ru_c  = "#e74c3c" if n_hits >= 3  else ("#f39c12" if n_hits >= 1 else "#27ae60")
    en_c  = "#e74c3c" if ent >= 7.2   else ("#f39c12" if ent >= 6.0  else "#27ae60")

    for col, title, val, sub, color in [
        (c1, "🤖 BEST MODEL", f"{prob:.0%}",
         f"{best_name}", ml_c),
        (c2, "📋 RULE ENGINE", str(n_hits),
         f"rule{'s' if n_hits!=1 else ''} triggered", ru_c),
        (c3, "🔐 ENTROPY", f"{ent:.2f}",
         "max stream entropy (0–8)", en_c),
    ]:
        col.markdown(
            f'<div style="border:1px solid #ddd;border-radius:8px;'
            f'padding:12px;text-align:center;">'
            f'<div style="font-size:0.75rem;color:#888;">{title}</div>'
            f'<div style="font-size:2rem;font-weight:800;color:{color};'
            f'line-height:1.1;margin:6px 0;">{val}</div>'
            f'<div style="font-size:0.78rem;color:{color};font-weight:600;">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Key evidence
    st.markdown("#### 📌 Key Evidence")
    evidence = []
    if feat_dict.get("has_javascript"):
        n = feat_dict.get("javascript_count", 1)
        evidence.append(("🔴","CRITICAL",f"JavaScript detected ({n} occurrence(s)) — rare in benign PDFs"))
    if feat_dict.get("has_openaction"):
        evidence.append(("🔴","CRITICAL","OpenAction trigger — executes automatically on PDF open"))
    if feat_dict.get("has_launch"):
        evidence.append(("🔴","CRITICAL","Launch action — can execute external programs"))
    if feat_dict.get("has_aa"):
        evidence.append(("🟠","HIGH","Additional-Action (AA) dictionary present"))
    if feat_dict.get("embedded_file_count", 0) > 0:
        evidence.append(("🟠","HIGH",f'{feat_dict["embedded_file_count"]} embedded file(s) detected'))
    if feat_dict.get("has_richmedia"):
        evidence.append(("🟠","HIGH","RichMedia (Flash/3D) object — historically exploited"))
    if feat_dict.get("has_xfa"):
        evidence.append(("🟠","HIGH","XFA form present — can contain executable logic"))
    if ent >= 7.2:
        evidence.append(("🟠","HIGH",f"Stream entropy {ent:.2f}/8.0 — heavily encoded/encrypted payload"))
    if feat_dict.get("has_suspicious_url"):
        evidence.append(("🟡","MEDIUM","Suspicious domain TLD found in URLs"))
    if feat_dict.get("has_objstm"):
        evidence.append(("🟡","MEDIUM","Object streams present — can hide objects from scanners"))
    if feat_dict.get("url_count", 0) >= 5:
        evidence.append(("🟡","MEDIUM",f'{feat_dict["url_count"]} URLs extracted — unusually high'))
    if not feat_dict.get("has_javascript") and not feat_dict.get("has_openaction"):
        evidence.append(("🟢","CLEAN","No JavaScript or auto-execute triggers — key benign indicator"))
    if ent < 6.0:
        evidence.append(("🟢","CLEAN",f"Normal stream entropy ({ent:.2f}) — no obfuscation detected"))
    if feat_dict.get("embedded_file_count", 0) == 0:
        evidence.append(("🟢","CLEAN","No embedded files — payload delivery risk reduced"))

    lc_map = {"CRITICAL":"#c0392b","HIGH":"#e67e22","MEDIUM":"#f39c12","CLEAN":"#27ae60"}
    for icon, lvl, desc in evidence:
        lc = lc_map.get(lvl, "#888")
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;margin:5px 0;'
            f'padding:7px 12px;border-radius:6px;background:#f8f9fa;">'
            f'<span style="font-size:1rem;margin-right:8px;">{icon}</span>'
            f'<span style="background:{lc};color:white;font-size:0.7rem;font-weight:700;'
            f'padding:2px 7px;border-radius:4px;margin-right:8px;white-space:nowrap;">{lvl}</span>'
            f'<span style="color:#333;font-size:0.88rem;">{desc}</span></div>',
            unsafe_allow_html=True,
        )

    # Score formula
    st.markdown("#### ⚖️ Score Formula")
    st.markdown(
        f'<div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:14px;">'
        f'<table style="width:100%;border-collapse:collapse;font-size:0.88rem;">'
        f'<tr style="background:#f0f0f0;">'
        f'<th style="padding:6px 10px;text-align:left;">Signal</th>'
        f'<th style="padding:6px 10px;text-align:center;">Value</th>'
        f'<th style="padding:6px 10px;text-align:center;">Weight</th>'
        f'<th style="padding:6px 10px;text-align:center;">Verdict</th></tr>'
        f'<tr><td style="padding:6px 10px;">Best Model Probability</td>'
        f'<td style="padding:6px 10px;text-align:center;"><b>{prob:.1%}</b></td>'
        f'<td style="padding:6px 10px;text-align:center;">70 %</td>'
        f'<td style="padding:6px 10px;text-align:center;">'
        f'{"🔴 Malicious" if prob>=0.6 else "🟡 Suspicious" if prob>=0.3 else "🟢 Benign"}</td></tr>'
        f'<tr style="background:#f9f9f9;">'
        f'<td style="padding:6px 10px;">Rule Hits</td>'
        f'<td style="padding:6px 10px;text-align:center;"><b>{n_hits} rules</b></td>'
        f'<td style="padding:6px 10px;text-align:center;">30 %</td>'
        f'<td style="padding:6px 10px;text-align:center;">'
        f'{"🔴 High" if n_hits>=3 else "🟡 Moderate" if n_hits>=1 else "🟢 Clean"}</td></tr>'
        f'<tr><td style="padding:6px 10px;font-weight:700;">Combined Risk Score</td>'
        f'<td style="padding:6px 10px;text-align:center;font-weight:700;">{score:.0f}/100</td>'
        f'<td style="padding:6px 10px;text-align:center;">Final</td>'
        f'<td style="padding:6px 10px;text-align:center;font-weight:700;color:{vc};">'
        f'{risk.classification.upper()} ({risk.confidence})</td></tr></table>'
        f'<div style="margin-top:8px;font-size:0.8rem;color:#888;">'
        f'Formula: ({prob:.2f} × 70) + {risk.rule_score:.1f} = <b>{score:.1f}/100</b></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature table
# ─────────────────────────────────────────────────────────────────────────────
def _feature_table(features: dict, risk):
    triggered = {rh.feature for rh in risk.rule_hits}
    display = {
        "has_javascript":"JavaScript Present","javascript_count":"JS Occurrences",
        "has_openaction":"OpenAction Trigger","has_aa":"Additional Actions (AA)",
        "has_launch":"Launch Action","action_count":"Action Count",
        "embedded_file_count":"Embedded Files","has_richmedia":"RichMedia",
        "url_count":"URLs Found","has_suspicious_url":"Suspicious URL TLD",
        "annotation_count":"Annotations","uri_annotation_count":"URI Annotations",
        "has_acroform":"AcroForm Present","has_xfa":"XFA Form",
        "is_encrypted":"Encrypted","has_objstm":"Object Streams",
        "has_xrefstm":"XRef Stream","page_count":"Pages",
        "obj_count":"Objects","stream_count":"Streams",
        "max_stream_entropy":"Max Entropy","avg_stream_entropy":"Avg Entropy",
        "high_entropy_stream_count":"High-Entropy Streams",
    }
    rows = [{"Indicator": lbl,
             "Value":     str(features.get(k, 0)),
             "Status":    "⚠️" if k in triggered else "✅"}
            for k, lbl in display.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Rule hits
# ─────────────────────────────────────────────────────────────────────────────
def _rule_hits(risk):
    if not risk.rule_hits:
        st.success("No rule-based indicators triggered.")
        return
    for rh in sorted(risk.rule_hits, key=lambda r: -r.weight):
        st.markdown(
            f'<div class="ev-hit">⚠️ <b>{rh.description}</b>'
            f'<span style="color:#888;font-size:0.82rem;"> '
            f'(value={rh.value}, threshold≥{rh.threshold}, '
            f'weight={rh.weight})</span></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SHAP explanation
# ─────────────────────────────────────────────────────────────────────────────
def _shap_tab(best_model, feat_dict: dict, meta: dict):
    if not _HAVE_EXPLAINER:
        st.warning("Explainer not available.")
        return
    feat_cols = meta.get("feature_names", FEATURE_COLUMNS)
    X = np.array([[feat_dict.get(c, 0) for c in feat_cols]], dtype=float)
    try:
        with st.spinner("Computing SHAP values…"):
            exp = explain_prediction(best_model, X, feat_cols, top_n=15)
        st.pyplot(exp["figure"], use_container_width=True)
        st.markdown("**Factors increasing risk:**")
        for nm, sv, fv in exp.get("top_positive", [])[:8]:
            st.markdown(
                f'<div class="ev-hit">🔺 <b>{nm}</b> = {fv:.3g} → +{sv:.4f}</div>',
                unsafe_allow_html=True)
        if exp.get("top_negative"):
            st.markdown("**Factors decreasing risk:**")
            for nm, sv, fv in exp["top_negative"][:5]:
                st.markdown(
                    f'<div class="ev-ok">🔻 <b>{nm}</b> = {fv:.3g} → {sv:.4f}</div>',
                    unsafe_allow_html=True)
    except Exception as exc:
        st.warning(f"SHAP failed: {exc}")
        fig = native_importance_chart(best_model, feat_cols)
        if fig:
            st.pyplot(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis pipeline
# ─────────────────────────────────────────────────────────────────────────────
def _run_analysis(uploaded_file, use_shap: bool, gen_report: bool):
    meta    = load_meta()
    models  = load_all_models()

    if not models:
        st.error("No trained models found in `models/`. "
                 "Run `python src/train_models.py` first.")
        return

    tmp = ROOT / "data" / f"_upload_{uploaded_file.name}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(uploaded_file.read())

    prog   = st.progress(0, text="Validating file…")
    t0     = time.time()

    try:
        # 1. Validate
        val = validate_pdf(tmp)
        prog.progress(10, text="Static analysis…")

        # 2. Analyse
        analysis = analyze_pdf(tmp)
        prog.progress(25, text="Extracting features…")

        # 3. Features
        feat_dict, _ = extract_features(tmp)
        prog.progress(40, text=f"Running {len(models)} models…")

        # 4. Run ALL models
        results = run_all_models(feat_dict, meta)
        if not results:
            st.error("All models failed to produce a prediction.")
            return
        prog.progress(65, text="Selecting best model…")

        # 5. Pick best model dynamically
        best_name  = pick_best(results, meta)
        best_d     = results[best_name]
        best_model = models[best_name]
        ml_prob    = best_d["prob"]

        # 6. Risk score
        risk = compute_risk(
            features=feat_dict,
            ml_probability=ml_prob,
            file_path=str(tmp),
            sha256=val.sha256,
        )
        prog.progress(78, text="Generating explanation…")

        # 7. SHAP explanation
        explanation = None
        if use_shap and _HAVE_EXPLAINER:
            try:
                feat_cols = meta.get("feature_names", FEATURE_COLUMNS)
                X = np.array([[feat_dict.get(c, 0) for c in feat_cols]], dtype=float)
                explanation = explain_prediction(best_model, X, feat_cols, top_n=15)
            except Exception:
                pass
        prog.progress(90, text="Generating report…")

        # 8. Report
        report_path = None
        if gen_report and _HAVE_REPORT:
            try:
                report_path = generate_report(
                    validation_result=val,
                    analysis=analysis,
                    features=feat_dict,
                    risk_result=risk,
                    explanation=explanation,
                    model_name=best_name,
                )
            except Exception as exc:
                st.warning(f"Report generation failed: {exc}")

        prog.progress(100, text=f"Done in {time.time()-t0:.1f}s")
        time.sleep(0.3)
        prog.empty()

        # ── Display ────────────────────────────────────────────────────
        st.markdown("---")

        # Risk card
        st.markdown('<div class="section-hdr">🎯 Risk Assessment</div>',
                    unsafe_allow_html=True)
        _risk_card(risk, best_name, ml_prob)

        # All-models comparison — hidden, logic still runs internally

        # Verdict justification
        _verdict_justification(risk, feat_dict, best_name, best_d)

        # File info
        st.markdown('<div class="section-hdr">📄 File Information</div>',
                    unsafe_allow_html=True)
        # File info — only show errors/warnings, hide routine valid status
        ci1, ci2 = st.columns(2)
        with ci1:
            st.markdown(f"**File:** `{uploaded_file.name}`")
            st.markdown(f"**Size:** {val.file_size_bytes:,} bytes")
            st.markdown(f"**PDF Version:** {val.pdf_version}")
            st.markdown(f"**Pages:** {analysis.get('page_count','N/A')}")
        with ci2:
            st.markdown(f"**SHA-256:** `{val.sha256[:32]}…`")
            # Only show validation status if something went wrong
            if not val.is_valid:
                st.error("❌ File validation failed")
            for e in val.errors:   st.error(f"❌ {e}")
            for w in val.warnings: st.caption(f"⚠️ {w}")

        # Tabs
        t1, t2, t3, t4, t5 = st.tabs([
            "📊 Features", "⚠️ Rules", "🧠 SHAP", "🔗 URLs", "🔍 Raw"
        ])
        with t1:
            st.markdown('<div class="section-hdr">Static Analysis Features</div>',
                        unsafe_allow_html=True)
            _feature_table(feat_dict, risk)
        with t2:
            st.markdown('<div class="section-hdr">Rule-Based Indicators</div>',
                        unsafe_allow_html=True)
            _rule_hits(risk)
        with t3:
            st.markdown('<div class="section-hdr">SHAP — Best Model Explanation</div>',
                        unsafe_allow_html=True)
            if explanation:
                _shap_tab(best_model, feat_dict, meta)
            elif use_shap:
                st.info("SHAP computation skipped (model type may not support TreeExplainer).")
            else:
                st.info("Enable SHAP in the sidebar to see this tab.")
        with t4:
            urls = analysis.get("urls", [])
            if urls:
                st.markdown(f"**{len(urls)} URL(s) extracted:**")
                for u in urls[:50]: st.code(u, language=None)
            else:
                st.success("No URLs found.")
        with t5:
            st.json({k: v for k, v in analysis.items() if k != "urls"})

        # Download report
        if report_path and Path(report_path).exists():
            st.markdown("---")
            with open(report_path, "rb") as fh:
                rb = fh.read()
            suf = Path(report_path).suffix
            st.download_button(
                label=f"📥 Download Security Report ({suf.upper()})",
                data=rb,
                file_name=Path(report_path).name,
                mime="application/pdf" if suf == ".pdf" else "text/plain",
                use_container_width=True,
            )

    finally:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# Home page
# ─────────────────────────────────────────────────────────────────────────────
def _home():
    meta   = load_meta()
    models = load_all_models()

    st.markdown(
        '<div style="background:#eaf3fb;border-radius:10px;padding:20px;margin-bottom:20px;">'
        '<h3 style="margin-top:0;">How it works</h3>'
        '<ol>'
        '<li>Upload a PDF — it is analysed <b>locally</b>, never sent externally.</li>'
        '<li>Static analysis extracts <b>25 structural features</b> without executing the file.</li>'
        f'<li>All <b>{len(models)} trained models</b> score the PDF simultaneously.</li>'
        '<li>The <b>best model</b> (highest training F1) is auto-selected for the final verdict.</li>'
        '<li>A <b>0–100 risk score</b> combines ML probability (70%) + rule hits (30%).</li>'
        '<li><b>SHAP</b> explains which features drove the decision.</li>'
        '</ol></div>',
        unsafe_allow_html=True,
    )

    # Model count on home page — no metrics table
    c1, c2, c3 = st.columns(3)
    c1.metric("Models", len(models))
    c2.metric("Features", "25")
    c3.metric("Analysis", "Static + ML")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown(
        '<h1 class="main-title">🛡️ Malicious PDF Detector</h1>'
        '<p class="sub-title">5 models · Dynamic best-model selection · '
        'SHAP explainability · Static analysis only</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    meta               = load_meta()
    use_shap, gen_rpt  = _sidebar(meta)

    uploaded = st.file_uploader(
        "📂  Drop a PDF here to analyse",
        type=["pdf"],
        help="Up to 200 MB. Analysed locally — never sent to external servers.",
    )

    if uploaded is None:
        _home()
        return

    st.markdown(f"**Analysing:** `{uploaded.name}` ({uploaded.size:,} bytes)")
    _run_analysis(uploaded, use_shap, gen_rpt)


if __name__ == "__main__":
    main()
