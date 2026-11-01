"""
explainer.py
------------
SHAP-based explainability for the PDF malware detector.

Two entry points:
  1. explain_prediction()  – explains a single prediction (waterfall / force plot)
  2. explain_dataset()     – global feature importance over a dataset (beeswarm)

All chart functions return matplotlib Figure objects so they can be
rendered both in notebooks and embedded in the Streamlit app.
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import shap
    _HAVE_SHAP = True
except ImportError:
    _HAVE_SHAP = False

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


# ---------------------------------------------------------------------------
# SHAP explainer factory
# ---------------------------------------------------------------------------
def build_shap_explainer(model_pipeline, feature_names: List[str]):
    """
    Construct an appropriate SHAP explainer for the underlying classifier.

    Supports Random Forest and XGBoost (tree explainer); falls back to
    KernelExplainer for other model types.
    """
    if not _HAVE_SHAP:
        raise ImportError("shap is not installed. Run: pip install shap")

    clf = model_pipeline.named_steps["clf"]
    clf_type = type(clf).__name__

    # TreeExplainer is fastest and most accurate for tree-based models
    if clf_type in ("RandomForestClassifier", "XGBClassifier",
                    "GradientBoostingClassifier", "ExtraTreesClassifier"):
        explainer = shap.TreeExplainer(clf)
    else:
        # Fallback: slower but universal
        scaler = model_pipeline.named_steps.get("scaler")
        def _predict(X):
            if scaler is not None:
                X = scaler.transform(X)
            return model_pipeline.predict_proba(X)[:, 1]

        explainer = shap.KernelExplainer(_predict, shap.sample(
            np.zeros((1, len(feature_names))), 50
        ))

    return explainer


# ---------------------------------------------------------------------------
# Single-prediction explanation
# ---------------------------------------------------------------------------
def explain_prediction(
    model_pipeline,
    feature_vector: np.ndarray,           # shape (1, n_features) or (n_features,)
    feature_names: List[str],
    top_n: int = 15,
) -> Dict[str, Any]:
    """
    Explain one prediction.

    Returns
    -------
    dict with keys:
      shap_values     – np.ndarray of SHAP values for the malicious class
      base_value      – float, the model's expected output
      feature_names   – list of names
      feature_values  – np.ndarray of actual feature values
      top_positive    – list of (feature, shap_val, feat_val) sorted desc
      top_negative    – list of (feature, shap_val, feat_val) sorted asc
      figure          – matplotlib Figure (waterfall bar chart)
    """
    X = np.array(feature_vector).reshape(1, -1)

    # Apply scaler if present (TreeExplainer works on raw clf input)
    scaler = model_pipeline.named_steps.get("scaler")
    X_scaled = scaler.transform(X) if scaler is not None else X

    explainer  = build_shap_explainer(model_pipeline, feature_names)
    shap_vals  = explainer.shap_values(X_scaled)

    # For binary classifiers: shap_values may be [class0, class1] or just array
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]          # malicious class, first sample
        base = explainer.expected_value[1]
    else:
        sv = shap_vals[0]
        base = float(explainer.expected_value)

    # Sort by absolute SHAP value
    pairs = sorted(zip(feature_names, sv, X[0]), key=lambda t: abs(t[1]), reverse=True)
    top_n_pairs = pairs[:top_n]

    top_positive = [(n, v, fv) for n, v, fv in top_n_pairs if v > 0]
    top_negative = [(n, v, fv) for n, v, fv in top_n_pairs if v <= 0]

    fig = _waterfall_chart(top_n_pairs, base)

    return {
        "shap_values":    sv,
        "base_value":     base,
        "feature_names":  feature_names,
        "feature_values": X[0],
        "top_positive":   top_positive,
        "top_negative":   top_negative,
        "figure":         fig,
        "top_n_pairs":    top_n_pairs,
    }


# ---------------------------------------------------------------------------
# Dataset-level explanation
# ---------------------------------------------------------------------------
def explain_dataset(
    model_pipeline,
    X: np.ndarray,
    feature_names: List[str],
    max_samples: int = 500,
) -> Dict[str, Any]:
    """
    Compute global SHAP values over a sample of the dataset.

    Returns
    -------
    dict with:
      shap_matrix  – np.ndarray (n_samples, n_features)
      mean_abs     – pd.Series  feature → mean |SHAP| (importance ranking)
      figure       – matplotlib Figure (horizontal bar chart)
    """
    scaler  = model_pipeline.named_steps.get("scaler")
    X_scaled = scaler.transform(X) if scaler is not None else X

    # Sub-sample for speed
    if len(X_scaled) > max_samples:
        idx = np.random.choice(len(X_scaled), max_samples, replace=False)
        X_scaled = X_scaled[idx]

    explainer = build_shap_explainer(model_pipeline, feature_names)
    shap_matrix = explainer.shap_values(X_scaled)

    if isinstance(shap_matrix, list):
        shap_matrix = shap_matrix[1]   # malicious class

    mean_abs = pd.Series(
        np.abs(shap_matrix).mean(axis=0),
        index=feature_names,
    ).sort_values(ascending=False)

    fig = _global_importance_chart(mean_abs)

    return {
        "shap_matrix": shap_matrix,
        "mean_abs":    mean_abs,
        "figure":      fig,
    }


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def _waterfall_chart(
    top_pairs: List[Tuple[str, float, float]],
    base_value: float,
) -> Optional["plt.Figure"]:
    """Horizontal waterfall / contribution bar chart for one prediction."""
    if not _HAVE_MPL:
        return None

    names = [p[0] for p in top_pairs]
    shap_v = [p[1] for p in top_pairs]
    feat_v = [p[2] for p in top_pairs]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))

    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_v]
    bars   = ax.barh(
        [f"{n}\n(val={v:.2g})" for n, v in zip(names, feat_v)],
        shap_v,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value  →  pushes toward Malicious", fontsize=11)
    ax.set_title(
        f"Feature Contributions  (base value = {base_value:.3f})",
        fontsize=13, fontweight="bold",
    )

    red_p   = mpatches.Patch(color="#e74c3c", label="Increases risk")
    green_p = mpatches.Patch(color="#2ecc71", label="Decreases risk")
    ax.legend(handles=[red_p, green_p], loc="lower right", fontsize=9)

    plt.tight_layout()
    return fig


def _global_importance_chart(mean_abs: "pd.Series") -> Optional["plt.Figure"]:
    """Horizontal bar chart of global mean |SHAP| per feature."""
    if not _HAVE_MPL:
        return None

    top = mean_abs.head(20)
    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.45)))

    ax.barh(top.index[::-1], top.values[::-1], color="#3498db", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|  (global feature importance)", fontsize=11)
    ax.set_title("Global Feature Importance (SHAP)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fallback: model-native feature importance (no SHAP)
# ---------------------------------------------------------------------------
def native_importance_chart(
    model_pipeline,
    feature_names: List[str],
    top_n: int = 20,
) -> Optional["plt.Figure"]:
    """
    Bar chart using sklearn / XGBoost feature_importances_.
    Used when SHAP is unavailable.
    """
    if not _HAVE_MPL:
        return None

    clf = model_pipeline.named_steps["clf"]
    importances = getattr(clf, "feature_importances_", None)
    if importances is None:
        return None

    s = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, len(s) * 0.45)))
    ax.barh(s.index[::-1], s.values[::-1], color="#9b59b6", edgecolor="white")
    ax.set_xlabel("Feature Importance (Gini / Gain)", fontsize=11)
    ax.set_title("Model Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save figure to bytes (for Streamlit / report embedding)
# ---------------------------------------------------------------------------
def fig_to_bytes(fig: "plt.Figure", fmt: str = "png", dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import joblib

    if len(sys.argv) < 3:
        print("Usage: python explainer.py <model.pkl> <features.csv>")
        sys.exit(1)

    model = joblib.load(sys.argv[1])
    df    = pd.read_csv(sys.argv[2])

    # Drop meta columns
    meta_cols = {"file_path", "sha256", "label", "valid", "error"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X_all     = df[feat_cols].fillna(0).values.astype(float)

    print(f"Computing global SHAP over {len(X_all)} samples …")
    result = explain_dataset(model, X_all, feat_cols)
    print("\nTop 10 features by mean |SHAP|:")
    print(result["mean_abs"].head(10).to_string())

    if result["figure"]:
        result["figure"].savefig("models/shap_global.png", dpi=150, bbox_inches="tight")
        print("\nSaved → models/shap_global.png")
