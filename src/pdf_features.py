"""
pdf_features.py
---------------
Converts a single PDF file into a fixed-length numerical feature vector
suitable for machine-learning models.

Pipeline
--------
1. Validate the file  (file_validator.py)
2. Run static analysis (pdf_analyzer.py)
3. Assemble the 25-element feature vector
4. Optionally persist to CSV

Feature vector definition is stored in FEATURE_COLUMNS so every module
that needs the column order can import it from one place.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Local modules
from file_validator import validate_pdf, ValidationResult
from pdf_analyzer import analyze_pdf, FEATURE_KEYS


# ---------------------------------------------------------------------------
# Canonical column order  (must stay stable once training data is generated)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: List[str] = FEATURE_KEYS  # 25 features


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------
def extract_features(
    file_path: str | Path,
    label: Optional[int] = None,
) -> Tuple[Dict, ValidationResult]:
    """
    Extract the feature vector for one PDF.

    Parameters
    ----------
    file_path : str or Path
    label     : 0 (benign) or 1 (malicious), or None for unseen PDFs

    Returns
    -------
    (feature_dict, validation_result)

    ``feature_dict`` contains:
      - all keys from FEATURE_COLUMNS  (numeric values)
      - "file_path"   (str)
      - "sha256"      (str)
      - "label"       (int or None)
      - "valid"       (bool)
      - "error"       (str — populated on failure)
    """
    path = Path(file_path)

    # ── 1. Validate ─────────────────────────────────────────────────────
    val = validate_pdf(path)

    base: Dict = {
        "file_path": str(path),
        "sha256": val.sha256,
        "label": label,
        "valid": val.is_valid,
        "error": "; ".join(val.errors) if val.errors else "",
    }

    if not val.is_valid:
        # Return zero vector for invalid files so CSV rows are uniform
        zero_features = {col: 0 for col in FEATURE_COLUMNS}
        zero_features["max_stream_entropy"] = 0.0
        zero_features["avg_stream_entropy"] = 0.0
        return {**base, **zero_features}, val

    # ── 2. Analyse ──────────────────────────────────────────────────────
    analysis = analyze_pdf(path)

    # ── 3. Assemble vector ──────────────────────────────────────────────
    features: Dict = {}
    for col in FEATURE_COLUMNS:
        raw = analysis.get(col, 0)
        # Coerce to numeric; lists → length
        if isinstance(raw, list):
            features[col] = len(raw)
        elif isinstance(raw, bool):
            features[col] = int(raw)
        else:
            try:
                features[col] = float(raw) if "entropy" in col else int(raw)
            except (TypeError, ValueError):
                features[col] = 0

    # parse_error_count is stored for display only — NOT in FEATURE_COLUMNS
    # so it is never passed to the model (which was trained on exactly 25 cols)
    features["parse_error_count"] = len(analysis.get("parse_errors", []))

    return {**base, **features}, val


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------
def extract_features_batch(
    file_paths: List[str | Path],
    labels: Optional[List[int]] = None,
    save_csv: Optional[str | Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract features for a list of PDFs and return a DataFrame.

    Parameters
    ----------
    file_paths : list of paths
    labels     : parallel list of 0/1 labels (or None)
    save_csv   : if given, write the DataFrame to this CSV path
    verbose    : print progress

    Returns
    -------
    pd.DataFrame with one row per PDF
    """
    if labels is None:
        labels = [None] * len(file_paths)  # type: ignore[list-item]

    rows = []
    for i, (fp, lbl) in enumerate(zip(file_paths, labels)):
        if verbose:
            print(f"  [{i+1}/{len(file_paths)}] {Path(fp).name}", end=" ")
        feat, val = extract_features(fp, label=lbl)
        rows.append(feat)
        if verbose:
            status = "OK" if feat["valid"] else f"INVALID ({feat['error'][:40]})"
            print(f"→ {status}")

    df = pd.DataFrame(rows)

    # Ensure canonical column order
    cols_present = [c for c in FEATURE_COLUMNS if c in df.columns]
    meta_cols    = ["file_path", "sha256", "label", "valid", "error"]
    ordered_cols = meta_cols + cols_present
    df = df[[c for c in ordered_cols if c in df.columns]]

    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        if verbose:
            print(f"\nFeature CSV saved → {save_csv}")

    return df


# ---------------------------------------------------------------------------
# Utility: load feature matrix ready for sklearn
# ---------------------------------------------------------------------------
def load_feature_matrix(
    csv_path: str | Path,
    drop_invalid: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a feature CSV produced by extract_features_batch.

    Returns
    -------
    X        : (n_samples, n_features) float array
    y        : (n_samples,) int array  — labels
    feature_names : list of column names matching X columns
    """
    df = pd.read_csv(csv_path)

    if drop_invalid and "valid" in df.columns:
        df = df[df["valid"] == True].copy()

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(int) if "label" in df.columns else np.zeros(len(df), dtype=int)

    return X, y, feature_cols


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python pdf_features.py <path_to_pdf> [label]")
        sys.exit(1)

    fp  = sys.argv[1]
    lbl = int(sys.argv[2]) if len(sys.argv) > 2 else None

    feat, val = extract_features(fp, label=lbl)
    print("\n── Validation ─────────────────────────────────────")
    print(val.summary())
    print("\n── Feature Vector ──────────────────────────────────")
    for k, v in feat.items():
        if k not in ("file_path", "sha256", "label", "valid", "error"):
            print(f"  {k:<35} {v}")
