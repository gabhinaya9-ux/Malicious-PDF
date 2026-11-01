"""
dataset_builder.py
------------------
Builds the labelled training dataset from raw PDF files OR from an
existing structured dataset CSV (e.g. the CIC-Evasive-PDFMal2022 /
Contagio / PDGUM datasets which ship as pre-extracted feature CSVs).

Modes
-----
A) FROM_FILES  – point to two directories (benign/ malicious/) and this
                 module runs the full pdf_features pipeline on every file.

B) FROM_CSV    – point to a pre-extracted CSV that already has feature
                 columns + a label column.  The module normalises column
                 names and saves a clean version to data/pdf_features.csv.

Usage (CLI)
-----------
  # Mode A
  python dataset_builder.py --mode files \
      --benign  data/raw/benign \
      --malicious data/raw/malicious \
      --out data/pdf_features.csv

  # Mode B – using pre-extracted dataset (e.g. PDGUM / CIC)
  python dataset_builder.py --mode csv \
      --input data/raw/dataset.csv \
      --label-col class \
      --out data/pdf_features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Local
from pdf_features import FEATURE_COLUMNS, extract_features_batch


# ---------------------------------------------------------------------------
# Mode A – extract from PDF files on disk
# ---------------------------------------------------------------------------
def build_from_files(
    benign_dir: str | Path,
    malicious_dir: str | Path,
    out_csv: str | Path = "data/pdf_features.csv",
    glob: str = "**/*.pdf",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Walk *benign_dir* (label=0) and *malicious_dir* (label=1), extract
    features for every PDF, and save a combined CSV.
    """
    benign_paths   = sorted(Path(benign_dir).glob(glob))
    malicious_paths = sorted(Path(malicious_dir).glob(glob))

    if verbose:
        print(f"Found {len(benign_paths)} benign PDFs")
        print(f"Found {len(malicious_paths)} malicious PDFs")

    paths  = benign_paths + malicious_paths
    labels = [0] * len(benign_paths) + [1] * len(malicious_paths)

    df = extract_features_batch(paths, labels, save_csv=out_csv, verbose=verbose)
    _print_summary(df)
    return df


# ---------------------------------------------------------------------------
# Mode B – normalise a pre-extracted CSV
# ---------------------------------------------------------------------------
# Known column mappings for popular public datasets
_COLUMN_ALIASES = {
    # CIC-Evasive-PDFMal2022 / similar
    "Class":          "label",
    "class":          "label",
    "Label":          "label",
    "label":          "label",
    "legitimate":     "label",   # some datasets use 0=legit,1=mal
    # Common feature name variants
    "pdfsize":        "file_size_bytes",
    "metadata size":  "metadata_size",
    "pages":          "page_count",
    "isEncrypted":    "is_encrypted",
    "embedded files": "embedded_file_count",
    "JS":             "has_javascript",
    "Javascript":     "has_javascript",
    "Header":         "pdf_version",
    "obj":            "obj_count",
    "endobj":         "endobj_count",
    "stream":         "stream_count",
    "endstream":      "endstream_count",
    "xref":           "xref_count",
    "startxref":      "startxref_count",
    "pageno":         "page_count",
    "ActionCount":    "action_count",
    "OpenAction":     "has_openaction",
    "AcroForm":       "has_acroform",
    "XFA":            "has_xfa",
    "ObjStm":         "has_objstm",
    "URICount":       "url_count",
}

def build_from_csv(
    input_csv: str | Path,
    label_col: str = "label",
    out_csv: str | Path = "data/pdf_features.csv",
    label_map: Optional[dict] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load *input_csv*, rename columns to canonical names, binarise the
    label column, and save a clean copy to *out_csv*.

    Parameters
    ----------
    input_csv  : path to the raw dataset CSV
    label_col  : name of the label column *before* renaming
    out_csv    : destination path
    label_map  : optional dict to map original label values → 0/1
                 e.g. {"legitimate": 0, "malicious": 1}
    """
    df = pd.read_csv(input_csv, low_memory=False)
    if verbose:
        print(f"Loaded {len(df):,} rows × {len(df.columns)} columns from {input_csv}")

    # Rename columns
    rename = {old: new for old, new in _COLUMN_ALIASES.items() if old in df.columns}
    df.rename(columns=rename, inplace=True)

    # Handle the label column (may have been renamed or may be user-supplied)
    if label_col in df.columns and label_col != "label":
        df.rename(columns={label_col: "label"}, inplace=True)

    if "label" not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. "
            f"Available columns: {list(df.columns)[:20]}"
        )

    # Binarise label
    if label_map:
        df["label"] = df["label"].map(label_map)
    else:
        # Try to auto-detect: if strings → use first occurrence heuristic
        unique_vals = df["label"].dropna().unique()
        if not all(isinstance(v, (int, float, np.integer)) for v in unique_vals):
            # Alphabetic sort → first gets 0 (benign), second gets 1 (malicious)
            sorted_vals = sorted(str(v) for v in unique_vals)
            auto_map = {v: i for i, v in enumerate(sorted_vals)}
            if verbose:
                print(f"Auto label mapping: {auto_map}")
            df["label"] = df["label"].astype(str).map(auto_map)
        else:
            df["label"] = df["label"].astype(int)

    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

    # Drop non-numeric columns that aren't metadata
    _keep_meta = {"file_path", "sha256", "label", "valid", "error"}
    numeric_cols = []
    for col in df.columns:
        if col in _keep_meta:
            numeric_cols.append(col)
        else:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                numeric_cols.append(col)
            except Exception:
                pass

    df = df[numeric_cols].copy()
    df.fillna(0, inplace=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if verbose:
        _print_summary(df)
        print(f"\nClean dataset saved → {out_csv}")

    return df


# ---------------------------------------------------------------------------
# Train / validation / test split utility
# ---------------------------------------------------------------------------
def split_dataset(
    csv_path: str | Path,
    test_size: float = 0.20,
    val_size:  float = 0.10,
    random_state: int = 42,
    stratify: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train / validation / test DataFrames and save
    them as separate CSVs alongside the source file.

    Returns (train_df, val_df, test_df)
    """
    src = Path(csv_path)
    df  = pd.read_csv(src)

    strat = df["label"] if stratify and "label" in df.columns else None

    # First cut: train+val vs test
    df_trainval, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat
    )

    # Second cut: train vs val
    val_ratio = val_size / (1 - test_size)
    strat2 = df_trainval["label"] if stratify and "label" in df_trainval.columns else None
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_ratio, random_state=random_state, stratify=strat2
    )

    stem = src.stem
    df_train.to_csv(src.parent / f"{stem}_train.csv", index=False)
    df_val.to_csv(  src.parent / f"{stem}_val.csv",   index=False)
    df_test.to_csv( src.parent / f"{stem}_test.csv",  index=False)

    if verbose:
        print(f"Split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _print_summary(df: pd.DataFrame) -> None:
    print(f"\n── Dataset Summary ─────────────────────────")
    print(f"  Total samples : {len(df):,}")
    if "label" in df.columns:
        vc = df["label"].value_counts()
        benign    = vc.get(0, 0)
        malicious = vc.get(1, 0)
        print(f"  Benign        : {benign:,}")
        print(f"  Malicious     : {malicious:,}")
        ratio = malicious / max(benign, 1)
        print(f"  Mal/Ben ratio : {ratio:.2f}")
    print(f"  Columns       : {len(df.columns)}")
    print(f"────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build labelled PDF feature dataset")
    p.add_argument("--mode", choices=["files", "csv"], default="csv")
    # Mode A
    p.add_argument("--benign",    default="data/raw/benign")
    p.add_argument("--malicious", default="data/raw/malicious")
    # Mode B
    p.add_argument("--input",     default="data/raw/dataset.csv")
    p.add_argument("--label-col", default="label")
    # Common
    p.add_argument("--out", default="data/pdf_features.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.mode == "files":
        build_from_files(args.benign, args.malicious, out_csv=args.out)
    else:
        build_from_csv(args.input, label_col=args.label_col, out_csv=args.out)
