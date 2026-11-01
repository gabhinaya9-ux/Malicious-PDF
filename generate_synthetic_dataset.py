"""
generate_synthetic_dataset.py
------------------------------
Generates a realistic synthetic labelled PDF feature dataset for training.

Calibrated against real-world PDF research (PDGUM, Contagio, CIC-Evasive-PDFMal2022):
  - Benign PDFs: can have entropy up to 7.5 (compressed images are normal)
  - Malicious PDFs: distinguished by JS + OpenAction + Launch combinations,
    NOT just by entropy alone

Generates 5,000 benign + 5,000 malicious samples = 10,000 rows total.
"""

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

N_BENIGN    = 5000
N_MALICIOUS = 5000

print(f"Generating {N_BENIGN + N_MALICIOUS:,} synthetic PDF feature samples...")

def bernoulli(p, n):
    return rng.binomial(1, p, n).astype(int)

def poisson_clip(lam, n, lo=0, hi=None):
    v = rng.poisson(lam, n).astype(int)
    return np.clip(v, lo, hi if hi is not None else v.max() + 1)

def uniform_ent(low, high, n):
    return rng.uniform(low, high, n).round(4)

# ============================================================================
# BENIGN — calibrated to match real PDFs (research papers, forms, reports)
# Key insight: entropy can be HIGH in benign PDFs due to JPEG/image compression
# The distinguishing features are JS, OpenAction, Launch — NOT entropy alone
# ============================================================================
b = {}
b["page_count"]                = poisson_clip(12, N_BENIGN, lo=1, hi=800)
b["obj_count"]                 = poisson_clip(80, N_BENIGN, lo=10, hi=600)
b["stream_count"]              = poisson_clip(25, N_BENIGN, lo=2, hi=150)
b["endstream_count"]           = b["stream_count"].copy()
b["xref_count"]                = rng.integers(1, 5, N_BENIGN)
# Benign PDFs almost never have JavaScript
b["has_javascript"]            = bernoulli(0.01, N_BENIGN)
b["javascript_count"]          = b["has_javascript"] * poisson_clip(0.3, N_BENIGN, hi=2)
# Benign PDFs rarely have OpenAction
b["has_openaction"]            = bernoulli(0.02, N_BENIGN)
b["has_aa"]                    = bernoulli(0.01, N_BENIGN)
# Launch action is extremely rare in benign
b["has_launch"]                = bernoulli(0.001, N_BENIGN)
b["action_count"]              = poisson_clip(0.5, N_BENIGN, hi=5)
b["embedded_file_count"]       = poisson_clip(0.05, N_BENIGN, hi=2)
b["has_richmedia"]             = bernoulli(0.005, N_BENIGN)
b["url_count"]                 = poisson_clip(2.5, N_BENIGN, hi=30)
b["has_suspicious_url"]        = bernoulli(0.005, N_BENIGN)
b["annotation_count"]          = poisson_clip(5.0, N_BENIGN, hi=80)
b["uri_annotation_count"]      = poisson_clip(2.0, N_BENIGN, hi=30)
b["has_acroform"]              = bernoulli(0.20, N_BENIGN)
b["has_xfa"]                   = bernoulli(0.01, N_BENIGN)
b["is_encrypted"]              = bernoulli(0.08, N_BENIGN)
b["has_objstm"]                = bernoulli(0.25, N_BENIGN)
b["has_xrefstm"]               = bernoulli(0.20, N_BENIGN)
# CRITICAL FIX: benign PDFs with images/compression have entropy 5.0–7.8
# This is NORMAL — JPEG streams have very high entropy
b["max_stream_entropy"]        = uniform_ent(4.5, 7.8, N_BENIGN)
b["avg_stream_entropy"]        = (b["max_stream_entropy"] * rng.uniform(0.55, 0.90, N_BENIGN)).round(4)
b["high_entropy_stream_count"] = poisson_clip(1.5, N_BENIGN, hi=8)
b["parse_error_count"]         = bernoulli(0.02, N_BENIGN)
b["label"]                     = np.zeros(N_BENIGN, dtype=int)

# ============================================================================
# MALICIOUS — the distinguishing factor is BEHAVIOUR indicators, not entropy
# JS + OpenAction + Launch combinations = exploit delivery mechanism
# ============================================================================
m = {}
m["page_count"]                = poisson_clip(4, N_MALICIOUS, lo=1, hi=30)
m["obj_count"]                 = poisson_clip(50, N_MALICIOUS, lo=5, hi=400)
m["stream_count"]              = poisson_clip(15, N_MALICIOUS, lo=0, hi=100)
m["endstream_count"]           = m["stream_count"].copy()
m["xref_count"]                = rng.integers(1, 8, N_MALICIOUS)
# Malicious PDFs very commonly have JavaScript
m["has_javascript"]            = bernoulli(0.85, N_MALICIOUS)
m["javascript_count"]          = m["has_javascript"] * poisson_clip(8, N_MALICIOUS, lo=1, hi=40)
# OpenAction is a primary malicious indicator
m["has_openaction"]            = bernoulli(0.80, N_MALICIOUS)
m["has_aa"]                    = bernoulli(0.50, N_MALICIOUS)
# Launch action strongly indicates malicious
m["has_launch"]                = bernoulli(0.45, N_MALICIOUS)
m["action_count"]              = poisson_clip(6, N_MALICIOUS, lo=0, hi=25)
m["embedded_file_count"]       = poisson_clip(2.0, N_MALICIOUS, hi=12)
m["has_richmedia"]             = bernoulli(0.25, N_MALICIOUS)
m["url_count"]                 = poisson_clip(6, N_MALICIOUS, hi=60)
m["has_suspicious_url"]        = bernoulli(0.50, N_MALICIOUS)
m["annotation_count"]          = poisson_clip(5, N_MALICIOUS, hi=70)
m["uri_annotation_count"]      = poisson_clip(4, N_MALICIOUS, hi=50)
m["has_acroform"]              = bernoulli(0.40, N_MALICIOUS)
m["has_xfa"]                   = bernoulli(0.30, N_MALICIOUS)
m["is_encrypted"]              = bernoulli(0.35, N_MALICIOUS)
m["has_objstm"]                = bernoulli(0.60, N_MALICIOUS)
m["has_xrefstm"]               = bernoulli(0.45, N_MALICIOUS)
# Malicious entropy overlaps with benign — entropy ALONE is not decisive
m["max_stream_entropy"]        = uniform_ent(5.0, 8.0, N_MALICIOUS)
m["avg_stream_entropy"]        = (m["max_stream_entropy"] * rng.uniform(0.70, 0.98, N_MALICIOUS)).round(4)
m["high_entropy_stream_count"] = poisson_clip(3, N_MALICIOUS, lo=0, hi=15)
m["parse_error_count"]         = bernoulli(0.20, N_MALICIOUS)
m["label"]                     = np.ones(N_MALICIOUS, dtype=int)

# ── Combine and shuffle ──────────────────────────────────────────────────────
df = pd.concat([pd.DataFrame(b), pd.DataFrame(m)], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out = Path("data/pdf_features.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print(f"Dataset saved → {out}")
print(f"  Total  : {len(df):,}")
print(f"  Benign : {(df.label==0).sum():,}")
print(f"  Malicious : {(df.label==1).sum():,}")

print(f"\nFeature distributions (mean by class):")
key_feats = ["has_javascript","has_openaction","has_launch",
             "embedded_file_count","max_stream_entropy","url_count","has_objstm"]
for f in key_feats:
    bm = df[df.label==0][f].mean()
    mm = df[df.label==1][f].mean()
    print(f"  {f:<35} benign={bm:.3f}  malicious={mm:.3f}")
