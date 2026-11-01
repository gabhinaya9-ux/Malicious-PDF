# 🛡️ Malicious PDF Detection System

A complete, end-to-end pipeline for detecting malicious PDF files using static
analysis, machine learning (Random Forest + XGBoost), SHAP explainability, and
a Streamlit web interface.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Setup](#setup)
5. [Workflow: Step by Step](#workflow-step-by-step)
   - [Step 1 – Get a Dataset](#step-1--get-a-dataset)
   - [Step 2 – Build Feature Dataset](#step-2--build-feature-dataset)
   - [Step 3 – Train Models](#step-3--train-models)
   - [Step 4 – Run the Web App](#step-4--run-the-web-app)
   - [Step 5 – CLI Tools](#step-5--cli-tools)
6. [Feature Vector (25 features)](#feature-vector-25-features)
7. [Risk Scoring](#risk-scoring)
8. [Model Outputs](#model-outputs)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

| Component | What it does |
|---|---|
| **File Validator** | Checks extension, magic bytes, size, EOF marker, SHA-256 |
| **PDF Analyzer** | Static structural analysis — no PDF execution |
| **Feature Extractor** | Converts analysis output into a 25-element numeric vector |
| **Dataset Builder** | Builds labelled training data from PDF files or existing CSVs |
| **Model Trainer** | Trains RF + XGBoost, 5-fold CV, saves best model |
| **Risk Scorer** | Combines ML probability (70%) + rule hits (30%) → 0–100 score |
| **Explainer** | SHAP waterfall charts showing which features drove the prediction |
| **Report Generator** | Exports a formatted PDF security report |
| **Streamlit App** | Web UI — upload a PDF, get instant results |

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────────┐
│  file_validator │  → SHA-256 hash, size/header checks
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│  pdf_analyzer   │  → JS, actions, URLs, entropy, objects …
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│  pdf_features   │  → 25-column numeric feature vector
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌─────────┐
│  RF   │ │ XGBoost │   ← trained models (models/)
└───┬───┘ └────┬────┘
    └────┬─────┘
         │  ML probability
    ▼
┌─────────────────┐
│  risk_scorer    │  → 0-100 score + classification
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌──────────┐ ┌──────────────────┐
│ explainer│ │ report_generator │
└──────────┘ └──────────────────┘
         │
    ▼
┌─────────────────┐
│   Streamlit UI  │  → browser
└─────────────────┘
```

---

## File Structure

```
MALICIOUS PDF/
├── app.py                      ← Streamlit web application
├── requirements.txt
├── README.md
│
├── src/
│   ├── file_validator.py       ← Step 1: PDF validation + SHA-256
│   ├── pdf_analyzer.py         ← Step 2: deep static analysis
│   ├── pdf_features.py         ← Step 3: feature vector extraction
│   ├── dataset_builder.py      ← Step 4: build labelled dataset
│   ├── train_models.py         ← Step 5: train RF + XGBoost
│   ├── risk_scorer.py          ← Step 6: 0-100 risk score
│   ├── explainer.py            ← Step 7: SHAP explanations
│   └── report_generator.py     ← Step 8: PDF report generation
│
├── data/
│   ├── raw/                    ← put raw PDFs or dataset CSVs here
│   │   ├── benign/             ← benign PDF files (Mode A)
│   │   └── malicious/          ← malicious PDF files (Mode A)
│   ├── pdf_features.csv        ← generated feature dataset
│   ├── pdf_features_train.csv
│   ├── pdf_features_val.csv
│   └── pdf_features_test.csv
│
├── models/
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── best_model.pkl
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   └── training_meta.json
│
├── reports/                    ← generated analysis reports
└── tests/
```

---

## Setup

### Prerequisites

- Python 3.9 or higher
- pip

### Install dependencies

```powershell
# From the project root
pip install -r requirements.txt
```

> **Windows note:** `python-magic-bin` (in requirements.txt) provides
> file-type detection on Windows without needing libmagic installed separately.

### Verify installation

```powershell
python -c "import fitz, sklearn, xgboost, shap, streamlit; print('All OK')"
```

---

## Workflow: Step by Step

### Step 1 – Get a Dataset

You need labelled PDF samples. Three good free sources:

| Dataset | URL | Notes |
|---|---|---|
| **CIC-Evasive-PDFMal2022** | https://www.unb.ca/cic/datasets/ | Pre-extracted CSV, ~50k samples |
| **PDGUM** | Kaggle search "pdf malware dataset" | Pre-extracted CSV |
| **Contagio** | http://contagiodump.blogspot.com | Raw PDF files (password: infected) |

**Using a pre-extracted CSV (recommended for quick start):**

Download the CSV and place it in `data/raw/dataset.csv`.

```powershell
cd src
python dataset_builder.py --mode csv `
    --input ..\data\raw\dataset.csv `
    --label-col class `
    --out ..\data\pdf_features.csv
```

**Using raw PDF files:**

```powershell
cd src
python dataset_builder.py --mode files `
    --benign    ..\data\raw\benign `
    --malicious ..\data\raw\malicious `
    --out ..\data\pdf_features.csv
```

---

### Step 2 – Build Feature Dataset

The command above already does this. The output is `data/pdf_features.csv`
with one row per PDF and 25 numeric feature columns plus a `label` column
(`0 = benign`, `1 = malicious`).

To also create train/val/test splits:

```python
# Python snippet
import sys; sys.path.insert(0, 'src')
from dataset_builder import split_dataset
split_dataset('data/pdf_features.csv', test_size=0.20, val_size=0.10)
```

---

### Step 3 – Train Models

```powershell
cd src
python train_models.py --dataset ..\data\pdf_features.csv
```

This will:
- Train a Random Forest (300 trees, balanced class weights)
- Train an XGBoost (300 estimators, adjusted `scale_pos_weight`)
- Run 5-fold stratified cross-validation on the training set
- Evaluate both models on the held-out 20% test set
- Print a side-by-side comparison table
- Save the best model (by F1) as `models/best_model.pkl`
- Save feature importances to `models/feature_importance.csv`

**Expected output:**

```
── Random Forest ──────────────────────────────────────
  Accuracy  : 0.9850
  Precision : 0.9821
  Recall    : 0.9876
  F1        : 0.9848
  ROC-AUC   : 0.9961

── XGBoost ────────────────────────────────────────────
  Accuracy  : 0.9891
  Precision : 0.9867
  Recall    : 0.9912
  F1        : 0.9889
  ROC-AUC   : 0.9978

  ✓ Best model: XGBoost  (F1=0.9889)
```

*(Actual numbers depend on your dataset.)*

---

### Step 4 – Run the Web App

```powershell
# From the project root
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

**What you can do in the app:**

1. Select the detection model in the sidebar (RF / XGBoost / Best Model)
2. Toggle SHAP explanations and PDF report generation
3. Drag & drop or browse to upload a PDF
4. View:
   - 🎯 Risk score gauge (0–100) with classification badge
   - 📊 **Features tab** — all 25 extracted indicators
   - ⚠️ **Rules tab** — triggered rule-based indicators
   - 🧠 **Explanation tab** — SHAP waterfall chart
   - 🔗 **URLs tab** — extracted hyperlinks
   - 🔍 **Raw tab** — full JSON analysis output
5. Download the generated PDF security report

---

### Step 5 – CLI Tools

Every module can be used standalone:

```powershell
# Validate a PDF
python src/file_validator.py path/to/file.pdf

# Full static analysis (JSON output)
python src/pdf_analyzer.py path/to/file.pdf

# Extract feature vector
python src/pdf_features.py path/to/file.pdf

# Score a PDF (requires trained model)
python src/risk_scorer.py path/to/file.pdf models/best_model.pkl

# SHAP global importance (requires features CSV and model)
python src/explainer.py models/best_model.pkl data/pdf_features.csv
```

---

## Feature Vector (25 features)

| # | Feature | Description |
|---|---|---|
| 1 | `page_count` | Number of pages |
| 2 | `obj_count` | Total PDF objects |
| 3 | `stream_count` | Stream object count |
| 4 | `endstream_count` | Endstream marker count |
| 5 | `xref_count` | Cross-reference table count |
| 6 | `has_javascript` | Binary: JS present |
| 7 | `javascript_count` | JS keyword occurrences |
| 8 | `has_openaction` | Binary: OpenAction present |
| 9 | `has_aa` | Binary: Additional-Action dict |
| 10 | `has_launch` | Binary: Launch action |
| 11 | `action_count` | Total action annotations |
| 12 | `embedded_file_count` | Embedded file objects |
| 13 | `has_richmedia` | Binary: RichMedia (Flash/3D) |
| 14 | `url_count` | Extracted URL count |
| 15 | `has_suspicious_url` | Binary: suspicious TLD |
| 16 | `annotation_count` | Total annotations |
| 17 | `uri_annotation_count` | URI-type annotations |
| 18 | `has_acroform` | Binary: AcroForm present |
| 19 | `has_xfa` | Binary: XFA form present |
| 20 | `is_encrypted` | Binary: PDF encrypted |
| 21 | `has_objstm` | Binary: Object streams |
| 22 | `has_xrefstm` | Binary: XRef stream |
| 23 | `max_stream_entropy` | Max Shannon entropy of streams |
| 24 | `avg_stream_entropy` | Mean Shannon entropy of streams |
| 25 | `high_entropy_stream_count` | Streams with entropy ≥ 7.2 bits |

---

## Risk Scoring

```
Risk Score = (ML probability × 70) + (Rule score × 30 / max_rule_weight)
```

| Score | Classification |
|---|---|
| 60–100 | 🔴 Malicious |
| 30–59  | 🟡 Suspicious |
| 0–29   | 🟢 Benign |

**16 weighted rules** covering: JavaScript, OpenAction, Launch, AA,
embedded files, RichMedia, XFA, ObjStm, suspicious URLs, high entropy,
encryption, and more.

---

## Model Outputs

| File | Description |
|---|---|
| `models/random_forest.pkl` | Trained RF pipeline (scaler + classifier) |
| `models/xgboost.pkl` | Trained XGBoost pipeline |
| `models/best_model.pkl` | Best model by test-set F1 |
| `models/model_comparison.csv` | Side-by-side metrics table |
| `models/feature_importance.csv` | Feature importances (best model) |
| `models/training_meta.json` | Feature names, metrics, paths |
| `reports/report_*.pdf` | Per-file security reports |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'fitz'`**
```powershell
pip install PyMuPDF==1.24.0
```

**`ModuleNotFoundError: No module named 'xgboost'`**
```powershell
pip install xgboost==2.0.3
```

**App says "No trained model found"**

You need to train first:
```powershell
python src/train_models.py --dataset data/pdf_features.csv
```

**SHAP takes a long time**

TreeExplainer is fast for RF/XGBoost. If it still feels slow, disable SHAP
in the sidebar and use native feature importance instead.

**fpdf2 not found (text report fallback)**
```powershell
pip install fpdf2==2.7.9
```

---

## Disclaimer

This tool performs **static analysis only**. It does not execute or sandbox
the PDF. Static analysis can produce false positives (benign files flagged)
and false negatives (novel malware not detected). It is a screening tool —
not a replacement for dynamic sandboxing or expert forensic review.
