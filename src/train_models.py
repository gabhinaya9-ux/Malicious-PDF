"""
train_models.py
---------------
Trains FIVE classifiers on the labelled PDF feature dataset:

  1. Random Forest
  2. XGBoost
  3. Gradient Boosting
  4. Support Vector Machine (SVM)
  5. Logistic Regression

Each model is wrapped in a sklearn Pipeline (StandardScaler → Classifier).
All models are evaluated on a held-out test split and via 5-fold CV.
The best model (by F1) is saved as best_model.pkl.

Outputs (models/)
-----------------
  random_forest.pkl
  xgboost.pkl
  gradient_boosting.pkl
  svm.pkl
  logistic_regression.pkl
  best_model.pkl
  model_comparison.csv
  feature_importance.csv
  training_meta.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    _HAVE_XGB = True
except ImportError:
    _HAVE_XGB = False
    print("[WARN] XGBoost not installed — skipping XGBoost model.")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from pdf_features import FEATURE_COLUMNS, load_feature_matrix

MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_model(model: Pipeline, X_test: np.ndarray,
                   y_test: np.ndarray, name: str) -> Dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred),                    4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0),  4),
        "recall":    round(recall_score(y_test, y_pred,    zero_division=0),  4),
        "f1":        round(f1_score(y_test, y_pred,        zero_division=0),  4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),                    4),
    }
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    print(f"\n── {name} {'─'*(48-len(name))}")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {k:<12}: {metrics[k]:.4f}")
    print(f"  Confusion  : TN={tn} FP={fp} FN={fn} TP={tp}")
    print(classification_report(y_test, y_pred,
                                 target_names=["Benign", "Malicious"]))
    return metrics


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------
def _rf(rs=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=rs,
        )),
    ])

def _xgb(spw=1.0, rs=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", scale_pos_weight=float(spw),
            random_state=rs, verbosity=0,
        )),
    ])

def _gb(rs=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=rs,
        )),
    ])

def _svm(rs=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, class_weight="balanced",
            random_state=rs,
        )),
    ])

def _lr(rs=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=rs,
        )),
    ])


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    dataset_csv:  str | Path = "data/pdf_features.csv",
    test_size:    float = 0.20,
    random_state: int   = 42,
    cv_folds:     int   = 5,
    models_dir:   str | Path = "models",
) -> Dict:

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {dataset_csv}")
    X, y, feature_names = load_feature_matrix(dataset_csv)
    print(f"  Samples   : {len(X):,}")
    print(f"  Features  : {len(feature_names)}")
    print(f"  Benign    : {int((y==0).sum()):,}")
    print(f"  Malicious : {int((y==1).sum()):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train     : {len(X_train):,}  |  Test: {len(X_test):,}")

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = neg / max(pos, 1)

    # ── 2. Define models to train ────────────────────────────────────────
    model_defs = [
        ("Random Forest",      "random_forest.pkl",      _rf(random_state)),
        ("Gradient Boosting",  "gradient_boosting.pkl",  _gb(random_state)),
        ("SVM",                "svm.pkl",                _svm(random_state)),
        ("Logistic Regression","logistic_regression.pkl",_lr(random_state)),
    ]
    if _HAVE_XGB:
        model_defs.insert(1, ("XGBoost", "xgboost.pkl", _xgb(spw, random_state)))

    cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                   random_state=random_state)

    all_metrics:  List[Dict] = []
    saved_models: Dict[str, str] = {}
    trained:      Dict[str, Pipeline] = {}

    # ── 3. Train & evaluate each model ───────────────────────────────────
    total = len(model_defs)
    for idx, (name, fname, pipe) in enumerate(model_defs, 1):
        print(f"\n[{idx}/{total}] Training {name} …")
        pipe.fit(X_train, y_train)

        cv_f1 = cross_val_score(pipe, X_train, y_train,
                                 cv=cv_splitter, scoring="f1", n_jobs=-1)
        print(f"  CV F1 ({cv_folds}-fold): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

        m = evaluate_model(pipe, X_test, y_test, name)
        m["cv_f1_mean"] = round(float(cv_f1.mean()), 4)
        m["cv_f1_std"]  = round(float(cv_f1.std()),  4)
        all_metrics.append(m)

        path = models_dir / fname
        joblib.dump(pipe, path)
        saved_models[name] = str(path)
        trained[name]      = pipe
        print(f"  Saved → {path}")

    # ── 4. Model comparison table ────────────────────────────────────────
    comp_df = pd.DataFrame(all_metrics)
    comp_df = comp_df.sort_values("f1", ascending=False).reset_index(drop=True)
    comp_path = models_dir / "model_comparison.csv"
    comp_df.to_csv(comp_path, index=False)

    print("\n\n" + "="*60)
    print("  MODEL COMPARISON (sorted by F1)")
    print("="*60)
    print(comp_df[["model","accuracy","precision","recall","f1","roc_auc",
                   "cv_f1_mean"]].to_string(index=False))

    # ── 5. Select best model ─────────────────────────────────────────────
    best_row  = comp_df.iloc[0]
    best_name = best_row["model"]
    best_pipe = trained[best_name]

    best_path = models_dir / "best_model.pkl"
    joblib.dump(best_pipe, best_path)
    saved_models["Best Model"] = str(best_path)

    print(f"\n{'='*60}")
    print(f"  ✓ BEST MODEL : {best_name}")
    print(f"    F1         : {best_row['f1']:.4f}")
    print(f"    ROC-AUC    : {best_row['roc_auc']:.4f}")
    print(f"    Accuracy   : {best_row['accuracy']:.4f}")
    print(f"  Saved → {best_path}")
    print(f"{'='*60}\n")

    # ── 6. Feature importance (tree-based models only) ────────────────────
    fi_saved = False
    for name in [best_name, "Random Forest", "Gradient Boosting", "XGBoost"]:
        pipe = trained.get(name)
        if pipe is None:
            continue
        clf = pipe.named_steps["clf"]
        importances = getattr(clf, "feature_importances_", None)
        if importances is not None:
            fi_df = pd.DataFrame({
                "feature":    feature_names,
                "importance": importances,
            }).sort_values("importance", ascending=False)
            fi_path = models_dir / "feature_importance.csv"
            fi_df.to_csv(fi_path, index=False)
            print(f"── Top 10 Features ({name}) ──────────────────────────")
            print(fi_df.head(10).to_string(index=False))
            fi_saved = True
            break

    # ── 7. Save metadata ─────────────────────────────────────────────────
    meta = {
        "best_model_name": best_name,
        "best_model_path": str(best_path),
        "feature_names":   feature_names,
        "model_paths":     saved_models,
        "metrics": {m["model"]: m for m in all_metrics},
    }
    meta_path = models_dir / "training_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nMetadata saved → {meta_path}")
    print("✓ Training complete.\n")

    return saved_models


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Train PDF malware classifiers (5 models)")
    p.add_argument("--dataset",      default="data/pdf_features.csv")
    p.add_argument("--test-size",    type=float, default=0.20)
    p.add_argument("--cv-folds",     type=int,   default=5)
    p.add_argument("--random-state", type=int,   default=42)
    p.add_argument("--models-dir",   default="models")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        dataset_csv  = args.dataset,
        test_size    = args.test_size,
        cv_folds     = args.cv_folds,
        random_state = args.random_state,
        models_dir   = args.models_dir,
    )
