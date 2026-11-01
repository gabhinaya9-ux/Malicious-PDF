"""Quick test to verify pick_best selects differently for different PDFs."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import json, joblib
import numpy as np
from pdf_features import FEATURE_COLUMNS

with open("models/training_meta.json") as f:
    meta = json.load(f)
feat_cols = meta["feature_names"]

MODEL_FILES = {
    "Random Forest":       "models/random_forest.pkl",
    "XGBoost":             "models/xgboost.pkl",
    "Gradient Boosting":   "models/gradient_boosting.pkl",
    "SVM":                 "models/svm.pkl",
    "Logistic Regression": "models/logistic_regression.pkl",
}
models = {n: joblib.load(p) for n, p in MODEL_FILES.items()}

from collections import Counter
def pick_best(results):
    if len(results) == 1:
        return list(results.keys())[0]
    label_votes   = Counter(d["label"] for d in results.values())
    majority_label = label_votes.most_common(1)[0][0]
    majority_pool  = {n: d for n, d in results.items() if d["label"] == majority_label}
    return max(majority_pool, key=lambda n: abs(majority_pool[n]["prob"] - 0.5))

def run(features_dict, scenario):
    X = np.array([[features_dict.get(c, 0) for c in feat_cols]], dtype=float)
    results = {}
    for name, pipe in models.items():
        prob  = float(pipe.predict_proba(X)[0][1])
        label = "Malicious" if prob >= 0.6 else "Suspicious" if prob >= 0.3 else "Benign"
        results[name] = {"prob": prob, "label": label}
    best = pick_best(results)
    print(f"\n{scenario}")
    for n, d in results.items():
        marker = " ← SELECTED" if n == best else ""
        print(f"  {n:<25} {d['prob']:.1%}  {d['label']}{marker}")

# Scenario 1: clearly benign PDF
run({c: 0 for c in feat_cols}, "SCENARIO 1: Clearly Benign (all zeros)")

# Scenario 2: suspicious PDF (JS + OpenAction)
s2 = {c: 0 for c in feat_cols}
s2["has_javascript"] = 1
s2["javascript_count"] = 5
s2["has_openaction"] = 1
s2["action_count"] = 4
s2["url_count"] = 8
s2["max_stream_entropy"] = 6.5
s2["avg_stream_entropy"] = 5.8
run(s2, "SCENARIO 2: Suspicious (JS + OpenAction + URLs)")

# Scenario 3: highly malicious PDF
s3 = {c: 0 for c in feat_cols}
s3["has_javascript"] = 1
s3["javascript_count"] = 15
s3["has_openaction"] = 1
s3["has_launch"] = 1
s3["has_aa"] = 1
s3["embedded_file_count"] = 3
s3["has_objstm"] = 1
s3["action_count"] = 12
s3["url_count"] = 20
s3["has_suspicious_url"] = 1
s3["max_stream_entropy"] = 7.8
s3["avg_stream_entropy"] = 7.2
s3["high_entropy_stream_count"] = 5
run(s3, "SCENARIO 3: Malicious (all indicators)")
