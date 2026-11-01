import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import json, joblib, numpy as np
from pdf_features import extract_features
from collections import Counter

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

def pick_best(results):
    label_votes    = Counter(d["label"] for d in results.values())
    majority_label = label_votes.most_common(1)[0][0]
    majority_pool  = {n: d for n, d in results.items() if d["label"] == majority_label}
    return max(majority_pool, key=lambda n: abs(majority_pool[n]["prob"] - 0.5))

test_files = [
    ("data/test_malicious_sim.pdf", "SIMULATED MALICIOUS"),
    ("data/test_benign.pdf",        "CLEAN BENIGN"),
]

for pdf_path, label in test_files:
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  File: {pdf_path}")
    print(f"{'='*55}")
    feat, val = extract_features(pdf_path)
    X = np.array([[feat.get(c, 0) for c in feat_cols]], dtype=float)
    results = {}
    for name, pipe in models.items():
        prob = float(pipe.predict_proba(X)[0][1])
        lbl  = "Malicious" if prob >= 0.6 else "Suspicious" if prob >= 0.3 else "Benign"
        results[name] = {"prob": prob, "label": lbl}
    best = pick_best(results)
    for name, d in results.items():
        marker = "  <-- SELECTED" if name == best else ""
        print(f"  {name:<25} {d['prob']:>5.0%}  {d['label']}{marker}")
    print(f"\n  FINAL VERDICT : {results[best]['label']} ({results[best]['prob']:.0%})")
    print(f"  SELECTED MODEL: {best}")
