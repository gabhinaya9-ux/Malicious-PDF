import json
with open("models/training_meta.json") as f:
    m = json.load(f)
print(f"Best model : {m['best_model_name']}")
print(f"Features   : {len(m['feature_names'])}")
print(f"\nAll models (sorted by F1):")
for name, met in sorted(m["metrics"].items(), key=lambda x: -x[1]["f1"]):
    print(f"  {name:<25}  F1={met['f1']:.4f}  AUC={met['roc_auc']:.4f}  Acc={met['accuracy']:.4f}")
print(f"\nModel paths saved:")
for name, path in m.get("model_paths", {}).items():
    print(f"  {name:<25}  {path}")
