"""
risk_scorer.py
--------------
Combines the ML model's malicious probability with a deterministic
rule-based analysis to produce a final 0–100 risk score and a
human-readable classification.

Score composition
-----------------
  ML component  : 70 % of total score  (model probability × 70)
  Rule component: 30 % of total score  (weighted rule triggers × 30)

Classification thresholds
--------------------------
  0–29   → Benign
  30–59  → Suspicious
  60–100 → Malicious
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------
# Each rule: (feature_key, threshold, weight, description)
# weight is the contribution to the rule sub-score (normalised to 0-1 later)
_RULES: List[Tuple[str, Any, float, str]] = [
    # High-impact rules (weight 1.0)
    ("has_javascript",          1,    1.0, "JavaScript code present"),
    ("has_openaction",          1,    1.0, "OpenAction trigger detected"),
    ("has_launch",              1,    1.0, "Launch action detected"),
    # Medium-impact rules (weight 0.7)
    ("has_aa",                  1,    0.7, "Additional-Action (AA) dictionary present"),
    ("embedded_file_count",     1,    0.7, "Embedded files detected"),
    ("has_richmedia",           1,    0.7, "RichMedia (Flash/3D) content detected"),
    ("has_xfa",                 1,    0.7, "XFA form structure detected"),
    ("has_objstm",              1,    0.6, "Object streams (potential obfuscation)"),
    # Lower-impact rules (weight 0.4)
    ("url_count",               3,    0.4, "Multiple URLs extracted"),
    ("has_suspicious_url",      1,    0.5, "Suspicious domain TLD in URL"),
    ("high_entropy_stream_count", 2,  0.5, "High-entropy streams (possible obfuscation)"),
    ("max_stream_entropy",      7.2,  0.4, "Maximum stream entropy ≥ 7.2 bits"),
    ("is_encrypted",            1,    0.3, "PDF is encrypted"),
    ("has_xrefstm",             1,    0.3, "XRef stream present"),
    ("javascript_count",        3,    0.5, "Multiple JavaScript occurrences"),
    ("action_count",            3,    0.4, "Multiple actions detected"),
]

_MAX_RULE_WEIGHT = sum(w for _, _, w, _ in _RULES)  # normaliser


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class RuleHit:
    feature: str
    value: Any
    threshold: Any
    weight: float
    description: str


@dataclass
class RiskResult:
    # Inputs
    file_path: str
    sha256: str
    # ML
    ml_probability: float          # 0.0–1.0  raw model output
    # Rules
    rule_hits: List[RuleHit] = field(default_factory=list)
    rule_score: float = 0.0        # 0–30
    # Combined
    risk_score: float = 0.0        # 0–100
    classification: str = "Unknown"
    confidence: str = "Low"
    # Feature vector used
    features: Dict[str, Any] = field(default_factory=dict)

    # ── Derived ────────────────────────────────────────────────────────
    @property
    def ml_score(self) -> float:
        """ML contribution (0–70)."""
        return round(self.ml_probability * 70, 2)

    def summary(self) -> str:
        lines = [
            f"File        : {self.file_path}",
            f"SHA-256     : {self.sha256}",
            f"ML prob     : {self.ml_probability:.1%}",
            f"ML score    : {self.ml_score:.1f} / 70",
            f"Rule score  : {self.rule_score:.1f} / 30",
            f"Risk score  : {self.risk_score:.0f} / 100",
            f"Class       : {self.classification}",
            f"Confidence  : {self.confidence}",
        ]
        if self.rule_hits:
            lines.append("Rule hits   :")
            for rh in self.rule_hits:
                lines.append(f"  [{rh.weight:.1f}] {rh.description} "
                             f"(value={rh.value}, threshold={rh.threshold})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------
def compute_risk(
    features: Dict[str, Any],
    ml_probability: float,
    file_path: str = "",
    sha256: str = "",
    ml_weight: float = 0.70,
    rule_weight: float = 0.30,
) -> RiskResult:
    """
    Compute the composite risk score.

    Parameters
    ----------
    features       : feature dict as returned by pdf_features.extract_features()
    ml_probability : probability of malicious class from the trained model
    file_path      : PDF path (for display purposes)
    sha256         : file hash (for display purposes)
    ml_weight      : fraction of score from ML (default 0.70)
    rule_weight    : fraction of score from rules (default 0.30)

    Returns
    -------
    RiskResult
    """
    # ── Rule evaluation ─────────────────────────────────────────────────
    hits: List[RuleHit] = []
    raw_rule_score = 0.0

    for feat_key, threshold, weight, description in _RULES:
        val = features.get(feat_key, 0)
        try:
            triggered = float(val) >= float(threshold)
        except (TypeError, ValueError):
            triggered = bool(val)

        if triggered:
            hits.append(RuleHit(
                feature=feat_key,
                value=val,
                threshold=threshold,
                weight=weight,
                description=description,
            ))
            raw_rule_score += weight

    # Normalise rule score to [0, rule_weight × 100]
    normalised_rule = (raw_rule_score / _MAX_RULE_WEIGHT) * (rule_weight * 100)

    # ── ML component ────────────────────────────────────────────────────
    ml_component = ml_probability * (ml_weight * 100)

    # ── Combined score ──────────────────────────────────────────────────
    risk_score = round(min(ml_component + normalised_rule, 100.0), 1)

    # ── Classification ──────────────────────────────────────────────────
    classification, confidence = _classify(risk_score, ml_probability, len(hits))

    return RiskResult(
        file_path=file_path,
        sha256=sha256,
        ml_probability=round(ml_probability, 4),
        rule_hits=hits,
        rule_score=round(normalised_rule, 2),
        risk_score=risk_score,
        classification=classification,
        confidence=confidence,
        features=features,
    )


def _classify(score: float, ml_prob: float, n_rule_hits: int) -> Tuple[str, str]:
    """
    Map score → (classification, confidence).
    Confidence is boosted when ML and rules agree.
    """
    if score >= 60:
        label = "Malicious"
    elif score >= 30:
        label = "Suspicious"
    else:
        label = "Benign"

    # Agreement between ML signal and rule signal
    rule_signal   = "high" if n_rule_hits >= 3 else ("medium" if n_rule_hits >= 1 else "low")
    ml_signal     = "high" if ml_prob >= 0.7 else ("medium" if ml_prob >= 0.3 else "low")
    both_high     = rule_signal == "high" and ml_signal == "high"
    both_low      = rule_signal == "low"  and ml_signal == "low"

    if both_high or both_low:
        confidence = "High"
    elif rule_signal == "low" or ml_signal == "low":
        confidence = "Low"
    else:
        confidence = "Medium"

    return label, confidence


# ---------------------------------------------------------------------------
# Convenience: score a PDF end-to-end (no need to import multiple modules)
# ---------------------------------------------------------------------------
def score_pdf(
    file_path: str | Path,
    model_path: str | Path = "models/best_model.pkl",
) -> RiskResult:
    """
    Full pipeline: validate → analyse → extract features → predict → score.
    Loads the trained model from disk each call (use load_model() for caching).
    """
    import joblib
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from pdf_features import extract_features, FEATURE_COLUMNS

    feat_dict, val = extract_features(file_path)

    if not val.is_valid:
        return RiskResult(
            file_path=str(file_path),
            sha256=val.sha256,
            ml_probability=0.0,
            risk_score=0.0,
            classification="Invalid",
            confidence="N/A",
            features=feat_dict,
        )

    # Build feature vector in canonical order
    feature_cols = FEATURE_COLUMNS
    X = np.array([[feat_dict.get(c, 0) for c in feature_cols]], dtype=float)

    model = joblib.load(model_path)
    ml_prob = float(model.predict_proba(X)[0][1])

    return compute_risk(
        features=feat_dict,
        ml_probability=ml_prob,
        file_path=str(file_path),
        sha256=val.sha256,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python risk_scorer.py <path_to_pdf> [model_path]")
        sys.exit(1)

    pdf_path   = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/best_model.pkl"

    result = score_pdf(pdf_path, model_path)
    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60)
