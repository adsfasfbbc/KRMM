"""Evidence-structure candidate reliability predictor P(candidate_correct | evidence_features) (Section 5.5)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np
from sklearn.linear_model import LogisticRegression

from .schemas import EvidenceAssessment, assert_method_boundary

EVIDENCE_RELIABILITY_FEATURES = [
    "n_minimal_sets",
    "dependency_overlap",
    "distinct_path_flag",
    "independent_source_flag",
    "same_source_redundant_flag",
    "single_chain_flag",
    "conflict_flag",
    "direct_confidence",
    "candidate_change",
]


def extract_evidence_features(
    assessment: EvidenceAssessment,
    direct_confidence: float,
    candidate_change: int,
    extra_meta: Optional[Dict[str, object]] = None,
) -> List[float]:
    assert_method_boundary(extra_meta)
    cat = assessment.evidence_category
    return [
        float(assessment.number_of_support_paths),
        float(assessment.dependency_overlap),
        float(cat in {"redundant_distinct_paths", "independent_source_support"}),
        float(cat == "independent_source_support"),
        float(cat == "redundant_same_source"),
        float(cat == "single_chain"),
        float(assessment.conflict_status),
        float(direct_confidence),
        float(candidate_change),
    ]


class EvidenceReliabilityModel:
    """Regularized logistic regression for P(candidate_correct | evidence_features) with strict status guards."""

    def __init__(self, min_samples: int = 8, min_per_class: int = 2, c_reg: float = 1.0, seed: int = 107) -> None:
        self.min_samples = int(min_samples)
        self.min_per_class = int(min_per_class)
        self.c_reg = float(c_reg)
        self.seed = int(seed)
        self.model: Optional[LogisticRegression] = None
        self.calibrator: Optional[LogisticRegression] = None
        self.calibration_status: str = "untrained"
        self.fit_stats: Dict[str, object] = {}

    def fit(self, X: np.ndarray, y_cand_correct: Sequence[int]) -> str:
        X_arr = np.asarray(X, dtype=np.float64)
        y = [int(v) for v in y_cand_correct]
        n = len(y)
        n_pos = sum(y)
        n_neg = n - n_pos
        self.fit_stats = {"n_total": n, "n_pos": n_pos, "n_neg": n_neg}
        if n < self.min_samples or len(set(y)) < 2 or min(n_pos, n_neg) < self.min_per_class:
            self.model = None
            self.calibrator = None
            self.calibration_status = "insufficient_data"
            return self.calibration_status
        self.model = LogisticRegression(C=self.c_reg, random_state=self.seed, max_iter=1000).fit(X_arr, y)
        self.calibrator = None
        self.calibration_status = "fitted_uncalibrated"
        return self.calibration_status

    def calibrate(self, X_cal: np.ndarray, y_cal: Sequence[int]) -> str:
        if self.model is None or self.calibration_status not in {"fitted_uncalibrated", "calibrated"}:
            return self.calibration_status
        X_arr = np.asarray(X_cal, dtype=np.float64)
        y = [int(v) for v in y_cal]
        n = len(y)
        n_pos = sum(y)
        n_neg = n - n_pos
        if n < self.min_samples or len(set(y)) < 2 or min(n_pos, n_neg) < self.min_per_class:
            self.calibrator = None
            self.calibration_status = "fitted_uncalibrated"
            return self.calibration_status
        raw_p = self.model.predict_proba(X_arr)[:, list(self.model.classes_).index(1)]
        clipped = np.clip(raw_p, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        self.calibrator = LogisticRegression(C=1.0, random_state=self.seed, max_iter=1000).fit(logits, y)
        self.calibration_status = "calibrated"
        return self.calibration_status

    def predict_proba_one(self, feats: Sequence[float]) -> Optional[float]:
        if self.model is None or self.calibration_status not in {"fitted_uncalibrated", "calibrated"}:
            return None
        x = np.asarray([feats], dtype=np.float64)
        p = float(self.model.predict_proba(x)[0][list(self.model.classes_).index(1)])
        if self.calibrator is not None and self.calibration_status == "calibrated":
            c = np.clip(p, 1e-6, 1.0 - 1e-6)
            logit = np.asarray([[np.log(c / (1.0 - c))]], dtype=np.float64)
            p = float(self.calibrator.predict_proba(logit)[0][list(self.calibrator.classes_).index(1)])
        return p
