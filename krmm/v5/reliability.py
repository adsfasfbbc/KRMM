"""Factorized repairability & candidate reliability estimator for v0.5 (Section 6.3)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np

from krmm.v3.repairability import (
    CandidateReliabilityPredictor,
    ConditionalWriteFeasibilityPredictor,
    DecomposedPredictor,
    DirectSuccessPredictor,
)


class V5ReliabilityEstimator:
    """Wraps correctly factorized P(strict|x) = P(cand|x) * P(strict|cand=1, x) with strict sparsity guard."""

    def __init__(self, min_samples: int = 8, min_write_failures: int = 5, seed: int = 107) -> None:
        self.min_samples = int(min_samples)
        self.min_write_failures = int(min_write_failures)
        self.cand_model = CandidateReliabilityPredictor(min_samples=min_samples, min_per_class=2, seed=seed)
        self.write_model = ConditionalWriteFeasibilityPredictor(
            min_samples=min_samples, min_per_class=min_write_failures, seed=seed, label_source="correct_candidates"
        )
        self.direct_model = DirectSuccessPredictor(min_samples=min_samples, min_per_class=2, seed=seed)

    def fit(self, X: np.ndarray, y_cand_correct: Sequence[int], y_strict_success: Sequence[int]) -> Dict[str, str]:
        s_cand = self.cand_model.fit(X, y_cand_correct)
        s_write = self.write_model.fit(X, y_strict_success, y_cand_correct=y_cand_correct)
        s_dir = self.direct_model.fit(X, y_strict_success)
        return {
            "candidate_reliability_status": s_cand,
            "conditional_write_status": s_write,
            "direct_success_status": s_dir,
            "mode_active": (
                "two_stage_decomposed"
                if s_cand in {"fitted_uncalibrated", "calibrated"} and s_write in {"fitted_uncalibrated", "calibrated"}
                else "candidate_reliability_only"
            ),
        }

    def predict(self, feats: Sequence[float]) -> Dict[str, Optional[float]]:
        p_c = self.cand_model.predict_proba_one(feats)
        p_w = self.write_model.predict_proba_one(feats)
        p_d = self.direct_model.predict_proba_one(feats)
        return {
            "p_candidate_correct": p_c,
            "p_write_given_correct": p_w,
            "p_strict_decomposed": float(p_c * p_w) if (p_c is not None and p_w is not None) else None,
            "p_strict_direct": p_d,
            "conditional_write_status": self.write_model.calibration_status,
        }
