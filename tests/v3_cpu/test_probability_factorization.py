"""Verify exact conditional probability factorization P(success|x) = P(cand|x) * P(success|cand=1, x)."""
from __future__ import annotations

import numpy as np
from krmm.v3.repairability import DecomposedPredictor


class _ConstModel:
    def __init__(self, p: float) -> None:
        self.p = float(p)
        self.classes_ = [0, 1]

    def predict_proba(self, X):
        return np.asarray([[1.0 - self.p, self.p] for _ in range(len(X))], dtype=np.float64)


def test_exact_conditional_factorization():
    pred = DecomposedPredictor()
    pred.rel_predictor.model = _ConstModel(0.5)
    pred.rel_predictor.calibration_status = "fitted_uncalibrated"
    pred.write_predictor.model = _ConstModel(0.8)
    pred.write_predictor.calibration_status = "fitted_uncalibrated"
    pred._sync_status()

    out = pred.predict([0.0] * 11)
    assert abs(out["candidate_reliability"] - 0.5) < 1e-9
    assert abs(out["write_feasibility"] - 0.8) < 1e-9
    assert abs(out["repair_success"] - 0.4) < 1e-9
    # Must NOT be 0.5 * (0.5 * 0.8) = 0.2 (the legacy double-counting bug)
    assert abs(out["repair_success"] - 0.2) > 0.1


def test_conditional_training_excludes_wrong_candidates():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 11))
    # First 20 have candidate_correct=0 (and strict_success=0);
    # Next 20 have candidate_correct=1 with 14 successes and 6 failures (0.70 rate)
    yc = [0] * 20 + [1] * 20
    ys = [0] * 20 + [1] * 14 + [0] * 6

    pred = DecomposedPredictor(min_samples=6, min_per_class=2)
    status = pred.fit(X, yc, ys)
    assert status == "fitted_uncalibrated"
    assert pred.write_predictor.fit_stats["n_total"] == 20
    assert pred.write_predictor.fit_stats["n_pos"] == 14
    assert pred.write_predictor.fit_stats["n_neg"] == 6
    assert abs(pred.write_predictor.fit_stats["pos_rate"] - 0.70) < 1e-9
