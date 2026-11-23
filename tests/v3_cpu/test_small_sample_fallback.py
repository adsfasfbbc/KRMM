"""Verify small-sample and single-class fallback returns insufficient_data without forced classifier."""
from __future__ import annotations

import numpy as np
from krmm.v3.repairability import ConditionalWriteFeasibilityPredictor, DecomposedPredictor, DirectSuccessPredictor


def test_single_class_returns_insufficient_data():
    X = np.zeros((20, 11))
    m = DirectSuccessPredictor()
    assert m.fit(X, [1] * 20) == "insufficient_data"
    assert m.model is None
    assert m.predict_proba_one(X[0]) is None


def test_one_failure_minority_returns_insufficient_data():
    X = np.zeros((20, 11))
    yc = [1] * 20
    # Only 1 write failure among 20 correct candidates -> insufficient_data (min_per_class=2)
    ys = [1] * 19 + [0]
    cw = ConditionalWriteFeasibilityPredictor(min_samples=6, min_per_class=2)
    assert cw.fit(X, ys, y_cand_correct=yc) == "insufficient_data"
    assert cw.model is None

    dp = DecomposedPredictor(min_samples=6, min_per_class=2)
    # Candidate reliability has 15 pos / 5 neg, but conditional write has 14 pos / 1 neg
    yc2 = [1] * 15 + [0] * 5
    ys2 = [1] * 14 + [0] * 1 + [0] * 5
    assert dp.fit(X, yc2, ys2) == "insufficient_data"
    out = dp.predict(X[0])
    assert out["candidate_reliability"] is not None
    assert out["write_feasibility"] is None
    assert out["repair_success"] is None


def test_too_few_samples_returns_insufficient_data():
    X = np.zeros((4, 11))
    m = DirectSuccessPredictor(min_samples=6)
    assert m.fit(X, [1, 1, 0, 0]) == "insufficient_data"
    assert m.model is None
