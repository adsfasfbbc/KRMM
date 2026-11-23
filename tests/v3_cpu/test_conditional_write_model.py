"""Verify ConditionalWriteFeasibilityPredictor modes (correct_candidates vs oracle_writeback)."""
from __future__ import annotations

import numpy as np
import pytest
from krmm.v3.repairability import ConditionalWriteFeasibilityPredictor, DirectSuccessPredictor, CandidateReliabilityPredictor


def test_three_independent_predictors_callable():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 11))
    yc = [1] * 18 + [0] * 12
    ys = [1] * 12 + [0] * 6 + [0] * 12

    m_dir = DirectSuccessPredictor()
    m_rel = CandidateReliabilityPredictor()
    m_wr = ConditionalWriteFeasibilityPredictor(label_source="correct_candidates")

    assert m_dir.fit(X, ys) == "fitted_uncalibrated"
    assert m_rel.fit(X, yc) == "fitted_uncalibrated"
    assert m_wr.fit(X, ys, y_cand_correct=yc) == "fitted_uncalibrated"
    assert m_wr.fit_stats["n_total"] == 18
    assert m_wr.fit_stats["n_pos"] == 12
    assert m_wr.fit_stats["n_neg"] == 6

    p_dir = m_dir.predict_proba_one(X[0])
    p_rel = m_rel.predict_proba_one(X[0])
    p_wr = m_wr.predict_proba_one(X[0])
    assert all(0.0 <= p <= 1.0 for p in (p_dir, p_rel, p_wr))


def test_oracle_writeback_label_source():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(20, 11))
    y_oracle = [1] * 15 + [0] * 5
    m_wr = ConditionalWriteFeasibilityPredictor(label_source="oracle_writeback")
    assert m_wr.fit(X, y_oracle) == "fitted_uncalibrated"
    assert m_wr.fit_stats["label_source"] == "oracle_writeback"
    assert m_wr.fit_stats["n_total"] == 20


def test_invalid_label_source_rejected():
    with pytest.raises(ValueError):
        ConditionalWriteFeasibilityPredictor(label_source="invalid_mode")
