"""Verify strict separation of untrained, fitted_uncalibrated, calibrated, and insufficient_data."""
from __future__ import annotations

import numpy as np
from krmm.v3.repairability import DecomposedPredictor, DirectSuccessPredictor, VALID_CALIBRATION_STATUSES


def test_calibration_status_lifecycle():
    rng = np.random.default_rng(123)
    X_tr = rng.normal(size=(24, 11))
    y_tr = [1] * 14 + [0] * 10
    X_cal = rng.normal(size=(16, 11))
    y_cal = [1] * 9 + [0] * 7

    m = DirectSuccessPredictor()
    assert m.calibration_status == "untrained"
    assert m.predict_proba_one(X_tr[0]) is None

    st_fit = m.fit(X_tr, y_tr)
    assert st_fit == "fitted_uncalibrated"
    assert m.calibration_status == "fitted_uncalibrated"
    assert m.calibration_status != "calibrated"

    # Calibration with single-class cal split must NOT claim calibrated
    st_bad_cal = m.calibrate(X_cal, [1] * len(X_cal))
    assert st_bad_cal == "fitted_uncalibrated"

    # Independent calibration split with both classes transitions to calibrated
    st_good_cal = m.calibrate(X_cal, y_cal)
    assert st_good_cal == "calibrated"
    assert m.calibration_status in VALID_CALIBRATION_STATUSES
    p = m.predict_proba_one(X_cal[0])
    assert p is not None and 0.0 <= p <= 1.0


def test_decomposed_never_claims_calibrated_on_fit():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(30, 11))
    yc = [1] * 20 + [0] * 10
    ys = [1] * 15 + [0] * 5 + [0] * 10
    dp = DecomposedPredictor()
    assert dp.calibration_status == "untrained"
    assert dp.fit(X, yc, ys) == "fitted_uncalibrated"
    assert dp.calibration_status == "fitted_uncalibrated"
