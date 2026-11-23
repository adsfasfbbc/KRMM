"""Legacy buggy DecomposedPredictor preserved strictly for offline diff auditing."""
from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


class LegacyBuggyDecomposedPredictor:
    """Reproduces the pre-audit double-counting bug: P(cand|x) * P(strict|x) without conditioning."""

    def __init__(self) -> None:
        self.rel_model = None
        self.write_model = None
        self.calibration_status = "untrained"

    def fit(self, X: np.ndarray, y_cand_correct: Sequence[int], y_strict_success: Sequence[int]) -> str:
        yc = list(y_cand_correct)
        yw = list(y_strict_success)
        if len(X) < 6 or len(set(yc)) < 2 or len(set(yw)) < 2:
            self.calibration_status = "insufficient_data"
            self.rel_model = DummyClassifier(strategy="prior").fit(X, yc) if len(X) > 0 else None
            self.write_model = DummyClassifier(strategy="prior").fit(X, yw) if len(X) > 0 else None
            return self.calibration_status
        self.rel_model = LogisticRegression(C=1.0, random_state=107, max_iter=1000).fit(X, yc)
        self.write_model = LogisticRegression(C=1.0, random_state=107, max_iter=1000).fit(X, yw)
        # Historical bug: labeled "calibrated" immediately after fitting
        self.calibration_status = "legacy_fitted_mislabeled_calibrated"
        return self.calibration_status

    def predict(self, feats: Sequence[float]) -> Dict[str, Optional[float]]:
        if self.rel_model is None or self.write_model is None or self.calibration_status == "insufficient_data":
            return {"candidate_reliability": None, "write_feasibility": None, "repair_success": None}
        x = np.asarray([feats], dtype=np.float64)
        pr = float(self.rel_model.predict_proba(x)[0][list(self.rel_model.classes_).index(1)])
        pw = float(self.write_model.predict_proba(x)[0][list(self.write_model.classes_).index(1)])
        return {"candidate_reliability": pr, "write_feasibility": pw, "repair_success": float(pr * pw)}
