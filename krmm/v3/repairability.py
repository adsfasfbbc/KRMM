"""Decomposed pre-update repairability assessment and baselines (M4)."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from .schemas import CandidateReport, DamageReport, RepairabilityReport

DIAGNOSTIC_FEATURES = [
    "support_completeness",
    "ambiguity",
    "conflict",
    "key_sensitivity",
    "sham_stability",
    "provenance_ratio",
    "direct_confidence",
    "direct_entropy",
    "self_consistency",
    "candidate_change",
    "anomaly_score",
]


def extract_diagnostic_features(
    cand: CandidateReport,
    dmg: Optional[DamageReport],
    aux: Dict[str, object],
) -> List[float]:
    direct_probs = aux.get("direct_probs", [[0.25, 0.25, 0.25, 0.25]])
    p0 = np.clip(np.asarray(direct_probs[0], dtype=np.float64), 1e-12, 1.0)
    direct_conf = float(np.max(p0))
    direct_ent = float(-np.sum(p0 * np.log(p0)))
    preds = ["ABCD"[int(np.argmax(p))] for p in direct_probs]
    consistency = float(sum(1 for pr in preds if pr == preds[0]) / max(1, len(preds)))
    n_facts = max(1, len(cand.atomic_facts))
    n_groups = len(aux.get("unique_provenance_groups", []))
    prov_ratio = float(n_groups / n_facts)
    support_comp = float(cand.candidate_answer is not None and not cand.conflict)
    key_sens = float(cand.key_intervention_result.get("sensitive", False))
    sham_stab = float(cand.sham_intervention_result.get("stable", True))
    cand_change = float(aux.get("candidate_change_vs_direct", 0))
    anom = float(dmg.anomaly_score) if dmg is not None else 0.5
    return [
        support_comp,
        float(cand.ambiguity),
        float(cand.conflict),
        key_sens,
        sham_stab,
        prov_ratio,
        direct_conf,
        direct_ent,
        consistency,
        cand_change,
        anom,
    ]


def baseline_score(
    name: str,
    feats: Sequence[float],
    cand: CandidateReport,
    seed: int = 107,
) -> float:
    """Compute baseline priority scores under fixed candidate & budget (M4.4)."""
    idx = {k: i for i, k in enumerate(DIAGNOSTIC_FEATURES)}
    if name == "random":
        rng = random.Random(f"{seed}::{cand.unit_id}")
        return rng.random()
    if name == "direct_confidence":
        return float(feats[idx["direct_confidence"]])
    if name == "entropy":
        return float(feats[idx["direct_entropy"]])
    if name == "self_consistency":
        return float(1.0 - feats[idx["self_consistency"]])
    if name == "candidate_change":
        return float(feats[idx["candidate_change"]])
    if name == "support_completeness":
        return float(feats[idx["support_completeness"]] * (1.0 - feats[idx["ambiguity"]]))
    raise ValueError(f"Unknown baseline: {name}")


class DecomposedPredictor:
    """Separate models for candidate reliability and write feasibility, calibrated only on independent data."""

    def __init__(self) -> None:
        self.rel_model = None
        self.write_model = None
        self.calibration_status = "uncalibrated"

    def fit(
        self,
        X: np.ndarray,
        y_cand_correct: Sequence[int],
        y_write_ok: Sequence[int],
    ) -> str:
        yc = list(y_cand_correct)
        yw = list(y_write_ok)
        if len(X) < 6 or len(set(yc)) < 2 or len(set(yw)) < 2:
            self.calibration_status = "insufficient_data"
            self.rel_model = DummyClassifier(strategy="prior").fit(X, yc) if len(X) > 0 else None
            self.write_model = DummyClassifier(strategy="prior").fit(X, yw) if len(X) > 0 else None
            return self.calibration_status

        self.rel_model = LogisticRegression(C=1.0, random_state=107, max_iter=1000).fit(X, yc)
        self.write_model = LogisticRegression(C=1.0, random_state=107, max_iter=1000).fit(X, yw)
        self.calibration_status = "calibrated"
        return self.calibration_status

    def predict(self, feats: Sequence[float]) -> Dict[str, Optional[float]]:
        if self.calibration_status != "calibrated" or self.rel_model is None or self.write_model is None:
            return {"candidate_reliability": None, "write_feasibility": None, "repair_success": None}
        x = np.asarray([feats], dtype=np.float64)
        pr = float(self.rel_model.predict_proba(x)[0][list(self.rel_model.classes_).index(1)])
        pw = float(self.write_model.predict_proba(x)[0][list(self.write_model.classes_).index(1)])
        return {
            "candidate_reliability": pr,
            "write_feasibility": pw,
            "repair_success": float(pr * pw),
        }


def assess_repairability(
    cand: CandidateReport,
    dmg: Optional[DamageReport],
    aux: Dict[str, object],
    budget: Dict[str, object],
    predictor: Optional[DecomposedPredictor] = None,
) -> RepairabilityReport:
    """Assess decomposed candidate reliability and write risk before updating parameters."""
    feats = extract_diagnostic_features(cand, dmg, aux)
    limitations: List[str] = []

    if aux.get("same_model_correlated_only"):
        limitations.append("All extracted facts originate from the same damaged model.")

    if cand.candidate_answer is None or cand.conflict:
        return RepairabilityReport(
            unit_id=cand.unit_id,
            candidate_reliability=0.0,
            predicted_repair_success=0.0,
            estimated_write_risk=None,
            recommended_action="request_evidence" if not cand.conflict else "abstain",
            calibration_status=predictor.calibration_status if predictor else "uncalibrated",
            limitations=limitations + [str(cand.abstain_reason or "no_valid_candidate")],
        )

    if predictor is not None and predictor.calibration_status == "calibrated":
        preds = predictor.predict(feats)
        p_rel = preds["candidate_reliability"]
        p_write = preds["write_feasibility"]
        p_succ = preds["repair_success"]
        write_risk = 1.0 - float(p_write) if p_write is not None else None
        cal_status = "calibrated"
    else:
        # Rule-based uncalibrated fallback: strictly marked uncalibrated!
        p_rel = None
        p_succ = None
        write_risk = None
        cal_status = predictor.calibration_status if predictor else "uncalibrated"
        limitations.append("Predictor is uncalibrated; falling back to rule gate.")

    min_prob = float(budget.get("min_repair_prob", 0.5))
    if cal_status == "calibrated" and p_succ is not None:
        if p_succ >= min_prob:
            action = "attempt_repair"
        elif (p_rel or 0.0) >= 0.75 and (write_risk or 1.0) > 0.5:
            action = "nonparametric_recovery"
        else:
            action = "abstain"
    else:
        # Simple structural gate when uncalibrated
        if aux.get("candidate_change_vs_direct", 0) == 1 and not cand.conflict:
            action = "attempt_repair"
        elif not cand.conflict and cand.key_intervention_result.get("sensitive", False):
            action = "attempt_repair"
        else:
            action = "abstain"

    return RepairabilityReport(
        unit_id=cand.unit_id,
        candidate_reliability=p_rel,
        predicted_repair_success=p_succ,
        estimated_write_risk=write_risk,
        recommended_action=action,
        calibration_status=cal_status,
        limitations=limitations,
    )
