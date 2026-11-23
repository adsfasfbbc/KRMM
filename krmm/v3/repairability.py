"""Decomposed pre-update repairability assessment and baselines (M4).

Corrected probability factorization:
    P(strict_success | x) = P(candidate_correct | x) * P(strict_success | candidate_correct=1, x)

Provides three independent predictors:
    - DirectSuccessPredictor: P(strict_success | x)
    - CandidateReliabilityPredictor: P(candidate_correct | x)
    - ConditionalWriteFeasibilityPredictor: P(strict_success | candidate_correct=1, x)
and DecomposedPredictor combining CandidateReliabilityPredictor and ConditionalWriteFeasibilityPredictor.

Calibration statuses are strictly distinguished:
    - "untrained"
    - "fitted_uncalibrated"
    - "calibrated"  (ONLY after independent calibration split fitting)
    - "insufficient_data"
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence
import numpy as np
from sklearn.linear_model import LogisticRegression

from .schemas import CandidateReport, DamageReport, RepairabilityReport

VALID_CALIBRATION_STATUSES = frozenset(
    {"untrained", "fitted_uncalibrated", "calibrated", "insufficient_data"}
)

FORBIDDEN_LEAKAGE_KEYS = frozenset(
    {
        "gold_answer",
        "gold_evaluator_only",
        "wrong_target",
        "clean_state",
        "damage_delta",
        "hidden_test_labels",
        "strict_restoration",
        "repairs",
    }
)

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


def _check_no_forbidden_keys(mapping: Optional[Dict[str, object]]) -> None:
    if not isinstance(mapping, dict):
        return
    leaked = FORBIDDEN_LEAKAGE_KEYS.intersection(mapping.keys())
    if leaked:
        raise ValueError(f"Forbidden evaluator/post-repair leakage keys detected in diagnostic input: {sorted(leaked)}")


def extract_diagnostic_features(
    cand: CandidateReport,
    dmg: Optional[DamageReport],
    aux: Dict[str, object],
) -> List[float]:
    _check_no_forbidden_keys(aux)
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


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


class _BinaryProbabilityModel:
    """Base binary logistic predictor with strict small-sample and calibration-status guards."""

    def __init__(
        self,
        min_samples: int = 6,
        min_per_class: int = 2,
        c_reg: float = 1.0,
        seed: int = 107,
    ) -> None:
        self.min_samples = int(min_samples)
        self.min_per_class = int(min_per_class)
        self.c_reg = float(c_reg)
        self.seed = int(seed)
        self.model: Optional[LogisticRegression] = None
        self.calibrator: Optional[LogisticRegression] = None
        self.calibration_status: str = "untrained"
        self.fit_stats: Dict[str, object] = {
            "n_total": 0,
            "n_pos": 0,
            "n_neg": 0,
            "pos_rate": None,
            "reason": "untrained",
        }

    def _fit_binary(self, X: np.ndarray, y: Sequence[int]) -> str:
        X_arr = np.asarray(X, dtype=np.float64)
        y_list = [int(v) for v in y]
        n = len(y_list)
        n_pos = sum(1 for v in y_list if v == 1)
        n_neg = n - n_pos
        pos_rate = float(n_pos / n) if n > 0 else None

        self.model = None
        self.calibrator = None
        self.fit_stats = {
            "n_total": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_rate": pos_rate,
            "min_samples": self.min_samples,
            "min_per_class": self.min_per_class,
        }

        if n < self.min_samples or len(set(y_list)) < 2 or min(n_pos, n_neg) < self.min_per_class:
            self.calibration_status = "insufficient_data"
            self.fit_stats["reason"] = (
                "single_class_or_insufficient_minority"
                if n >= self.min_samples
                else "insufficient_total_samples"
            )
            return self.calibration_status

        self.model = LogisticRegression(
            C=self.c_reg,
            random_state=self.seed,
            max_iter=1000,
        ).fit(X_arr, y_list)
        # Training alone NEVER yields "calibrated".
        self.calibration_status = "fitted_uncalibrated"
        raw_probs = self.model.predict_proba(X_arr)[:, list(self.model.classes_).index(1)]
        self.fit_stats["reason"] = "fitted"
        self.fit_stats["train_prob_min"] = float(np.min(raw_probs))
        self.fit_stats["train_prob_max"] = float(np.max(raw_probs))
        self.fit_stats["near_separation_warning"] = bool(
            min(n_pos, n_neg) <= 3 or np.mean((raw_probs > 0.98) | (raw_probs < 0.02)) > 0.5
        )
        return self.calibration_status

    def calibrate(self, X_cal: np.ndarray, y_cal: Sequence[int]) -> str:
        """Independently calibrate fitted probabilities via 1D Platt scaling on disjoint data."""
        if self.model is None or self.calibration_status not in {"fitted_uncalibrated", "calibrated"}:
            return self.calibration_status
        X_arr = np.asarray(X_cal, dtype=np.float64)
        y_list = [int(v) for v in y_cal]
        n = len(y_list)
        n_pos = sum(1 for v in y_list if v == 1)
        n_neg = n - n_pos
        if n < self.min_samples or len(set(y_list)) < 2 or min(n_pos, n_neg) < self.min_per_class:
            self.calibrator = None
            self.calibration_status = "fitted_uncalibrated"
            self.fit_stats["calibration_note"] = "insufficient_calibration_split_data"
            return self.calibration_status
        raw_p = self.model.predict_proba(X_arr)[:, list(self.model.classes_).index(1)]
        logits = _logit(raw_p).reshape(-1, 1)
        self.calibrator = LogisticRegression(C=1.0, random_state=self.seed, max_iter=1000).fit(logits, y_list)
        self.calibration_status = "calibrated"
        self.fit_stats["calibration_n"] = n
        self.fit_stats["calibration_pos"] = n_pos
        self.fit_stats["calibration_neg"] = n_neg
        return self.calibration_status

    def predict_proba_one(self, feats: Sequence[float]) -> Optional[float]:
        if self.model is None or self.calibration_status not in {"fitted_uncalibrated", "calibrated"}:
            return None
        x = np.asarray([feats], dtype=np.float64)
        raw_p = float(self.model.predict_proba(x)[0][list(self.model.classes_).index(1)])
        if self.calibrator is not None and self.calibration_status == "calibrated":
            cal_p = float(
                self.calibrator.predict_proba(_logit(np.asarray([raw_p])).reshape(-1, 1))[0][
                    list(self.calibrator.classes_).index(1)
                ]
            )
            return cal_p
        return raw_p


class DirectSuccessPredictor(_BinaryProbabilityModel):
    """Mode A: Directly fits P(strict_success | x)."""

    def fit(self, X: np.ndarray, y_strict_success: Sequence[int]) -> str:
        return self._fit_binary(X, y_strict_success)


class CandidateReliabilityPredictor(_BinaryProbabilityModel):
    """Mode B: Fits P(candidate_correct | x)."""

    def fit(self, X: np.ndarray, y_cand_correct: Sequence[int]) -> str:
        return self._fit_binary(X, y_cand_correct)


class ConditionalWriteFeasibilityPredictor(_BinaryProbabilityModel):
    """Mode C: Fits P(strict_success | candidate_correct=1, x) or oracle write feasibility."""

    def __init__(
        self,
        min_samples: int = 6,
        min_per_class: int = 2,
        c_reg: float = 1.0,
        seed: int = 107,
        label_source: str = "correct_candidates",
    ) -> None:
        super().__init__(min_samples=min_samples, min_per_class=min_per_class, c_reg=c_reg, seed=seed)
        if label_source not in {"correct_candidates", "oracle_writeback"}:
            raise ValueError(f"Unsupported label_source: {label_source}")
        self.label_source = label_source

    def fit(
        self,
        X: np.ndarray,
        y_strict_success: Sequence[int],
        y_cand_correct: Optional[Sequence[int]] = None,
    ) -> str:
        X_arr = np.asarray(X, dtype=np.float64)
        y_succ = [int(v) for v in y_strict_success]
        if self.label_source == "correct_candidates":
            if y_cand_correct is None:
                raise ValueError("y_cand_correct is required when label_source='correct_candidates'")
            yc = [int(v) for v in y_cand_correct]
            keep = [i for i, c in enumerate(yc) if c == 1]
            X_sub = X_arr[keep] if keep else np.empty((0, X_arr.shape[1] if X_arr.ndim == 2 else 0))
            y_sub = [y_succ[i] for i in keep]
        else:
            X_sub = X_arr
            y_sub = y_succ
        status = self._fit_binary(X_sub, y_sub)
        self.fit_stats["label_source"] = self.label_source
        return status

    def calibrate_conditional(
        self,
        X_cal: np.ndarray,
        y_strict_cal: Sequence[int],
        y_cand_cal: Optional[Sequence[int]] = None,
    ) -> str:
        X_arr = np.asarray(X_cal, dtype=np.float64)
        y_succ = [int(v) for v in y_strict_cal]
        if self.label_source == "correct_candidates":
            if y_cand_cal is None:
                raise ValueError("y_cand_cal is required when label_source='correct_candidates'")
            keep = [i for i, c in enumerate(y_cand_cal) if int(c) == 1]
            X_sub = X_arr[keep] if keep else np.empty((0, X_arr.shape[1] if X_arr.ndim == 2 else 0))
            y_sub = [y_succ[i] for i in keep]
            return self.calibrate(X_sub, y_sub)
        return self.calibrate(X_arr, y_succ)


class DecomposedPredictor:
    """Correctly factorized predictor: P(success|x) = P(cand_correct|x) * P(success|cand_correct=1, x)."""

    def __init__(
        self,
        min_samples: int = 6,
        min_per_class: int = 2,
        write_label_source: str = "correct_candidates",
        seed: int = 107,
    ) -> None:
        self.rel_predictor = CandidateReliabilityPredictor(
            min_samples=min_samples, min_per_class=min_per_class, seed=seed
        )
        self.write_predictor = ConditionalWriteFeasibilityPredictor(
            min_samples=min_samples,
            min_per_class=min_per_class,
            seed=seed,
            label_source=write_label_source,
        )
        self.write_label_source = write_label_source
        self.calibration_status: str = "untrained"

    @property
    def rel_model(self):
        return self.rel_predictor.model

    @property
    def write_model(self):
        return self.write_predictor.model

    def _sync_status(self) -> str:
        s_rel = self.rel_predictor.calibration_status
        s_wr = self.write_predictor.calibration_status
        if s_rel == "untrained" and s_wr == "untrained":
            self.calibration_status = "untrained"
        elif s_rel == "insufficient_data" or s_wr == "insufficient_data":
            self.calibration_status = "insufficient_data"
        elif s_rel == "calibrated" and s_wr == "calibrated":
            self.calibration_status = "calibrated"
        else:
            self.calibration_status = "fitted_uncalibrated"
        return self.calibration_status

    def fit(
        self,
        X: np.ndarray,
        y_cand_correct: Sequence[int],
        y_strict_success: Sequence[int],
        X_oracle: Optional[np.ndarray] = None,
        y_oracle_write_ok: Optional[Sequence[int]] = None,
    ) -> str:
        self.rel_predictor.fit(X, y_cand_correct)
        if self.write_label_source == "oracle_writeback":
            if y_oracle_write_ok is None:
                raise ValueError("y_oracle_write_ok is required when write_label_source='oracle_writeback'")
            self.write_predictor.fit(X_oracle if X_oracle is not None else X, y_oracle_write_ok)
        else:
            self.write_predictor.fit(X, y_strict_success, y_cand_correct=y_cand_correct)
        return self._sync_status()

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cand_cal: Sequence[int],
        y_strict_cal: Sequence[int],
        X_oracle_cal: Optional[np.ndarray] = None,
        y_oracle_cal: Optional[Sequence[int]] = None,
    ) -> str:
        self.rel_predictor.calibrate(X_cal, y_cand_cal)
        if self.write_label_source == "oracle_writeback":
            if y_oracle_cal is not None:
                self.write_predictor.calibrate_conditional(
                    X_oracle_cal if X_oracle_cal is not None else X_cal, y_oracle_cal
                )
        else:
            self.write_predictor.calibrate_conditional(X_cal, y_strict_cal, y_cand_cal=y_cand_cal)
        return self._sync_status()

    def predict(self, feats: Sequence[float]) -> Dict[str, Optional[float]]:
        p_rel = self.rel_predictor.predict_proba_one(feats)
        p_write = self.write_predictor.predict_proba_one(feats)
        p_succ = float(p_rel * p_write) if (p_rel is not None and p_write is not None) else None
        return {
            "candidate_reliability": p_rel,
            "write_feasibility": p_write,
            "repair_success": p_succ,
            "rel_status": self.rel_predictor.calibration_status,
            "write_status": self.write_predictor.calibration_status,
            "calibration_status": self.calibration_status,
        }


def assess_repairability(
    cand: CandidateReport,
    dmg: Optional[DamageReport],
    aux: Dict[str, object],
    budget: Dict[str, object],
    predictor: Optional[DecomposedPredictor] = None,
) -> RepairabilityReport:
    """Assess decomposed candidate reliability and write risk before updating parameters."""
    _check_no_forbidden_keys(aux)
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
            calibration_status=predictor.calibration_status if predictor else "untrained",
            limitations=limitations + [str(cand.abstain_reason or "no_valid_candidate")],
        )

    if predictor is not None and predictor.calibration_status in {"calibrated", "fitted_uncalibrated"}:
        preds = predictor.predict(feats)
        p_rel = preds["candidate_reliability"]
        p_write = preds["write_feasibility"]
        p_succ = preds["repair_success"]
        write_risk = 1.0 - float(p_write) if p_write is not None else None
        cal_status = predictor.calibration_status
        if cal_status == "fitted_uncalibrated":
            limitations.append("Predictor probabilities are fitted_uncalibrated (not independently calibrated).")
    else:
        p_rel = predictor.rel_predictor.predict_proba_one(feats) if predictor is not None else None
        p_succ = None
        write_risk = None
        cal_status = predictor.calibration_status if predictor else "untrained"
        limitations.append(f"Predictor status is {cal_status}; falling back to structural rule gate.")

    min_prob = float(budget.get("min_repair_prob", 0.5))
    if cal_status in {"calibrated", "fitted_uncalibrated"} and p_succ is not None:
        if p_succ >= min_prob:
            action = "attempt_repair"
        elif (p_rel or 0.0) >= 0.75 and (write_risk or 1.0) > 0.5:
            action = "nonparametric_recovery"
        else:
            action = "abstain"
    else:
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
