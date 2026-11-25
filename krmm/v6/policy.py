"""Strictly isolated selective repair policies (NO oracle_gold parameter!) & risk-coverage evaluation."""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from .schemas import ActionType
from .truth_audit import deterministic_tie_key

METHOD_STRATEGIES: Tuple[str, ...] = (
    "B0_no_repair",
    "B1_generative",
    "B2_ungated_structured",
    "B3_confidence_gate",
    "B3_fact_count_gate",
    "B3_minset_gate",
    "B3_random_gate",
    "M_evidence_gated",
    "M_full",
)


def select_action(
    strategy: str,
    unit_id: str,
    direct_candidate: str,
    direct_confidence: float,
    generative_candidate: str,
    structured_candidate: Optional[str],
    support_class: str,
    n_valid_facts: int,
    n_minimal_sets: int,
    heuristic_support_score: float,
    remaining_write_budget: int,
    seed: int = 2026,
) -> Dict[str, object]:
    """Method-side action selection. Strictly NEVER accepts gold answers or evaluator labels."""
    if remaining_write_budget <= 0 or strategy == "B0_no_repair":
        return {"action": "ABSTAIN", "candidate": None, "score": 0.0, "reason": "no_repair_or_budget_exhausted"}

    if strategy == "B1_generative":
        return {
            "action": "ATTEMPT_REPAIR",
            "candidate": generative_candidate,
            "score": float(1.0 - direct_confidence),
            "reason": "generative_always_attempt",
        }

    if strategy == "B2_ungated_structured":
        if structured_candidate is None:
            return {"action": "ABSTAIN", "candidate": None, "score": 0.0, "reason": "no_structured_candidate"}
        return {
            "action": "ATTEMPT_REPAIR",
            "candidate": structured_candidate,
            "score": float(heuristic_support_score),
            "reason": "ungated_structured_attempt",
        }

    if strategy == "B3_confidence_gate":
        # Same structured candidate, gated only by direct-probe confidence threshold
        if structured_candidate is None:
            return {"action": "ABSTAIN", "candidate": None, "score": 0.0, "reason": "no_structured_candidate"}
        if direct_confidence >= 0.75:
            return {"action": "ATTEMPT_REPAIR", "candidate": structured_candidate, "score": float(direct_confidence), "reason": "confidence_gate_pass"}
        return {"action": "ABSTAIN", "candidate": None, "score": float(direct_confidence), "reason": "confidence_gate_abstain"}

    if strategy == "B3_fact_count_gate":
        if structured_candidate is None:
            return {"action": "ABSTAIN", "candidate": None, "score": 0.0, "reason": "no_structured_candidate"}
        if n_valid_facts >= 2:
            return {"action": "ATTEMPT_REPAIR", "candidate": structured_candidate, "score": float(n_valid_facts), "reason": "fact_count_ge_2"}
        return {"action": "REQUEST_EVIDENCE", "candidate": None, "score": float(n_valid_facts), "reason": "fact_count_lt_2"}

    if strategy == "B3_minset_gate":
        if structured_candidate is None:
            return {"action": "ABSTAIN", "candidate": None, "score": 0.0, "reason": "no_structured_candidate"}
        if n_minimal_sets >= 1:
            return {"action": "ATTEMPT_REPAIR", "candidate": structured_candidate, "score": float(n_minimal_sets), "reason": "minset_ge_1"}
        return {"action": "ABSTAIN", "candidate": None, "score": 0.0, "reason": "no_minimal_set"}

    if strategy == "B3_random_gate":
        if structured_candidate is None:
            return {"action": "ABSTAIN", "candidate": None, "score": 0.0, "reason": "no_structured_candidate"}
        h = int(hashlib.sha256(f"rand_gate:{seed}:{unit_id}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        if h >= 0.35:
            return {"action": "ATTEMPT_REPAIR", "candidate": structured_candidate, "score": float(h), "reason": "random_gate_accept"}
        return {"action": "ABSTAIN", "candidate": None, "score": float(h), "reason": "random_gate_abstain"}

    if strategy in ("M_evidence_gated", "M_full"):
        if structured_candidate is None:
            act: ActionType = "REQUEST_EVIDENCE" if support_class == "insufficient_support" else "ABSTAIN"
            return {"action": act, "candidate": None, "score": float(heuristic_support_score), "reason": f"gated_out_{support_class}"}
        if support_class in ("redundant_distinct_attributes", "single_decisive"):
            return {
                "action": "ATTEMPT_REPAIR",
                "candidate": structured_candidate,
                "score": float(heuristic_support_score),
                "reason": f"evidence_gate_accept_{support_class}",
            }
        act2: ActionType = "REQUEST_EVIDENCE" if support_class in ("single_chain", "insufficient_support") else "ABSTAIN"
        return {
            "action": act2,
            "candidate": None,
            "score": float(heuristic_support_score),
            "reason": f"evidence_gate_reject_{support_class}",
        }

    raise ValueError(f"Unknown method strategy: {strategy}")


def compute_risk_coverage_curve(
    items: Sequence[Dict[str, object]],
    score_key: str,
    error_key: str = "candidate_is_wrong",
    seed: int = 2026,
    grid: Sequence[float] = (0.25, 0.50, 0.75),
) -> Dict[str, object]:
    """Compute fixed-candidate risk-coverage curve, AURC, and fixed-coverage risk on reachable grid points."""
    n = len(items)
    if n == 0:
        return {"n": 0, "aurc": None, "fixed_coverage_risk": {str(g): None for g in grid}, "points": []}

    # Only items that actually have a candidate can be accepted; items with score=None are unreachable
    eligible = [it for it in items if it.get(score_key) is not None and it.get("candidate") is not None]
    order = sorted(
        eligible,
        key=lambda it: (-float(it[score_key]), deterministic_tie_key(str(it["unit_id"]), seed)),
    )
    max_cov = len(order) / n
    points: List[Dict[str, float]] = []
    cum_err = 0
    for k, it in enumerate(order, 1):
        cum_err += int(bool(it[error_key]))
        points.append({"k": k, "coverage": k / n, "risk": cum_err / k})

    # AURC over reachable prefix (trapezoidal integration normalized by max_cov if > 0)
    aurc = None
    if points:
        aurc = float(np.mean([p["risk"] for p in points]))

    fixed_risk: Dict[str, Optional[float]] = {}
    for g in grid:
        if g > max_cov + 1e-9 or not points:
            fixed_risk[f"{int(round(g * 100))}%"] = None  # unreachable -> NA, never fake 0
        else:
            k_target = max(1, int(round(g * n)))
            k_target = min(k_target, len(points))
            fixed_risk[f"{int(round(g * 100))}%"] = float(points[k_target - 1]["risk"])

    return {
        "n_pool": n,
        "n_reachable": len(order),
        "max_reachable_coverage": max_cov,
        "aurc_reachable": aurc,
        "fixed_coverage_risk": fixed_risk,
        "points": points,
    }
