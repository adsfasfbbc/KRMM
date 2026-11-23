"""End-to-end blind repair & full-pool discovery metrics for KRMM v0.4 (Section 4.4, 4.5, 6.3)."""
from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np

from krmm.v3.metrics import safe_auprc, safe_auroc


def compute_end_to_end_metrics(
    unit_outcomes: Sequence[Dict[str, object]],
    total_query_cost: int = 0,
    gpu_train_steps: int = 0,
) -> Dict[str, object]:
    """Compute End-to-End Strict Safe Repair Yield, Wrong-Write Rate, Healthy-Unit Harm Rate, and opportunity cost."""
    eligible_damaged = [u for u in unit_outcomes if u.get("pool_category") == "damaged_mastered"]
    healthy = [u for u in unit_outcomes if u.get("pool_category") == "healthy_mastered"]
    unknown = [u for u in unit_outcomes if u.get("pool_category") == "originally_unknown"]

    n_elig_dmg = len(eligible_damaged)
    n_healthy = len(healthy)
    n_unknown = len(unknown)
    n_all = len(unit_outcomes)

    detected_dmg = [u for u in eligible_damaged if u.get("detected_suspected")]
    cand_produced_dmg = [u for u in eligible_damaged if u.get("candidate_produced")]
    attempted_all = [u for u in unit_outcomes if u.get("action") == "ATTEMPT_PARAMETRIC_REPAIR"]
    attempted_dmg = [u for u in eligible_damaged if u.get("action") == "ATTEMPT_PARAMETRIC_REPAIR"]
    strict_safe_dmg = [
        u for u in eligible_damaged
        if u.get("detected_suspected")
        and u.get("action") == "ATTEMPT_PARAMETRIC_REPAIR"
        and u.get("strict_parametric_recovery")
    ]
    wrong_writes_all = [u for u in attempted_all if not u.get("candidate_matches_gold")]
    healthy_harmed = [u for u in healthy if u.get("action") == "ATTEMPT_PARAMETRIC_REPAIR" and u.get("healthy_harmed")]
    abstained_dmg = [u for u in eligible_damaged if u.get("action") in {"ABSTAIN", "REQUEST_EXTERNAL_EVIDENCE"}]
    missed_repairable = [
        u for u in abstained_dmg if u.get("oracle_repairable", True)
    ]

    return {
        "n_pool_total": n_all,
        "n_eligible_damaged": n_elig_dmg,
        "n_healthy_mastered": n_healthy,
        "n_originally_unknown": n_unknown,
        "n_detected_damaged": len(detected_dmg),
        "n_attempted_all": len(attempted_all),
        "n_attempted_damaged": len(attempted_dmg),
        "n_strict_safe_repaired": len(strict_safe_dmg),
        "end_to_end_strict_safe_repair_yield": float(len(strict_safe_dmg) / max(1, n_elig_dmg)),
        "wrong_write_count": len(wrong_writes_all),
        "wrong_write_rate_among_attempted": float(len(wrong_writes_all) / max(1, len(attempted_all))),
        "wrong_write_rate_over_eligible_damaged": float(len(wrong_writes_all) / max(1, n_elig_dmg)),
        "healthy_unit_harm_count": len(healthy_harmed),
        "healthy_unit_harm_rate": float(len(healthy_harmed) / max(1, n_healthy)),
        "candidate_coverage_damaged": float(len(cand_produced_dmg) / max(1, n_elig_dmg)),
        "repair_attempt_rate_damaged": float(len(attempted_dmg) / max(1, n_elig_dmg)),
        "abstention_rate_damaged": float(len(abstained_dmg) / max(1, n_elig_dmg)),
        "strict_success_among_attempted_damaged": float(len(strict_safe_dmg) / max(1, len(attempted_dmg))),
        "opportunity_cost_missed_repairable_count": len(missed_repairable),
        "total_query_cost": int(total_query_cost),
        "gpu_train_steps": int(gpu_train_steps),
    }


def full_pool_discovery_metrics(
    reports: Sequence[Dict[str, object]],
    truth_records: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """Compute strictly blind full-pool discovery metrics (never filtering unknown controls before ranking)."""
    truth_by_id = {t["unit_id"]: t for t in truth_records}
    scores = [float(r["anomaly_score"]) for r in reports]
    y_true = [1 if truth_by_id[r["unit_id"]]["pool_category"] in {"damaged_mastered", "collateral_damaged"} else 0 for r in reports]
    cats = [truth_by_id[r["unit_id"]]["pool_category"] for r in reports]

    n_pos = sum(y_true)
    k = max(1, n_pos)
    top_k = set(sorted(range(len(reports)), key=lambda i: scores[i], reverse=True)[:k])
    recall_k = sum(y_true[i] for i in top_k) / max(1, n_pos)
    healthy_idx = [i for i, c in enumerate(cats) if c == "healthy_mastered"]
    unk_idx = [i for i, c in enumerate(cats) if c == "originally_unknown"]
    healthy_fpr = sum(1 for i in healthy_idx if reports[i]["status"] == "suspected") / max(1, len(healthy_idx))
    unk_fpr = sum(1 for i in unk_idx if reports[i]["status"] == "suspected") / max(1, len(unk_idx))
    total_q = sum(int(r.get("query_cost", 0)) for r in reports)
    true_found = sum(1 for i, r in enumerate(reports) if y_true[i] == 1 and r["status"] == "suspected")

    return {
        "n_pool": len(reports),
        "n_true_degraded": n_pos,
        "full_pool_auroc": safe_auroc(y_true, scores),
        "full_pool_auprc": safe_auprc(y_true, scores),
        "full_pool_recall_at_k": float(recall_k),
        "healthy_fpr": float(healthy_fpr),
        "originally_unknown_fpr": float(unk_fpr),
        "total_query_cost": int(total_q),
        "cost_per_true_damage_found": float(total_q / max(1, true_found)) if true_found > 0 else None,
    }
