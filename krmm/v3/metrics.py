"""Metrics, statistical inference, and G0/G1 gate computation (M6.3, M6.4)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def safe_auroc(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    if len(set(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def safe_brier(y_true: Sequence[int], y_prob: Sequence[float]) -> Optional[float]:
    if not y_true:
        return None
    return float(brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0)))


def paired_cluster_bootstrap_ci(
    a_values: Sequence[float],
    b_values: Sequence[float],
    cluster_ids: Sequence[str],
    n_boot: int = 2000,
    seed: int = 107,
) -> Dict[str, object]:
    """Entity/source-group clustered paired bootstrap 95% CI for mean(a - b)."""
    if len(a_values) != len(b_values) or len(a_values) != len(cluster_ids):
        raise ValueError("Paired arrays and cluster_ids must match in length")
    if not a_values:
        return {"mean_diff": 0.0, "ci95": [0.0, 0.0], "clusters": 0}
    diffs = np.asarray(a_values, dtype=np.float64) - np.asarray(b_values, dtype=np.float64)
    groups: Dict[str, List[float]] = {}
    for d, cid in zip(diffs, cluster_ids):
        groups.setdefault(cid, []).append(float(d))
    cluster_means = [float(np.mean(v)) for v in groups.values()]
    rng = np.random.default_rng(seed)
    k = len(cluster_means)
    arr = np.asarray(cluster_means, dtype=np.float64)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, k, size=k)
        boots.append(float(np.mean(arr[idx])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": float(np.mean(arr)), "ci95": [float(lo), float(hi)], "clusters": k}


def detection_metrics(reports: Sequence[dict], is_damaged: Sequence[int], is_unknown_ctrl: Sequence[int]) -> Dict[str, object]:
    """Compute M2 detection AUROC, AUPRC, Recall@Budget, and unknown false-positive rate."""
    scores = [float(r["anomaly_score"]) for r in reports]
    y = [int(v) for v in is_damaged]
    known_idx = [i for i, unk in enumerate(is_unknown_ctrl) if not unk]
    unk_idx = [i for i, unk in enumerate(is_unknown_ctrl) if unk]

    y_known = [y[i] for i in known_idx]
    s_known = [scores[i] for i in known_idx]
    k_budget = max(1, sum(y_known))
    top_k = set(sorted(known_idx, key=lambda i: scores[i], reverse=True)[:k_budget])
    recall_at_k = sum(y[i] for i in top_k) / max(1, sum(y_known))
    prec_at_k = sum(y[i] for i in top_k) / max(1, len(top_k))
    healthy_fpr = (
        sum(1 for i in known_idx if y[i] == 0 and reports[i]["status"] == "suspected")
        / max(1, sum(1 for i in known_idx if y[i] == 0))
    )
    unknown_fpr = (
        sum(1 for i in unk_idx if reports[i]["status"] == "suspected")
        / max(1, len(unk_idx))
    )
    mean_cost = float(np.mean([r["query_cost"] for r in reports])) if reports else 0.0
    return {
        "n_total": len(reports),
        "n_known": len(known_idx),
        "n_damaged": sum(y_known),
        "n_unknown_ctrl": len(unk_idx),
        "auroc": safe_auroc(y_known, s_known),
        "auprc": safe_auprc(y_known, s_known),
        "recall_at_damaged_budget": float(recall_at_k),
        "precision_at_damaged_budget": float(prec_at_k),
        "healthy_false_positive_rate": float(healthy_fpr),
        "unknown_false_positive_rate": float(unknown_fpr),
        "mean_query_cost": mean_cost,
    }


def summarize_p0_results(records: Sequence[dict], gate_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    """Summarize P0 mastery, damage validity, independent locality, and oracle writeback feasibility."""
    thresh = gate_thresholds or {
        "min_independent_units": 8,
        "min_clean_mastery_rate": 0.75,
        "min_damage_eligibility_rate": 0.50,
        "min_oracle_success_rate": 0.75,
        "min_retention_agreement": 0.90,
    }
    total = len(records)
    mastered = [r for r in records if r["mastery"]["stably_mastered"]]
    heldout_mastered = [r for r in mastered if r["mastery"]["heldout_clean_mastered"]]
    dmg_eff = [r for r in mastered if r.get("damage_effective")]
    op_elig = [r for r in records if r.get("operational_eligible")]
    dmg_loc_pass = [r for r in op_elig if r.get("damage_independent_locality", {}).get("independent_locality_pass")]
    strict_elig = [r for r in records if r.get("strict_eligible")]

    evaluated = [r for r in records if "repairs" in r and "oracle" in r["repairs"]]
    oracle_op_succ = [r for r in evaluated if r["repairs"]["oracle"]["operational_success"]]
    oracle_write_feas = [r for r in evaluated if r["repairs"]["oracle"]["repair_write_feasible"]]
    oracle_strict = [r for r in evaluated if r["repairs"]["oracle"]["strict_restoration"]]

    clean_mastery_rate = len(mastered) / max(1, total)
    op_elig_rate = len(op_elig) / max(1, len(mastered))
    strict_elig_rate = len(strict_elig) / max(1, len(mastered))
    oracle_op_rate = len(oracle_op_succ) / max(1, len(evaluated))
    oracle_write_feas_rate = len(oracle_write_feas) / max(1, len(evaluated))
    oracle_strict_rate_strict_elig = (
        sum(1 for r in strict_elig if r.get("repairs", {}).get("oracle", {}).get("strict_restoration"))
        / max(1, len(strict_elig))
    )
    min_ret = min((r.get("damage_train_retention", 0.0) for r in op_elig), default=0.0)

    exclusions: Dict[str, int] = {}
    for r in records:
        if not r.get("strict_eligible"):
            reason = str(r.get("exclusion_reason") or "unknown")
            exclusions[reason] = exclusions.get(reason, 0) + 1

    # G0 strict criteria: A (mastery >= 0.75), B (strict_eligible >= 8 & rate >= 0.5), C (oracle_strict >= 0.75)
    g0_a_ok = clean_mastery_rate >= thresh["min_clean_mastery_rate"]
    g0_b_ok = (
        len(strict_elig) >= thresh["min_independent_units"]
        and strict_elig_rate >= thresh["min_damage_eligibility_rate"]
        and min_ret >= thresh["min_retention_agreement"]
    )
    g0_c_ok = len(strict_elig) > 0 and oracle_strict_rate_strict_elig >= thresh["min_oracle_success_rate"]
    g0_pass = bool(g0_a_ok and g0_b_ok and g0_c_ok)

    # Diagnostic operational check (without independent locality filter)
    diag_op_ok = (
        clean_mastery_rate >= thresh["min_clean_mastery_rate"]
        and len(op_elig) >= thresh["min_independent_units"]
        and op_elig_rate >= thresh["min_damage_eligibility_rate"]
        and oracle_op_rate >= thresh["min_oracle_success_rate"]
    )

    return {
        "total_candidates": total,
        "stably_mastered": len(mastered),
        "heldout_clean_mastered": len(heldout_mastered),
        "clean_mastery_rate": clean_mastery_rate,
        "damage_effective": len(dmg_eff),
        "operational_eligible": len(op_elig),
        "operational_eligibility_rate": op_elig_rate,
        "damage_independent_locality_pass": len(dmg_loc_pass),
        "strict_eligible": len(strict_elig),
        "strict_eligibility_rate": strict_elig_rate,
        "evaluated_counterfactual": len(evaluated),
        "oracle_operational_success": len(oracle_op_succ),
        "oracle_operational_rate": oracle_op_rate,
        "oracle_write_feasible_count": len(oracle_write_feas),
        "oracle_write_feasible_rate": oracle_write_feas_rate,
        "oracle_strict_restorations": len(oracle_strict),
        "oracle_strict_rate_among_strict_eligible": oracle_strict_rate_strict_elig,
        "min_damage_train_retention": min_ret,
        "exclusion_breakdown": exclusions,
        "gate_g0": {
            "pass": g0_pass,
            "A_clean_mastery_ok": g0_a_ok,
            "B_damage_local_and_effective_ok": g0_b_ok,
            "C_oracle_strict_writeback_ok": g0_c_ok,
            "operational_only_ok": diag_op_ok,
        },
    }
