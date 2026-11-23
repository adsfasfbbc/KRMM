"""Equal-budget blind damage discovery (7 comparators + 3-class & 2-class full-pool metrics, Section 4.4-4.6)."""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence
import numpy as np

from krmm.v3.discovery import BudgetExhaustedError, BudgetedQueryProxy, detect
from krmm.v3.evidence import extract_atomic_facts
from krmm.v3.knowledge_registry import question_prompt
from krmm.v3.metrics import safe_auprc, safe_auroc
from krmm.v3.schemas import DamageReport, KnowledgeUnit
from .support_structure import analyze_support_structure

V5_DETECTORS = (
    "direct_confidence",
    "entropy",
    "self_consistency",
    "direct_paraphrases",
    "constraint_only",
    "m2_reference_free",
    "full_m2",
)


class CostTrackedProxy(BudgetedQueryProxy):
    """Extends BudgetedQueryProxy with approximate input/output token and wall-time accounting."""

    def __init__(self, backend=None, query_budget: int = 0, cached_responses=None) -> None:
        super().__init__(backend=backend, query_budget=query_budget, cached_responses=cached_responses)
        self.input_tokens = 0
        self.output_tokens = 0
        self.wall_seconds = 0.0

    def probabilities(self, prompt: str, media=None):
        t0 = time.perf_counter()
        res = super().probabilities(prompt, media=media)
        self.wall_seconds += time.perf_counter() - t0
        self.input_tokens += max(1, len(prompt.split()))
        self.output_tokens += 1
        return res

    def generate(self, prompt: str, max_new_tokens: int = 32, media=None) -> str:
        t0 = time.perf_counter()
        res = super().generate(prompt, max_new_tokens=max_new_tokens, media=media)
        self.wall_seconds += time.perf_counter() - t0
        self.input_tokens += max(1, len(prompt.split()))
        self.output_tokens += max(1, len(res.split()))
        return res


def _choose(probs: Sequence[float]) -> str:
    return "ABCD"[int(np.argmax(probs))]


def _entropy(probs: Sequence[float]) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def run_v5_detector(
    proxy: CostTrackedProxy,
    public_units: Sequence[KnowledgeUnit],
    query_budget: int,
    method: str,
) -> Dict[str, object]:
    """Run one of the 7 v0.5 blind detectors under strict equal query_budget."""
    if method not in V5_DETECTORS:
        raise ValueError(f"Unsupported v5 detector: {method}")

    if method in {"m2_reference_free", "full_m2"}:
        frac = 0.65 if method == "m2_reference_free" else 0.95
        reps = detect(proxy, public_units, probe_budget=query_budget, mode="reference_free", constraint_probe_fraction=frac)
        rep_dicts = [r.to_dict() for r in reps]
    else:
        rep_dicts = []
        for u in public_units:
            proxy.set_current_unit(u.unit_id)
            c0 = proxy.total_queries
            try:
                if method in {"direct_confidence", "entropy"}:
                    p2, _, _ = question_prompt(u, variant=2)
                    probs, mass = proxy.probabilities(p2)
                    conf = float(np.max(probs))
                    ent = _entropy(probs)
                    norm_ent = min(1.0, ent / np.log(4.0))
                    score = float(1.0 - conf) if method == "direct_confidence" else norm_ent
                    if mass < 0.35:
                        status = "unknown"
                    elif score >= 0.45:
                        status = "suspected"
                    else:
                        status = "no_evidence_of_damage"
                    rep_dicts.append(
                        DamageReport(u.unit_id, score, [], [{"kind": method, "conf": conf, "ent": ent, "mass": mass}],
                                     [], "reference_free", norm_ent, proxy.total_queries - c0, status).to_dict()
                    )
                elif method in {"self_consistency", "direct_paraphrases"}:
                    p2, _, _ = question_prompt(u, variant=2)
                    probs2, mass2 = proxy.probabilities(p2)
                    pred2 = _choose(probs2)
                    p3, _, _ = question_prompt(u, variant=3)
                    probs3, mass3 = proxy.probabilities(p3)
                    pred3 = _choose(probs3)
                    perm_dis = 0.0
                    if method == "direct_paraphrases" and proxy.total_queries < query_budget:
                        pp, d2c, _ = question_prompt(u, variant=2, permutation=("D", "C", "B", "A"))
                        raw_pp, _ = proxy.probabilities(pp)
                        perm_dis = float(d2c[_choose(raw_pp)] != pred2)
                    var_dis = float(pred2 != pred3)
                    score = min(1.0, 0.5 * var_dis + 0.5 * perm_dis) if method == "direct_paraphrases" else var_dis
                    avg_m = (mass2 + mass3) / 2.0
                    status = "unknown" if avg_m < 0.35 else ("suspected" if score >= 0.5 else "no_evidence_of_damage")
                    rep_dicts.append(
                        DamageReport(u.unit_id, float(score), [], [{"kind": method, "pred2": pred2, "pred3": pred3}],
                                     [], "reference_free", min(1.0, _entropy(probs2) / np.log(4.0)), proxy.total_queries - c0, status).to_dict()
                    )
                elif method == "constraint_only":
                    facts, _ = extract_atomic_facts(proxy, u, max_new_tokens=24)
                    st = analyze_support_structure(u.domain, facts)
                    score = 0.90 if st.conflict_status else (0.35 if not st.identifiable else 0.10)
                    status = "suspected" if st.conflict_status else ("unknown" if not st.identifiable else "no_evidence_of_damage")
                    rep_dicts.append(
                        DamageReport(u.unit_id, score, [], [{"kind": "constraint_only", "class": st.support_class}],
                                     [{"reason": "conflict"}] if st.conflict_status else [], "reference_free",
                                     0.2 if st.identifiable else 0.9, proxy.total_queries - c0, status).to_dict()
                    )
            except BudgetExhaustedError:
                rep_dicts.append(
                    DamageReport(u.unit_id, 0.0, [], [{"kind": "budget_exhausted"}], [], "reference_free", 1.0,
                                 proxy.total_queries - c0, "unknown").to_dict()
                )

    if proxy.total_queries > int(query_budget):
        raise RuntimeError(f"Budget violated by {method}: {proxy.total_queries} > {query_budget}")

    return {
        "method": method,
        "query_budget": int(query_budget),
        "total_queries": int(proxy.total_queries),
        "mean_query_cost": float(proxy.total_queries / max(1, len(public_units))),
        "input_tokens": int(proxy.input_tokens),
        "output_tokens": int(proxy.output_tokens),
        "total_tokens": int(proxy.input_tokens + proxy.output_tokens),
        "wall_seconds": float(proxy.wall_seconds),
        "reports": rep_dicts,
    }


def evaluate_v5_discovery_metrics(
    run_out: Dict[str, object],
    truth_records: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """Compute binary full-pool + 3-class (damaged / healthy / unknown) + collateral damage recall metrics."""
    reports = run_out["reports"]
    tmap = {t["unit_id"]: t for t in truth_records}
    scores = [float(r["anomaly_score"]) for r in reports]
    cats = [tmap[r["unit_id"]]["pool_category"] for r in reports]
    y_bin = [1 if c in {"damaged_mastered", "collateral_damaged"} else 0 for c in cats]

    n_dmg = sum(y_bin)
    k = max(1, n_dmg)
    top_k = set(sorted(range(len(reports)), key=lambda i: scores[i], reverse=True)[:k])
    rec_k = sum(y_bin[i] for i in top_k) / max(1, n_dmg)
    prec_k = sum(y_bin[i] for i in top_k) / max(1, len(top_k))

    hlt_idx = [i for i, c in enumerate(cats) if c == "healthy_mastered"]
    unk_idx = [i for i, c in enumerate(cats) if c == "originally_unknown"]
    col_idx = [i for i, c in enumerate(cats) if c == "collateral_damaged"]
    dmg_idx = [i for i, c in enumerate(cats) if c == "damaged_mastered"]

    healthy_fpr = sum(1 for i in hlt_idx if reports[i]["status"] == "suspected") / max(1, len(hlt_idx))
    unknown_fpr = sum(1 for i in unk_idx if reports[i]["status"] == "suspected") / max(1, len(unk_idx))
    col_recall = (sum(1 for i in col_idx if reports[i]["status"] == "suspected") / len(col_idx)) if col_idx else None
    det_cov = sum(1 for i in dmg_idx if reports[i]["status"] == "suspected") / max(1, len(dmg_idx))

    # 3-class mapping: suspected -> damaged, no_evidence_of_damage -> healthy, unknown -> unknown
    pred_3 = [
        "damaged" if r["status"] == "suspected" else ("unknown" if r["status"] == "unknown" else "healthy")
        for r in reports
    ]
    true_3 = [
        "damaged" if c in {"damaged_mastered", "collateral_damaged"}
        else ("unknown" if c in {"originally_unknown", "ambiguous_or_unverifiable"} else "healthy")
        for c in cats
    ]
    acc_3 = sum(1 for p, t in zip(pred_3, true_3) if p == t) / max(1, len(true_3))

    return {
        "method": run_out["method"],
        "query_budget": run_out["query_budget"],
        "total_queries": run_out["total_queries"],
        "mean_query_cost": run_out["mean_query_cost"],
        "total_tokens": run_out["total_tokens"],
        "wall_seconds": run_out["wall_seconds"],
        "n_pool": len(reports),
        "n_damaged_mastered": len(dmg_idx),
        "n_healthy_mastered": len(hlt_idx),
        "n_originally_unknown": len(unk_idx),
        "n_collateral_damaged": len(col_idx),
        "binary_full_pool": {
            "auroc": safe_auroc(y_bin, scores),
            "auprc": safe_auprc(y_bin, scores),
            "precision_at_budget": float(prec_k),
            "recall_at_budget": float(rec_k),
            "healthy_fpr": float(healthy_fpr),
            "unknown_fpr": float(unknown_fpr),
            "collateral_damage_recall": col_recall,
            "end_to_end_detection_coverage": float(det_cov),
        },
        "three_class": {
            "accuracy": float(acc_3),
            "unknown_correctly_uncertain_rate": sum(1 for i in unk_idx if pred_3[i] == "unknown") / max(1, len(unk_idx)),
        },
    }
