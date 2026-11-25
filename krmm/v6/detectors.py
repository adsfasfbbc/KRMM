"""Fair equal-budget blind damage detectors (Section 5.3) with AUPRC primary metric & fixed top-k."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple
import numpy as np

from krmm.v3.evidence import extract_atomic_facts
from krmm.v3.knowledge_registry import question_prompt
from krmm.v3.schemas import KnowledgeUnit
from krmm.v5.support_structure import analyze_support_structure
from .budget import BudgetExhaustedError, StrictBudgetProxy
from .schemas import PublicUnit
from .truth_audit import deterministic_tie_key

V6_DETECTORS: Tuple[str, ...] = (
    "low_confidence",
    "high_confidence_rev",
    "entropy",
    "neg_entropy_rev",
    "self_consistency",
    "repeated_paraphrases_perm",
    "direct_vs_constraint",
    "m2_reference_free",
    "full_m2",
)


def _to_ku(pu: PublicUnit) -> KnowledgeUnit:
    return pu.to_knowledge_unit("test")


def _entropy(probs: Sequence[float]) -> float:
    h = 0.0
    for p in probs:
        if p > 1e-12:
            h -= p * math.log(p)
    return float(h / math.log(max(2, len(probs))))


def _argmax_letter(probs: Sequence[float]) -> str:
    return "ABCD"[int(np.argmax(probs))]


def _auroc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    pos = [s for y, s in zip(y_true, scores) if y == 1]
    neg = [s for y, s in zip(y_true, scores) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _auprc(y_true: Sequence[int], scores: Sequence[float], uids: Sequence[str], seed: int) -> float:
    order = sorted(
        range(len(y_true)),
        key=lambda i: (-float(scores[i]), deterministic_tie_key(uids[i], seed)),
    )
    n_pos = sum(y_true)
    if n_pos == 0:
        return 0.0
    tp = 0
    ap = 0.0
    for rank, idx in enumerate(order, 1):
        if y_true[idx] == 1:
            tp += 1
            ap += tp / rank
    return ap / n_pos


def run_v6_detector(
    proxy: StrictBudgetProxy,
    public_units: Sequence[PublicUnit],
    method: str,
    seed: int = 2026,
) -> Dict[str, object]:
    """Run one detector strictly within proxy's budget on shuffled PublicUnits."""
    if method not in V6_DETECTORS:
        raise ValueError(f"Unknown detector: {method}")

    n = len(public_units)
    reports: List[Dict[str, object]] = []
    per_unit_budget = max(1, proxy.query_budget // max(1, n))

    for pu in public_units:
        ku = _to_ku(pu)
        score = 0.0
        pred_state = "healthy"
        meta: Dict[str, object] = {}
        try:
            p0, _, _ = question_prompt(ku, variant=0)
            probs0, mass0 = proxy.probabilities(p0)
            c0 = _argmax_letter(probs0)
            max_p0 = float(max(probs0))
            ent0 = _entropy(probs0)
            meta["p0_max"] = max_p0
            meta["p0_entropy"] = ent0
            meta["p0_mass"] = float(mass0)

            if method == "low_confidence":
                score = 1.0 - max_p0
                pred_state = "damaged" if score >= 0.45 else ("uncertain" if mass0 < 0.25 else "healthy")
            elif method == "high_confidence_rev":
                # Reverse direction frozen on dev (D1 wrong-fact overconfidence)
                score = max_p0
                pred_state = "damaged" if max_p0 >= 0.90 else ("uncertain" if max_p0 < 0.55 else "healthy")
            elif method == "entropy":
                score = ent0
                pred_state = "damaged" if ent0 >= 0.55 else ("uncertain" if mass0 < 0.25 else "healthy")
            elif method == "neg_entropy_rev":
                score = 1.0 - ent0
                pred_state = "damaged" if ent0 <= 0.15 else ("uncertain" if ent0 >= 0.55 else "healthy")
            elif method == "self_consistency":
                p1, _, _ = question_prompt(ku, variant=1)
                probs1, _ = proxy.probabilities(p1)
                c1 = _argmax_letter(probs1)
                disagree = float(c0 != c1)
                tv = 0.5 * sum(abs(a - b) for a, b in zip(probs0, probs1))
                score = 0.6 * disagree + 0.4 * tv
                pred_state = "damaged" if score >= 0.35 else ("uncertain" if mass0 < 0.25 else "healthy")
            elif method == "repeated_paraphrases_perm":
                # Uses up to per_unit_budget queries (variants + option permutations)
                preds = [c0]
                probs_list = [probs0]
                extras = [
                    (1, None),
                    (2, ("D", "C", "B", "A")),
                    (3, ("B", "D", "A", "C")),
                    (0, ("C", "A", "D", "B")),
                ]
                for v, perm in extras[: max(0, per_unit_budget - 1)]:
                    if proxy.remaining() <= 0:
                        break
                    pv, d2c, _ = question_prompt(ku, variant=v, permutation=perm)
                    pp, _ = proxy.probabilities(pv)
                    preds.append(d2c[_argmax_letter(pp)])
                    probs_list.append(pp)
                maj = max(set(preds), key=preds.count)
                disagree_rate = 1.0 - preds.count(maj) / len(preds)
                mean_ent = float(np.mean([_entropy(pr) for pr in probs_list]))
                score = 0.7 * disagree_rate + 0.3 * mean_ent
                pred_state = "damaged" if disagree_rate > 0 else ("uncertain" if mean_ent > 0.55 else "healthy")
            elif method == "direct_vs_constraint":
                # 1 direct query + 1-2 atomic fact extraction queries -> constraint conflict check
                if proxy.remaining() >= 1:
                    facts, _ = extract_atomic_facts(proxy, ku, max_new_tokens=20)
                    sup = analyze_support_structure(ku.domain, facts)
                    if sup.candidate_answer is not None and sup.candidate_answer != c0:
                        score = 0.90 if sup.support_class in ("redundant_distinct_attributes", "single_decisive") else 0.70
                        pred_state = "damaged"
                    elif sup.support_class == "conflicting_support":
                        score = 0.55
                        pred_state = "uncertain"
                    elif sup.candidate_answer is None:
                        score = 0.20
                        pred_state = "uncertain"
                    else:
                        score = 0.05
                        pred_state = "healthy"
            elif method in ("m2_reference_free", "full_m2"):
                # Paraphrase/permutation stability + atomic-fact constraint conflict
                p_perm, d2c, _ = question_prompt(ku, variant=2, permutation=("D", "C", "B", "A"))
                probs_p, mass_p = proxy.probabilities(p_perm)
                c_perm = d2c[_argmax_letter(probs_p)]
                perm_disagree = float(c0 != c_perm)
                facts = []
                sup_cand = None
                sup_cls = "insufficient_support"
                if proxy.remaining() >= 1:
                    try:
                        facts, _ = extract_atomic_facts(proxy, ku, max_new_tokens=20)
                        sup = analyze_support_structure(ku.domain, facts)
                        sup_cand = sup.candidate_answer
                        sup_cls = sup.support_class
                    except BudgetExhaustedError:
                        pass
                conflict = float(sup_cand is not None and sup_cand != c0)
                strong_sup = float(sup_cls in ("redundant_distinct_attributes", "single_decisive"))
                low_mass = float(min(mass0, mass_p) < 0.25)
                if method == "m2_reference_free":
                    # Conservative on unknown controls: requires actual support candidate conflict
                    score = 0.65 * conflict + 0.15 * (conflict * strong_sup) + 0.20 * perm_disagree * (1.0 - low_mass)
                    if conflict > 0:
                        pred_state = "damaged"
                    elif sup_cand is None or low_mass > 0:
                        pred_state = "uncertain"
                    else:
                        pred_state = "healthy"
                else:
                    # full_m2: adds permutation discrepancy & conflict signal even when support is partial
                    score = 0.60 * conflict + 0.15 * (conflict * strong_sup) + 0.25 * perm_disagree
                    if conflict > 0 or perm_disagree > 0:
                        pred_state = "damaged"
                    elif sup_cand is None:
                        pred_state = "uncertain"
                    else:
                        pred_state = "healthy"
                meta["sup_class"] = sup_cls
                meta["sup_candidate"] = sup_cand
                meta["perm_disagree"] = perm_disagree
        except BudgetExhaustedError:
            pred_state = "uncertain"
            score = 0.0
            meta["budget_truncated"] = True

        reports.append(
            {
                "unit_id": pu.unit_id,
                "entity_id": pu.entity_id,
                "domain": pu.domain,
                "source_group": pu.source_group,
                "score": float(score),
                "pred_state": pred_state,
                "meta": meta,
            }
        )

    # Sort deterministically with UID-hash tie-breaking
    ranked = sorted(
        reports,
        key=lambda r: (-float(r["score"]), deterministic_tie_key(str(r["unit_id"]), seed)),
    )
    for rank_idx, r in enumerate(ranked, 1):
        r["rank"] = rank_idx

    return {
        "method": method,
        "accounting": proxy.accounting(),
        "reports": ranked,
    }


def evaluate_v6_discovery(
    run_out: Dict[str, object],
    truth_records: Sequence[Dict[str, object]],
    fixed_k_fraction: float = 0.25,
    seed: int = 2026,
) -> Dict[str, object]:
    """Evaluate detector using full-pool AUPRC (primary), AUROC (secondary), fixed top-k (NO oracle k!), and 3-class confusion."""
    tmap = {r["unit_id"]: r for r in truth_records}
    reports = list(run_out["reports"])

    def _eval_subset(sub_reports: Sequence[Dict[str, object]]) -> Dict[str, object]:
        y_bin: List[int] = []
        scores: List[float] = []
        uids: List[str] = []
        for r in sub_reports:
            tr = tmap[r["unit_id"]]
            y_bin.append(1 if tr["is_degraded_target"] else 0)
            scores.append(float(r["score"]))
            uids.append(str(r["unit_id"]))

        n_sub = len(sub_reports)
        k_fixed = max(1, int(math.floor(fixed_k_fraction * n_sub)))
        order = sorted(range(n_sub), key=lambda i: (-scores[i], deterministic_tie_key(uids[i], seed)))
        top_k = order[:k_fixed]
        n_pos = sum(y_bin)
        tp_k = sum(y_bin[i] for i in top_k)

        # Category-specific rates using predicted state == "damaged"
        healthy_ids = [r for r in sub_reports if tmap[r["unit_id"]]["truth_category"] == "original_mastered_healthy"]
        unk_ids = [r for r in sub_reports if tmap[r["unit_id"]]["truth_category"] in ("previously_unmastered", "unverifiable")]
        collat_ids = [r for r in sub_reports if tmap[r["unit_id"]]["truth_category"] == "collateral_damaged"]
        dmg_ids = [r for r in sub_reports if tmap[r["unit_id"]]["is_degraded_target"]]

        healthy_fpr = sum(1 for r in healthy_ids if r["pred_state"] == "damaged") / max(1, len(healthy_ids))
        unk_fpr = sum(1 for r in unk_ids if r["pred_state"] == "damaged") / max(1, len(unk_ids))
        collat_rec = (
            sum(1 for r in collat_ids if r["pred_state"] == "damaged") / len(collat_ids)
            if collat_ids
            else None
        )
        dmg_recall = sum(1 for r in dmg_ids if r["pred_state"] == "damaged") / max(1, len(dmg_ids))

        # 3-class confusion matrix: true in {damaged, healthy, unknown_or_unverifiable}
        conf: Dict[str, Dict[str, int]] = {
            "damaged": {"damaged": 0, "healthy": 0, "uncertain": 0},
            "healthy": {"damaged": 0, "healthy": 0, "uncertain": 0},
            "unknown_or_unverifiable": {"damaged": 0, "healthy": 0, "uncertain": 0},
        }
        correct_3 = 0
        for r in sub_reports:
            cat = tmap[r["unit_id"]]["truth_category"]
            if cat in ("effective_planned_damage", "collateral_damaged"):
                t3 = "damaged"
            elif cat == "original_mastered_healthy":
                t3 = "healthy"
            else:
                t3 = "unknown_or_unverifiable"
            p3 = str(r["pred_state"])
            conf[t3][p3] = conf[t3].get(p3, 0) + 1
            if (t3 == "damaged" and p3 == "damaged") or (t3 == "healthy" and p3 == "healthy") or (t3 == "unknown_or_unverifiable" and p3 == "uncertain"):
                correct_3 += 1

        return {
            "n_units": n_sub,
            "n_degraded_D": n_pos,
            "auprc_primary": _auprc(y_bin, scores, uids, seed),
            "auroc_secondary": _auroc(y_bin, scores),
            "fixed_k": k_fixed,
            "precision_at_fixed_k": tp_k / k_fixed,
            "recall_at_fixed_k": tp_k / max(1, n_pos) if n_pos > 0 else None,
            "damaged_recall_thresholded": dmg_recall,
            "healthy_fpr": healthy_fpr,
            "unknown_fpr": unk_fpr,
            "collateral_recall": collat_rec,
            "three_class_accuracy": correct_3 / max(1, n_sub),
            "three_class_confusion": conf,
        }

    full_metrics = _eval_subset(reports)
    no_amb_reports = [r for r in reports if tmap[r["unit_id"]]["pool_role"] != "ambiguous"]
    no_amb_metrics = _eval_subset(no_amb_reports)

    return {
        "method": run_out["method"],
        "accounting": run_out["accounting"],
        "full_pool": full_metrics,
        "sensitivity_without_ambiguous": no_amb_metrics,
    }
