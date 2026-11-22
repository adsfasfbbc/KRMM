"""Damage auto-discovery in reference-based and reference-free modes (M2). Blind to attacker target list."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np

from .evidence import extract_atomic_facts
from .knowledge_registry import question_prompt
from .reconstruction import reconstruct_structured
from .schemas import DamageReport, KnowledgeUnit


def _choose(probs: Sequence[float]) -> str:
    return "ABCD"[int(np.argmax(probs))]


def _entropy(probs: Sequence[float]) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _l1_dist(p: Sequence[float], q: Sequence[float]) -> float:
    return float(np.sum(np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64))))


def detect(
    omni,
    public_units: Sequence[KnowledgeUnit],
    probe_budget: int,
    mode: str = "reference_free",
    baseline_records: Optional[Dict[str, Dict[str, object]]] = None,
    constraint_probe_fraction: float = 0.85,
) -> List[DamageReport]:
    """Discover suspected damaged units within probe_budget without accessing damage targets or truth."""
    if mode not in {"reference_based", "reference_free"}:
        raise ValueError(f"Unsupported detection mode: {mode}")
    if mode == "reference_based" and baseline_records is None:
        raise ValueError("reference_based mode requires baseline_records")

    calls_used = 0
    stage1: List[Dict[str, object]] = []

    # When constraint probing is active, keep Stage 1 lean (2 calls/unit) so budget remains for EvidenceGraph!
    if constraint_probe_fraction > 0.0 and probe_budget >= 4 * len(public_units):
        queries_per_unit = 2
    elif probe_budget >= 3 * len(public_units):
        queries_per_unit = 3
    elif probe_budget >= 2 * len(public_units):
        queries_per_unit = 2
    else:
        queries_per_unit = 1

    for u in public_units:
        if calls_used + queries_per_unit > probe_budget:
            stage1.append({"unit": u, "skipped": True, "cost": 0, "risk": 0.0})
            continue

        p2, _, _ = question_prompt(u, variant=2)
        probs2, mass2 = omni.probabilities(p2)
        calls_used += 1
        cost_u = 1

        pred2 = _choose(probs2)
        ent2 = _entropy(probs2)

        probs3, mass3, pred3 = probs2, mass2, pred2
        perm_pred = pred2

        if queries_per_unit >= 2:
            p3, _, _ = question_prompt(u, variant=3)
            probs3, mass3 = omni.probabilities(p3)
            calls_used += 1
            cost_u += 1
            pred3 = _choose(probs3)

        if queries_per_unit >= 3:
            pp, d2c, _ = question_prompt(u, variant=2, permutation=("D", "C", "B", "A"))
            raw_pp, _ = omni.probabilities(pp)
            calls_used += 1
            cost_u += 1
            perm_pred = d2c[_choose(raw_pp)]

        var_disagree = float(pred2 != pred3)
        perm_disagree = float(pred2 != perm_pred)
        avg_mass = float((mass2 + mass3) / 2.0)
        norm_ent = min(1.0, ent2 / np.log(4.0))

        if mode == "reference_based":
            base = baseline_records.get(u.unit_id)
            if base is None:
                risk = 0.0
                uncertainty = 1.0
                flipped = False
                shift = 0.0
                observations = [{"kind": "missing_baseline"}]
            else:
                base_probs = base["probs"]
                base_pred = _choose(base_probs)
                shift = _l1_dist(probs2, base_probs)
                flipped = bool(pred2 != base_pred or pred3 != base_pred or perm_pred != base_pred)
                risk = min(1.0, 0.60 * float(flipped) + 0.40 * (shift / 2.0))
                uncertainty = norm_ent
                observations = [
                    {"kind": "reference_shift", "l1": shift, "flipped": flipped, "pred2": pred2, "base_pred": base_pred}
                ]
        else:
            # Stage 1 heuristic separates unstable/inconsistent answers from low-confidence unknowns
            is_likely_unknown = (avg_mass < 0.35) or (ent2 > 1.10 and var_disagree == 0.0)
            if is_likely_unknown:
                risk = 0.15 * norm_ent
            else:
                risk = min(1.0, 0.45 * perm_disagree + 0.40 * var_disagree + 0.15 * (1.0 - norm_ent))
            uncertainty = norm_ent
            flipped = False
            shift = 0.0
            observations = [
                {
                    "kind": "patrol_probes",
                    "pred2": pred2,
                    "pred3": pred3,
                    "perm_pred": perm_pred,
                    "entropy": ent2,
                    "choice_mass": avg_mass,
                }
            ]

        stage1.append(
            {
                "unit": u,
                "skipped": False,
                "cost": cost_u,
                "risk": float(risk),
                "uncertainty": float(uncertainty),
                "pred2": pred2,
                "pred3": pred3,
                "perm_pred": perm_pred,
                "ent2": ent2,
                "avg_mass": avg_mass,
                "flipped": flipped,
                "shift": shift,
                "observations": observations,
            }
        )

    # Stage 2: Prioritize confident/unstable known-candidate units (NOT low-mass unknown controls) for EvidenceGraph!
    remaining = max(0, probe_budget - calls_used)
    eligible_s2 = [
        item for item in stage1
        if not item["skipped"] and item["avg_mass"] >= 0.35 and item["ent2"] <= 1.10
    ]
    # Rank by instability first, then confident units (since D1 makes wrong answers confident!)
    ranked = sorted(eligible_s2, key=lambda x: (x["risk"], 1.0 - x["ent2"]), reverse=True)
    max_deep = int(np.ceil(len(stage1) * constraint_probe_fraction))
    deep_targets = {item["unit"].unit_id for item in ranked[:max_deep]}

    domain_risks: Dict[str, List[float]] = {}
    for item in stage1:
        if not item["skipped"]:
            domain_risks.setdefault(item["unit"].domain, []).append(item["risk"])

    reports: List[DamageReport] = []
    for item in stage1:
        u: KnowledgeUnit = item["unit"]
        if item["skipped"]:
            reports.append(
                DamageReport(
                    unit_id=u.unit_id,
                    anomaly_score=0.0,
                    suspected_paths=[],
                    supporting_observations=[{"kind": "budget_exhausted"}],
                    conflicts=[],
                    reference_mode=mode,
                    uncertainty=1.0,
                    query_cost=0,
                    status="unknown",
                )
            )
            continue

        conflicts: List[Dict[str, object]] = []
        suspected_paths: List[str] = []
        observations = list(item["observations"])
        cost_u = int(item["cost"])
        score = float(item["risk"])
        uncertainty = float(item["uncertainty"])

        if item["pred2"] != item["perm_pred"]:
            suspected_paths.append("option_permutation_instability")
        if item["pred2"] != item["pred3"]:
            suspected_paths.append("template_paraphrase_instability")

        if u.unit_id in deep_targets and remaining >= 2:
            facts, probe_calls = extract_atomic_facts(omni, u, max_new_tokens=24)
            calls_used += probe_calls
            remaining = max(0, probe_budget - calls_used)
            cost_u += probe_calls
            struct = reconstruct_structured(u.domain, facts)
            observations.append(
                {
                    "kind": "constraint_graph_check",
                    "structured_candidate": struct["candidate"],
                    "compatible": struct["graph"]["compatible"],
                    "valid_fact_count": len(struct["valid_facts"]),
                }
            )
            # Only raise constraint conflict in reference_based if historical shift actually occurred!
            allow_conflict = (mode == "reference_free") or bool(item["flipped"] or item["shift"] >= 0.25)
            if allow_conflict:
                if struct["graph"]["conflict"]:
                    conflicts.append({"reason": "internal_attribute_conflict", "details": struct["graph"]["conflicts"]})
                    suspected_paths.append("attribute_graph_conflict")
                    score = max(score, 0.80)
                elif struct["candidate"] is not None:
                    if struct["candidate"] != item["pred2"] or struct["candidate"] != item["pred3"] or struct["candidate"] != item["perm_pred"]:
                        conflicts.append(
                            {
                                "reason": "direct_vs_constraint_mismatch",
                                "direct_pred2": item["pred2"],
                                "direct_pred3": item["pred3"],
                                "direct_perm": item["perm_pred"],
                                "constraint_candidate": struct["candidate"],
                            }
                        )
                        suspected_paths.append("direct_vs_retained_constraint_mismatch")
                        score = max(score, 0.92)
                        uncertainty = min(uncertainty, 0.20)
                    else:
                        # Direct prediction agrees with independent retained attribute graph -> lowers false suspicion
                        score = min(score, 0.20)

        dom_mean = float(np.mean(domain_risks.get(u.domain, [0.0])))
        observations.append({"kind": "domain_neighborhood_mean_risk", "domain": u.domain, "mean_risk": dom_mean})

        if mode == "reference_based":
            if item["flipped"] or (item["shift"] >= 0.35 and score >= 0.45):
                status = "suspected"
            elif item["avg_mass"] < 0.35 or (item["ent2"] > 1.10 and not conflicts):
                status = "unknown"
            else:
                status = "no_evidence_of_damage"
        else:
            if conflicts or score >= 0.50:
                status = "suspected"
            elif item["avg_mass"] < 0.35 or item["ent2"] > 1.05:
                status = "unknown"
            else:
                status = "no_evidence_of_damage"

        reports.append(
            DamageReport(
                unit_id=u.unit_id,
                anomaly_score=float(score),
                suspected_paths=suspected_paths,
                supporting_observations=observations,
                conflicts=conflicts,
                reference_mode=mode,
                uncertainty=float(uncertainty),
                query_cost=cost_u,
                status=status,
            )
        )
    return reports
