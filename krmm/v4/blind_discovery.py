"""Blind damage discovery with equal-budget comparators (Section 4.4, H1). Blind to truth & pool labels."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np

from krmm.v3.discovery import BudgetExhaustedError, BudgetedQueryProxy, detect
from krmm.v3.evidence import extract_atomic_facts
from krmm.v3.knowledge_registry import question_prompt
from krmm.v3.schemas import DamageReport, KnowledgeUnit
from .evidence_graph import EvidenceGraphAnalyzer


def _choose(probs: Sequence[float]) -> str:
    return "ABCD"[int(np.argmax(probs))]


def _entropy(probs: Sequence[float]) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def run_blind_detector(
    proxy: BudgetedQueryProxy,
    public_units: Sequence[KnowledgeUnit],
    query_budget: int,
    method: str = "full_m2",
) -> Dict[str, object]:
    """Run one of the 5 equal-budget M2 comparators without accessing pool truth or unknown_control tags."""
    if method not in {
        "direct_confidence",
        "self_consistency",
        "direct_paraphrases",
        "constraint_conflict_only",
        "full_m2",
    }:
        raise ValueError(f"Unsupported blind discovery method: {method}")

    if method == "full_m2":
        reports = detect(proxy, public_units, probe_budget=query_budget, mode="reference_free", constraint_probe_fraction=0.9)
        return {
            "method": method,
            "query_budget": int(query_budget),
            "total_queries": int(proxy.total_queries),
            "per_unit_queries": dict(proxy.per_unit_queries),
            "reports": [r.to_dict() for r in reports],
        }

    reports_out: List[Dict[str, object]] = []
    for u in public_units:
        proxy.set_current_unit(u.unit_id)
        cost_before = proxy.total_queries
        try:
            if method == "direct_confidence":
                p2, _, _ = question_prompt(u, variant=2)
                probs, mass = proxy.probabilities(p2)
                conf = float(np.max(probs))
                score = float(1.0 - conf)
                status = "suspected" if score >= 0.45 else ("unknown" if mass < 0.35 else "no_evidence_of_damage")
                reports_out.append(
                    DamageReport(
                        unit_id=u.unit_id,
                        anomaly_score=score,
                        suspected_paths=["low_direct_confidence"] if status == "suspected" else [],
                        supporting_observations=[{"kind": "direct_confidence", "conf": conf, "mass": mass}],
                        conflicts=[],
                        reference_mode="reference_free",
                        uncertainty=min(1.0, _entropy(probs) / np.log(4.0)),
                        query_cost=proxy.total_queries - cost_before,
                        status=status,
                    ).to_dict()
                )
            elif method in {"self_consistency", "direct_paraphrases"}:
                p2, _, _ = question_prompt(u, variant=2)
                probs2, mass2 = proxy.probabilities(p2)
                pred2 = _choose(probs2)
                p3, _, _ = question_prompt(u, variant=3)
                probs3, mass3 = proxy.probabilities(p3)
                pred3 = _choose(probs3)
                perm_disagree = 0.0
                if method == "direct_paraphrases" and proxy.total_queries < query_budget:
                    pp, d2c, _ = question_prompt(u, variant=2, permutation=("D", "C", "B", "A"))
                    raw_pp, _ = proxy.probabilities(pp)
                    perm_disagree = float(d2c[_choose(raw_pp)] != pred2)
                var_disagree = float(pred2 != pred3)
                score = min(1.0, 0.5 * var_disagree + 0.5 * perm_disagree) if method == "direct_paraphrases" else var_disagree
                status = "suspected" if score >= 0.5 else ("unknown" if (mass2 + mass3) / 2.0 < 0.35 else "no_evidence_of_damage")
                reports_out.append(
                    DamageReport(
                        unit_id=u.unit_id,
                        anomaly_score=float(score),
                        suspected_paths=["paraphrase_instability"] if status == "suspected" else [],
                        supporting_observations=[{"kind": method, "pred2": pred2, "pred3": pred3}],
                        conflicts=[],
                        reference_mode="reference_free",
                        uncertainty=min(1.0, _entropy(probs2) / np.log(4.0)),
                        query_cost=proxy.total_queries - cost_before,
                        status=status,
                    ).to_dict()
                )
            elif method == "constraint_conflict_only":
                facts, _ = extract_atomic_facts(proxy, u, max_new_tokens=24)
                assess = EvidenceGraphAnalyzer(u.domain).assess(facts)
                score = 0.90 if assess.conflict_status else 0.10
                status = "suspected" if assess.conflict_status else ("unknown" if not assess.identifiable else "no_evidence_of_damage")
                reports_out.append(
                    DamageReport(
                        unit_id=u.unit_id,
                        anomaly_score=score,
                        suspected_paths=["constraint_conflict"] if assess.conflict_status else [],
                        supporting_observations=[{"kind": "constraint_conflict_only", "category": assess.evidence_category}],
                        conflicts=[{"reason": "internal_conflict"}] if assess.conflict_status else [],
                        reference_mode="reference_free",
                        uncertainty=0.2 if assess.identifiable else 0.9,
                        query_cost=proxy.total_queries - cost_before,
                        status=status,
                    ).to_dict()
                )
        except BudgetExhaustedError:
            reports_out.append(
                DamageReport(
                    unit_id=u.unit_id,
                    anomaly_score=0.0,
                    suspected_paths=[],
                    supporting_observations=[{"kind": "budget_exhausted"}],
                    conflicts=[],
                    reference_mode="reference_free",
                    uncertainty=1.0,
                    query_cost=proxy.total_queries - cost_before,
                    status="unknown",
                ).to_dict()
            )

    if proxy.total_queries > int(query_budget):
        raise RuntimeError(f"Hard query budget exceeded: {proxy.total_queries} > {query_budget}")

    return {
        "method": method,
        "query_budget": int(query_budget),
        "total_queries": int(proxy.total_queries),
        "per_unit_queries": dict(proxy.per_unit_queries),
        "reports": reports_out,
    }
