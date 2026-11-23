"""End-to-end blind repair orchestrator for KRMM v0.5 (G1 -> G2-A -> G2-B -> G2-C)."""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

from krmm.v3.schemas import AtomicFact, KnowledgeUnit
from krmm.v4.metrics import compute_end_to_end_metrics
from .blind_discovery import CostTrackedProxy, evaluate_v5_discovery_metrics, run_v5_detector
from .selective_repair import select_candidate_for_strategy
from .support_structure import analyze_support_structure


def run_v5_end_to_end_mock(
    public_units: Sequence[KnowledgeUnit],
    backend,
    fact_extractor: Callable[[KnowledgeUnit], List[AtomicFact]],
    truth_records: Sequence[Dict[str, object]],
    gold_map: Dict[str, str],
    query_budget: int,
    strategy: str = "evidence_gated",
    detector_method: str = "full_m2",
) -> Dict[str, object]:
    """Run full blind end-to-end pipeline in CPU mock mode for fast verification."""
    proxy = CostTrackedProxy(backend=backend, query_budget=query_budget)
    disc = run_v5_detector(proxy, public_units, query_budget=query_budget, method=detector_method)
    disc_metrics = evaluate_v5_discovery_metrics(disc, truth_records)
    rep_map = {r["unit_id"]: r for r in disc["reports"]}
    tmap = {t["unit_id"]: t for t in truth_records}

    outcomes = []
    for u in public_units:
        dr = rep_map[u.unit_id]
        detected = dr["status"] == "suspected"
        if not detected:
            outcomes.append(
                {
                    "unit_id": u.unit_id,
                    "pool_category": tmap[u.unit_id]["pool_category"],
                    "detected_suspected": False,
                    "candidate_produced": False,
                    "action": "ABSTAIN",
                    "candidate_matches_gold": False,
                    "strict_parametric_recovery": False,
                    "healthy_harmed": False,
                }
            )
            continue
        facts = fact_extractor(u)
        sup = analyze_support_structure(u.domain, facts)
        sel = select_candidate_for_strategy(strategy, "B", "B", sup, oracle_gold=gold_map.get(u.unit_id))
        attempt = bool(sel["attempt"])
        cand = sel["candidate"]
        match = bool(cand == gold_map.get(u.unit_id))
        outcomes.append(
            {
                "unit_id": u.unit_id,
                "pool_category": tmap[u.unit_id]["pool_category"],
                "detected_suspected": True,
                "candidate_produced": sup.candidate_answer is not None,
                "action": "ATTEMPT_PARAMETRIC_REPAIR" if attempt else "ABSTAIN",
                "candidate_matches_gold": match,
                "strict_parametric_recovery": bool(attempt and match),
                "healthy_harmed": bool(attempt and tmap[u.unit_id]["pool_category"] == "healthy_mastered" and not match),
            }
        )

    return {
        "discovery_metrics": disc_metrics,
        "end_to_end_metrics": compute_end_to_end_metrics(outcomes, total_query_cost=disc["total_queries"]),
    }
