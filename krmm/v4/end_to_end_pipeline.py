"""End-to-end blind discovery -> evidence assessment -> selective repair -> audit pipeline (Section 4.1)."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from krmm.v3.discovery import BudgetedQueryProxy
from krmm.v3.schemas import AtomicFact, KnowledgeUnit
from .blind_discovery import run_blind_detector
from .evidence_graph import EvidenceGraphAnalyzer
from .metrics import compute_end_to_end_metrics, full_pool_discovery_metrics
from .repair_adapter import RepairAdapter
from .selective_policy import decide_repair


def run_blind_end_to_end(
    public_units: Sequence[KnowledgeUnit],
    model_backend,
    fact_extractor: Callable[[KnowledgeUnit], List[AtomicFact]],
    discovery_budget: int,
    repair_budget: int,
    discovery_method: str = "full_m2",
    write_strategy: str = "standard_kl",
    calibrated_policy: Optional[Dict[str, object]] = None,
    evaluator_auditor: Optional[Callable[[str, str, Dict[str, object]], Dict[str, object]]] = None,
    evaluator_truth_records: Optional[Sequence[Dict[str, object]]] = None,
    mock_repair_executor=None,
) -> Dict[str, object]:
    """Execute full blind pipeline; evaluator_truth_records is ONLY touched in isolated post-hoc scoring."""
    # Stage 1: Blind Discovery on complete mixed pool (no truth or unknown_control filtering)
    proxy = BudgetedQueryProxy(backend=model_backend, query_budget=discovery_budget)
    disc_res = run_blind_detector(proxy, public_units, query_budget=discovery_budget, method=discovery_method)
    reports_by_id = {r["unit_id"]: r for r in disc_res["reports"]}

    # Stage 2-5: For each suspected unit, extract evidence -> assess -> decide -> repair
    adapter = RepairAdapter(strategy=write_strategy, mock_mode=True)
    rem_repairs = int(repair_budget)
    method_trace: List[Dict[str, object]] = []

    for u in public_units:
        dmg_rep = reports_by_id[u.unit_id]
        if dmg_rep["status"] != "suspected":
            method_trace.append(
                {
                    "unit_id": u.unit_id,
                    "detected_suspected": False,
                    "candidate_produced": False,
                    "action": "ABSTAIN",
                    "reason": f"not_suspected_{dmg_rep['status']}",
                    "candidate_answer": None,
                    "evidence_category": None,
                }
            )
            continue

        facts = fact_extractor(u)
        assess = EvidenceGraphAnalyzer(u.domain, study_condition="internal_only").assess(facts)
        decision = decide_repair(
            dmg_rep,
            assess,
            repair_budget={"remaining_repairs": rem_repairs, "allow_nonparametric": True},
            calibrated_policy=calibrated_policy,
        )
        action = decision["action"]
        repair_out = None
        if action == "ATTEMPT_PARAMETRIC_REPAIR" and assess.candidate_answer is not None:
            rem_repairs = max(0, rem_repairs - 1)
            repair_out = adapter.apply_repair(u.unit_id, assess.candidate_answer, mock_executor=mock_repair_executor)

        method_trace.append(
            {
                "unit_id": u.unit_id,
                "detected_suspected": True,
                "candidate_produced": assess.candidate_answer is not None,
                "candidate_answer": assess.candidate_answer,
                "evidence_category": assess.evidence_category,
                "estimated_reliability": assess.estimated_candidate_reliability,
                "action": action,
                "reason": decision["reason"],
                "repair_out": repair_out,
            }
        )

    out: Dict[str, object] = {
        "discovery": {
            "method": discovery_method,
            "query_budget": discovery_budget,
            "total_queries": disc_res["total_queries"],
        },
        "method_trace": method_trace,
    }

    # Isolated Evaluator-Only Post-Hoc Scoring
    if evaluator_truth_records is not None and evaluator_auditor is not None:
        truth_by_id = {t["unit_id"]: t for t in evaluator_truth_records}
        scored_rows = []
        for tr in method_trace:
            uid = tr["unit_id"]
            tinfo = truth_by_id[uid]
            audit_res = evaluator_auditor(uid, tr["action"], tr)
            scored_rows.append(
                {
                    **tr,
                    "pool_category": tinfo["pool_category"],
                    "candidate_matches_gold": bool(audit_res.get("candidate_matches_gold", False)),
                    "strict_parametric_recovery": bool(audit_res.get("strict_parametric_recovery", False)),
                    "healthy_harmed": bool(audit_res.get("healthy_harmed", False)),
                    "oracle_repairable": bool(audit_res.get("oracle_repairable", True)),
                }
            )
        out["evaluator_discovery_metrics"] = full_pool_discovery_metrics(disc_res["reports"], evaluator_truth_records)
        out["evaluator_end_to_end_metrics"] = compute_end_to_end_metrics(
            scored_rows, total_query_cost=disc_res["total_queries"]
        )

    return out
