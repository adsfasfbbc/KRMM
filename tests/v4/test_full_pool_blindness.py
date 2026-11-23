"""Verify full-pool blind discovery & end-to-end metrics never filter unknown controls before ranking."""
from __future__ import annotations

from krmm.v4.metrics import compute_end_to_end_metrics, full_pool_discovery_metrics


def test_full_pool_discovery_includes_unknowns_in_ranking():
    reports = [
        {"unit_id": "d1", "anomaly_score": 0.8, "status": "suspected", "query_cost": 4},
        {"unit_id": "h1", "anomaly_score": 0.1, "status": "no_evidence_of_damage", "query_cost": 4},
        {"unit_id": "u1", "anomaly_score": 0.99, "status": "suspected", "query_cost": 4},
    ]
    truths = [
        {"unit_id": "d1", "pool_category": "damaged_mastered"},
        {"unit_id": "h1", "pool_category": "healthy_mastered"},
        {"unit_id": "u1", "pool_category": "originally_unknown"},
    ]
    m = full_pool_discovery_metrics(reports, truths)
    assert m["n_pool"] == 3
    assert m["full_pool_recall_at_k"] == 0.0
    assert m["originally_unknown_fpr"] == 1.0


def test_end_to_end_metrics_separate_healthy_harm_and_yield():
    outcomes = [
        {"pool_category": "damaged_mastered", "detected_suspected": True, "candidate_produced": True, "action": "ATTEMPT_PARAMETRIC_REPAIR", "candidate_matches_gold": True, "strict_parametric_recovery": True},
        {"pool_category": "damaged_mastered", "detected_suspected": True, "candidate_produced": False, "action": "REQUEST_EXTERNAL_EVIDENCE", "candidate_matches_gold": False, "strict_parametric_recovery": False, "oracle_repairable": True},
        {"pool_category": "healthy_mastered", "detected_suspected": True, "candidate_produced": True, "action": "ATTEMPT_PARAMETRIC_REPAIR", "candidate_matches_gold": False, "strict_parametric_recovery": False, "healthy_harmed": True},
    ]
    m = compute_end_to_end_metrics(outcomes, total_query_cost=12)
    assert m["end_to_end_strict_safe_repair_yield"] == 0.5
    assert m["healthy_unit_harm_rate"] == 1.0
    assert m["wrong_write_count"] == 1
    assert m["opportunity_cost_missed_repairable_count"] == 1
