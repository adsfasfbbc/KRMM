"""Verify detection_metrics separates conditional known-only from full-pool (including unknown controls)."""
from __future__ import annotations

from krmm.v3.metrics import detection_metrics


def test_full_pool_penalizes_high_scoring_unknown_controls():
    # 2 damaged (scores 0.8, 0.7), 2 healthy (scores 0.1, 0.2), 2 unknown controls (scores 0.99, 0.98)
    reports = [
        {"anomaly_score": 0.80, "status": "suspected", "query_cost": 4},
        {"anomaly_score": 0.70, "status": "suspected", "query_cost": 4},
        {"anomaly_score": 0.10, "status": "no_evidence_of_damage", "query_cost": 4},
        {"anomaly_score": 0.20, "status": "no_evidence_of_damage", "query_cost": 4},
        {"anomaly_score": 0.99, "status": "suspected", "query_cost": 4},
        {"anomaly_score": 0.98, "status": "suspected", "query_cost": 4},
    ]
    is_damaged = [1, 1, 0, 0, 0, 0]
    is_unknown = [0, 0, 0, 0, 1, 1]

    m = detection_metrics(reports, is_damaged, is_unknown)
    # Conditional known-only ignores unknowns -> perfect 1.0 AUROC and 1.0 Recall@2
    assert m["conditional_known_only"]["auroc"] == 1.0
    assert m["conditional_known_only"]["recall_at_damaged_budget"] == 1.0
    # Full-pool ranks unknowns alongside known units -> top-2 are both unknowns!
    assert m["full_pool"]["recall_at_damaged_budget"] == 0.0
    assert m["full_pool"]["unknown_in_topk_count"] == 2
    assert m["full_pool"]["auroc"] < m["conditional_known_only"]["auroc"]
