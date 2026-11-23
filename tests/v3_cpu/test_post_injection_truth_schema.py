"""Verify post-injection ground-truth schema separates planned, behavioral, local, and collateral damage."""
from __future__ import annotations

from krmm.v3.pipeline_p1_p3 import build_post_injection_truth_table, classify_post_injection_truth


def test_post_injection_truth_categories():
    rows = [
        # Planned, completed, degraded, local -> strict_local_damage
        classify_post_injection_truth("u1", True, True, post_accuracy=0.0, locality_pass=True),
        # Planned, completed, NOT degraded (injection failed) -> NOT behaviorally_degraded
        classify_post_injection_truth("u2", True, True, post_accuracy=1.0, locality_pass=True),
        # Planned, completed, degraded, locality failed -> degraded but NOT strict_local_damage
        classify_post_injection_truth("u3", True, True, post_accuracy=0.25, locality_pass=False),
        # NOT planned, degraded -> collateral_damage
        classify_post_injection_truth("u4", False, False, post_accuracy=0.25, locality_pass=None),
        # NOT planned, intact -> healthy
        classify_post_injection_truth("u5", False, False, post_accuracy=1.0, locality_pass=None),
    ]
    assert rows[0]["strict_local_damage"] is True and rows[0]["collateral_damage"] is False
    assert rows[1]["planned_damaged"] is True and rows[1]["behaviorally_degraded"] is False
    assert rows[2]["behaviorally_degraded"] is True and rows[2]["strict_local_damage"] is False
    assert rows[3]["collateral_damage"] is True and rows[3]["behaviorally_degraded"] is True
    assert rows[4]["behaviorally_degraded"] is False and rows[4]["collateral_damage"] is False

    tbl = build_post_injection_truth_table(rows)
    assert tbl["n_total"] == 5
    assert tbl["n_planned_damaged"] == 3
    assert tbl["n_behaviorally_degraded"] == 3
    assert tbl["n_strict_local_damage"] == 1
    assert tbl["n_collateral_damage"] == 1
