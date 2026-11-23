"""Verify E7 selector coverage distinguishes attempted vs all strict-eligible denominators."""
from __future__ import annotations

from krmm.v3.pipeline_p1_p3 import evaluate_selector_comparisons


def _make_record(uid: str, attempted: bool, strict_ok: bool, cand_ok: bool, conf: float):
    rec = {
        "unit": {"unit_id": uid, "source_group": f"grp_{uid}"},
        "gold_evaluator_only": "A",
        "strict_eligible": True,
        "diagnostic_features": [1.0 if attempted else 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, conf, 0.2, 1.0, 1.0, 0.8],
        "reconstruction": {
            "unit_id": uid,
            "candidate_answer": ("A" if cand_ok else "B") if attempted else None,
            "candidate_source_type": "structured_single_hop" if attempted else "insufficient_or_conflicting",
            "atomic_facts": [],
            "support_paths": [],
            "ambiguity": 0.0,
            "conflict": False,
            "key_intervention_result": {"sensitive": True},
            "sham_intervention_result": {"stable": True},
            "abstain_reason": None if attempted else "insufficient_or_ambiguous_evidence",
            "query_cost": 4,
        },
        "reconstruction_aux": {"direct_candidate": "B", "generative_candidate": "A"},
        "repairs": {},
    }
    if attempted:
        rec["repairs"]["structured"] = {"strict_restoration": strict_ok}
    return rec


def test_selector_denominators_do_not_conflate_attempted_and_eligible():
    # 10 strict_eligible total: 6 attempted (4 strict success, 2 wrong), 4 abstained
    recs = [
        _make_record("u1", True, True, True, 0.95),
        _make_record("u2", True, True, True, 0.90),
        _make_record("u3", True, True, True, 0.85),
        _make_record("u4", True, True, True, 0.80),
        _make_record("u5", True, False, False, 0.30),
        _make_record("u6", True, False, False, 0.20),
        _make_record("u7", False, False, False, 0.50),
        _make_record("u8", False, False, False, 0.50),
        _make_record("u9", False, False, False, 0.50),
        _make_record("u10", False, False, False, 0.50),
    ]
    res = evaluate_selector_comparisons(recs, budget_fraction=0.5, branch="structured")
    assert res["n_strict_eligible"] == 10
    assert res["n_attempted"] == 6
    assert abs(res["candidate_coverage"] - 0.6) < 1e-9
    assert abs(res["abstention_rate_all_eligible"] - 0.4) < 1e-9

    dc = res["methods"]["direct_confidence"]
    assert dc["selected_count"] == 3
    assert abs(dc["selected_coverage_among_attempted"] - 0.5) < 1e-9
    assert abs(dc["selected_coverage_among_all_eligible"] - 0.3) < 1e-9
    assert abs(dc["strict_success_among_selected"] - 1.0) < 1e-9
    # Crucial: success over ALL strict-eligible is 3/10 = 0.3, NOT 3/6 = 0.5!
    assert abs(dc["strict_success_all_eligible"] - 0.3) < 1e-9
    assert dc["wrong_write_count_selected"] == 0
    assert dc["is_calibrated_probability"] is False
