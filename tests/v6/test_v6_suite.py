"""KRMM v0.6 comprehensive CPU + subprocess smoke test suite (Section 3.2)."""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

from krmm.v6.audit import run_fresh_subprocess_audit
from krmm.v6.budget import BudgetExhaustedError, StrictBudgetProxy
from krmm.v6.detectors import V6_DETECTORS, evaluate_v6_discovery, run_v6_detector
from krmm.v6.metrics import compute_episode_metrics, strict_per_fact_locality_audit
from krmm.v6.policy import compute_risk_coverage_curve, select_action
from krmm.v6.truth_audit import classify_paired_truth, shuffle_public_pool
from krmm.v6.v6_catalog import TWIN_WORLD_PAIRS, build_v6_pilot_catalog, verify_v6_disjoint_from_history


def test_v6_catalog_100_percent_disjoint_from_v3_and_v5():
    res = verify_v6_disjoint_from_history()
    assert res["v6_unit_count"] == 50
    assert res["is_100_percent_disjoint"], f"Overlap found: {res['overlap_with_history']}"


def test_strict_public_vs_evaluator_isolation_and_no_oracle_gold_param():
    pub, _, _ = build_v6_pilot_catalog()
    for u in pub:
        d = u.to_public_dict()
        assert "gold_answer" not in d
        assert "pool_role" not in d
        assert "pool_category" not in d
    sig = inspect.signature(select_action)
    assert "oracle_gold" not in sig.parameters
    assert "gold" not in sig.parameters


def test_paired_truth_fixes_v5_clean_accuracy_zero_bug_and_blind_shuffling():
    tr_dmg = classify_paired_truth(
        "u1", "wolverine", "vertebrate_class", "sg1", "known_real",
        planned_damaged=True, injection_completed=True,
        clean_screen_passed=True, clean_screen_accuracy=1.0,
        clean_heldout_accuracy=1.0, post_heldout_accuracy=0.0,
    )
    assert tr_dmg.clean_heldout_accuracy == 1.0
    assert tr_dmg.heldout_delta == 1.0
    assert tr_dmg.truth_category == "effective_planned_damage"
    assert tr_dmg.is_degraded_target is True

    tr_col = classify_paired_truth(
        "u2", "meerkat", "vertebrate_class", "sg2", "known_real",
        planned_damaged=False, injection_completed=False,
        clean_screen_passed=True, clean_screen_accuracy=1.0,
        clean_heldout_accuracy=1.0, post_heldout_accuracy=0.33,
    )
    assert tr_col.truth_category == "collateral_damaged"
    assert tr_col.is_degraded_target is True

    pub, _, _ = build_v6_pilot_catalog()
    s1, h1 = shuffle_public_pool(pub, seed=2026)
    s2, h2 = shuffle_public_pool(pub, seed=2026)
    assert h1 == h2
    # Must not remain grouped in original order
    assert [u.unit_id for u in s1] != [u.unit_id for u in pub]


def test_strict_budget_proxy_accounting_and_hard_cap():
    class _MockModel:
        def probabilities(self, p):
            return [0.85, 0.05, 0.05, 0.05], 0.95
        def generate(self, p, max_new_tokens=24):
            return "fur and milk"

    proxy = StrictBudgetProxy(_MockModel(), query_budget=3, use_cache=True)
    proxy.probabilities("q1")
    proxy.probabilities("q1")  # cache hit
    proxy.probabilities("q2")
    acc = proxy.accounting()
    assert acc["logical_queries"] == 3
    assert acc["physical_calls"] == 2
    assert acc["cache_hits"] == 1
    assert acc["cold_cache_timing_valid"] is False
    try:
        proxy.probabilities("q3")
        assert False, "Expected BudgetExhaustedError"
    except BudgetExhaustedError:
        pass


def test_per_fact_locality_audit_cannot_offset_same_entity_regression():
    before = [
        {"kind": "same_entity_other_relation", "entity_id": "wolverine", "question": "q1", "pred": "A", "correct": True},
        {"kind": "same_entity_other_relation", "entity_id": "wolverine", "question": "q2", "pred": "B", "correct": False},
    ]
    # q1 regressed (True->False), q2 improved (False->True) -> mean acc unchanged, MUST STILL FAIL!
    after = [
        {"kind": "same_entity_other_relation", "entity_id": "wolverine", "question": "q1", "pred": "B", "correct": False},
        {"kind": "same_entity_other_relation", "entity_id": "wolverine", "question": "q2", "pred": "A", "correct": True},
    ]
    res = strict_per_fact_locality_audit(before, after)
    assert res["same_entity_zero_regression_pass"] is False
    assert res["strict_locality_pass"] is False


def test_episode_metrics_D_zero_yield_is_NA_and_twin_world_abstains():
    ep = compute_episode_metrics(
        truth_records=[], final_heldout_acc={}, step_records=[],
        locality_audit={"strict_locality_pass": True, "by_group": {}},
        fresh_reload_passed=True, eval_gold_map={}, eval_role_map={},
    )
    assert ep["denominator_D"] == 0
    assert ep["end_to_end_strict_safe_yield_Y"] is None  # NA, never 0 or 100%!

    # Risk-coverage unreachable grid -> None (NA), never fake 0
    rc = compute_risk_coverage_curve([{"unit_id": "u1", "candidate": None, "s": None, "candidate_is_wrong": True}], "s")
    assert rc["fixed_coverage_risk"]["50%"] is None
    assert len(TWIN_WORLD_PAIRS) == 2


def test_real_subprocess_reload_smoke_mock():
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        job_p = td_p / "job.json"
        out_p = td_p / "out.json"
        job = {
            "model_path": "mock",
            "backend_kind": "mock",
            "adapter_dirs": [],
            "consistency_probes": [{"prompt": "test", "expected_probs": [0.7, 0.1, 0.1, 0.1]}],
            "prob_diff_tolerance": 1e-4,
            "eval_units": [],
            "audit_controls": [],
            "baseline_controls": [],
            "output": str(out_p),
        }
        job_p.write_text(json.dumps(job), encoding="utf-8")
        res = run_fresh_subprocess_audit(job_p, python_exe=sys.executable)
        assert res["fresh_subprocess_ok"] is True
        assert res["fresh_reload_passed"] is True
        assert res["max_prob_diff"] == 0.0
