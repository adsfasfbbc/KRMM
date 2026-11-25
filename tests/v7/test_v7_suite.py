"""KRMM v0.7 comprehensive unit test suite (Section 11 required assertions)."""
from __future__ import annotations

import inspect
import torch

from krmm.v6.v6_catalog import build_v6_pilot_catalog
from krmm.v7.continual_writer import fit_continual_step
from krmm.v7.history_replay import build_history_replay_pairs
from krmm.v7.public_monitor import verify_public_transaction
from krmm.v7.relation_retain import (
    EXTRA_GENERIC_RETAIN_PROMPTS,
    build_public_monitor_prompts,
    build_public_relation_retain_prompts,
)
from krmm.v7.state_ledger import LedgerEntry, compute_effective_delta_w_frobenius
from krmm.v7.transition_audit import compute_step_transition_metrics


def test_transition_no_netting_3_degraded_11_improved():
    """Explicit deterministic test from Section 3.2: 3 c2w + 11 w2c improves mean acc but MUST FAIL strict gate!"""
    base = []
    after = []
    # 3 starting-correct that degrade -> c2w = 3
    for i in range(3):
        base.append({"kind": "same_entity_other_relation", "entity_id": f"e{i}", "question": f"q{i}", "pred": "A", "correct": True})
        after.append({"kind": "same_entity_other_relation", "entity_id": f"e{i}", "question": f"q{i}", "pred": "D", "correct": False})
    # 11 starting-wrong that improve -> w2c = 11 (net accuracy jumps +8/20 = +40 pp!)
    for i in range(3, 14):
        base.append({"kind": "same_entity_other_relation", "entity_id": f"e{i}", "question": f"q{i}", "pred": "B", "correct": False})
        after.append({"kind": "same_entity_other_relation", "entity_id": f"e{i}", "question": f"q{i}", "pred": "A", "correct": True})
    # 6 stable correct
    for i in range(14, 20):
        base.append({"kind": "same_entity_other_relation", "entity_id": f"e{i}", "question": f"q{i}", "pred": "A", "correct": True})
        after.append({"kind": "same_entity_other_relation", "entity_id": f"e{i}", "question": f"q{i}", "pred": "A", "correct": True})

    res = compute_step_transition_metrics(
        degraded_ids={"u1"},
        committed_ids_by_step=[set(), {"u1"}],
        heldout_acc_by_step=[{"u1": 0.0}, {"u1": 1.0}],
        controls_by_step=[base, after],
        fresh_reload_passed=True,
    )
    assert res["H_new_final"] == 3
    assert res["G_final_non_target"] == 11
    assert res["final_global_safety_pass"] is False
    assert res["Y_strict"] == 0.0
    assert res["R_final"] == 1.0


def test_empty_denominator_and_H_ever_vs_H_new():
    # D = 0 -> Y_strict = None (NA), never 0 or 100%
    res0 = compute_step_transition_metrics(
        degraded_ids=set(),
        committed_ids_by_step=[set()],
        heldout_acc_by_step=[{}],
        controls_by_step=[[]],
        fresh_reload_passed=True,
    )
    assert res0["Y_strict"] is None
    assert res0["H_new_rate_final"] is None

    # H_ever retains intermediate step-1 regression even if recovered at step-2!
    c0 = [{"kind": "same_entity_other_relation", "entity_id": "e0", "question": "q0", "pred": "A", "correct": True}]
    c1 = [{"kind": "same_entity_other_relation", "entity_id": "e0", "question": "q0", "pred": "D", "correct": False}]
    c2 = [{"kind": "same_entity_other_relation", "entity_id": "e0", "question": "q0", "pred": "A", "correct": True}]
    res = compute_step_transition_metrics(
        degraded_ids={"u1"},
        committed_ids_by_step=[set(), {"u1"}, {"u1"}],
        heldout_acc_by_step=[{"u1": 0.0}, {"u1": 1.0}, {"u1": 1.0}],
        controls_by_step=[c0, c1, c2],
        fresh_reload_passed=True,
    )
    assert res["H_new_final"] == 0
    assert res["H_ever_total"] == 1
    assert res["first_harm_by_fact"]["0"]["step"] == 1


def test_previous_repair_forgetting_and_active_vs_incidental_Y():
    c = [{"kind": "general_commonsense", "entity_id": "__g__", "question": "q", "pred": "A", "correct": True}]
    res = compute_step_transition_metrics(
        degraded_ids={"u1", "u2"},
        committed_ids_by_step=[set(), {"u1"}, {"u1"}],
        # step 1 restores u1; step 2 forgets u1 and incidentally restores u2!
        heldout_acc_by_step=[{"u1": 0.0, "u2": 0.0}, {"u1": 1.0, "u2": 0.0}, {"u1": 0.0, "u2": 1.0}],
        controls_by_step=[c, c, c],
        fresh_reload_passed=True,
    )
    assert res["max_Forget_t"] == 1.0
    assert res["first_forget_step"]["u1"] == 2
    assert res["Y_strict"] == 0.5  # 1/2 restored overall
    assert res["Y_active"] == 0.0  # 0/2 actively committed and restored!


def test_expected_ripple_excluded_from_immutable_harm():
    c0 = [{"kind": "expected_ripple", "entity_id": "e0", "question": "rq", "pred": "A", "correct": True}]
    c1 = [{"kind": "expected_ripple", "entity_id": "e0", "question": "rq", "pred": "B", "correct": False}]
    res = compute_step_transition_metrics(
        degraded_ids={"u1"},
        committed_ids_by_step=[set(), {"u1"}],
        heldout_acc_by_step=[{"u1": 0.0}, {"u1": 1.0}],
        controls_by_step=[c0, c1],
        fresh_reload_passed=True,
    )
    assert res["H_new_final"] == 0
    assert res["final_global_safety_pass"] is True
    assert res["Y_strict"] == 1.0


def test_train_monitor_audit_prompt_separation_and_no_gold_in_public_apis():
    pub, evs, _ = build_v6_pilot_catalog()
    hidden_qs = {sc["question"] for er in evs for sc in er.same_entity_controls}
    for u in pub[:6]:
        tr_ps = set(build_public_relation_retain_prompts(u, pub, max_prompts=8))
        mon_ps = set(build_public_monitor_prompts(u, pub, max_prompts=6))
        assert tr_ps.isdisjoint(mon_ps), "TRAIN_RETAIN and PUBLIC_MONITOR overlap!"
        assert tr_ps.isdisjoint(hidden_qs), "TRAIN_RETAIN overlaps HIDDEN_AUDIT!"
        assert mon_ps.isdisjoint(hidden_qs), "PUBLIC_MONITOR overlaps HIDDEN_AUDIT!"
    assert set(EXTRA_GENERIC_RETAIN_PROMPTS).isdisjoint(hidden_qs)

    for fn in (fit_continual_step, verify_public_transaction, build_history_replay_pairs):
        params = inspect.signature(fn).parameters
        assert "gold" not in params and "oracle_gold" not in params and "gold_map" not in params


def test_history_provisional_excludes_conflicted_and_effective_delta_w():
    pub, _, _ = build_v6_pilot_catalog()
    pmap = {u.unit_id: u for u in pub}
    entries = [
        LedgerEntry(1, pub[0].unit_id, pub[0].entity_id, pub[0].domain, "A", provisional=True, conflicted=False),
        LedgerEntry(2, pub[1].unit_id, pub[1].entity_id, pub[1].domain, "B", provisional=True, conflicted=True),
    ]
    pairs = build_history_replay_pairs(entries, pmap, max_pairs=8)
    assert len(pairs) == 1

    # Verify effective DeltaW = scaling * B @ A norm
    class _FakeMod(torch.nn.Module):
        def __init__(self, b_val: float):
            super().__init__()
            self.lora_A = {"default": torch.nn.Linear(4, 2, bias=False)}
            self.lora_B = {"default": torch.nn.Linear(2, 4, bias=False)}
            with torch.no_grad():
                self.lora_A["default"].weight.fill_(1.0)
                self.lora_B["default"].weight.fill_(b_val)
            self.scaling = {"default": 2.0}

    z = compute_effective_delta_w_frobenius(_FakeMod(0.0))
    nz = compute_effective_delta_w_frobenius(_FakeMod(0.5))
    assert z["effective_delta_w_frobenius"] == 0.0
    assert nz["effective_delta_w_frobenius"] > 0.0
