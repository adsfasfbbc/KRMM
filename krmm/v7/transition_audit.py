"""Evaluator-only per-fact state transition audit: R_t, A_t, Y_strict, Y_active, H_new, H_ever, Forget_t, G (Section 3.1)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set


def compute_step_transition_metrics(
    degraded_ids: Set[str],
    committed_ids_by_step: Sequence[Set[str]],
    heldout_acc_by_step: Sequence[Dict[str, float]],  # index 0 = M_0, index t = M_t
    controls_by_step: Sequence[List[Dict[str, object]]],  # index 0 = M_0, index t = M_t
    fresh_reload_passed: bool = True,
    max_group_acc_drop: float = 0.05,
    min_token_agreement: float = 0.90,
) -> Dict[str, object]:
    """Compute exact per-fact trajectory across M_0 -> M_1 -> ... -> M_T without netting regressions against improvements."""
    assert len(heldout_acc_by_step) >= 1 and len(controls_by_step) == len(heldout_acc_by_step)
    T = len(heldout_acc_by_step) - 1
    D = len(degraded_ids)

    base_ctrls = controls_by_step[0]
    n_ctrls = len(base_ctrls)

    # Identify starting-correct immutable facts H (separately by control kind, excluding expected_ripple)
    start_correct_indices = [
        i for i, c in enumerate(base_ctrls)
        if bool(c.get("correct", False)) and c.get("kind") != "expected_ripple"
    ]
    start_wrong_indices = [
        i for i, c in enumerate(base_ctrls)
        if not bool(c.get("correct", False)) and c.get("kind") != "expected_ripple"
    ]

    ever_harmed_idx: Set[int] = set()
    first_harm_step: Dict[int, int] = {}
    ever_restored_targets: Set[str] = set()
    first_restore_step: Dict[str, int] = {}
    first_forget_step: Dict[str, int] = {}

    step_trajectory: List[Dict[str, object]] = []

    for t in range(T + 1):
        h_acc = heldout_acc_by_step[t]
        ctrls_t = controls_by_step[t]
        comm_t = committed_ids_by_step[t] if t < len(committed_ids_by_step) else set()

        restored_t = {uid for uid in degraded_ids if float(h_acc.get(uid, 0.0)) >= 0.999}
        active_t = restored_t & comm_t
        incidental_t = restored_t - comm_t

        # Track previous-repair forgetting (Forget_t)
        prev_restored = set(ever_restored_targets)
        forgotten_t = {uid for uid in prev_restored if uid not in restored_t}
        for uid in forgotten_t:
            if uid not in first_forget_step:
                first_forget_step[uid] = t
        for uid in restored_t:
            if uid not in first_restore_step:
                first_restore_step[uid] = t
            ever_restored_targets.add(uid)

        # Per-fact harm tracking (H_new at step t and cumulative H_ever)
        h_new_t_by_kind: Dict[str, int] = {}
        g_t_by_kind: Dict[str, int] = {}
        same_entity_regs_t: List[Dict[str, object]] = []

        for i in start_correct_indices:
            if not bool(ctrls_t[i].get("correct", False)):
                kind = str(base_ctrls[i].get("kind", "unknown"))
                h_new_t_by_kind[kind] = h_new_t_by_kind.get(kind, 0) + 1
                ever_harmed_idx.add(i)
                if i not in first_harm_step:
                    first_harm_step[i] = t
                if kind == "same_entity_other_relation":
                    same_entity_regs_t.append({
                        "fact_idx": i,
                        "entity_id": base_ctrls[i].get("entity_id"),
                        "question": base_ctrls[i].get("question"),
                        "first_harm_step": first_harm_step[i],
                    })

        for i in start_wrong_indices:
            if bool(ctrls_t[i].get("correct", False)):
                kind = str(base_ctrls[i].get("kind", "unknown"))
                g_t_by_kind[kind] = g_t_by_kind.get(kind, 0) + 1

        # Group-level token agreement & accuracy drop vs M_0
        by_group: Dict[str, Dict[str, object]] = {}
        all_groups_ok = True
        agree_all = 0
        for b, a in zip(base_ctrls, ctrls_t):
            kind = str(b.get("kind", "unknown"))
            if kind == "expected_ripple":
                continue
            g = by_group.setdefault(kind, {"n": 0, "b_ok": 0, "a_ok": 0, "c2w": 0, "w2c": 0, "agree": 0})
            g["n"] += 1
            bo, ao = bool(b.get("correct", False)), bool(a.get("correct", False))
            g["b_ok"] += int(bo)
            g["a_ok"] += int(ao)
            if bo and not ao:
                g["c2w"] += 1
            if not bo and ao:
                g["w2c"] += 1
            if b.get("pred") == a.get("pred"):
                g["agree"] += 1
                agree_all += 1

        for kind, g in by_group.items():
            n_g = max(1, g["n"])
            g["before_acc"] = g["b_ok"] / n_g
            g["after_acc"] = g["a_ok"] / n_g
            g["acc_drop"] = g["before_acc"] - g["after_acc"]
            g["token_agreement"] = g["agree"] / n_g
            # Explicit AND: same_entity_other_relation requires ZERO c2w (cannot be offset by w2c!)
            if kind == "same_entity_other_relation":
                g["passed"] = (g["c2w"] == 0)
            else:
                g["passed"] = (g["acc_drop"] <= max_group_acc_drop + 1e-9) and (g["token_agreement"] >= min_token_agreement - 1e-9)
            if not g["passed"]:
                all_groups_ok = False

        n_non_ripple = sum(g["n"] for g in by_group.values())
        overall_ag = (agree_all / n_non_ripple) if n_non_ripple > 0 else 1.0
        global_safety_pass = bool(all_groups_ok and len(same_entity_regs_t) == 0 and overall_ag >= min_token_agreement - 1e-9)

        R_t = (len(restored_t) / D) if D > 0 else None
        A_t = (len(active_t) / D) if D > 0 else None
        Forget_t = (len(forgotten_t) / len(prev_restored)) if prev_restored else 0.0

        step_trajectory.append({
            "step": t,
            "R_t": R_t,
            "A_t": A_t,
            "restored_count": len(restored_t),
            "active_count": len(active_t),
            "incidental_count": len(incidental_t),
            "Forget_t": Forget_t,
            "forgotten_targets": sorted(forgotten_t),
            "H_new_count": sum(h_new_t_by_kind.values()),
            "H_new_rate": (sum(h_new_t_by_kind.values()) / len(start_correct_indices)) if start_correct_indices else None,
            "H_new_by_kind": h_new_t_by_kind,
            "H_ever_count": len(ever_harmed_idx),
            "H_ever_rate": (len(ever_harmed_idx) / len(start_correct_indices)) if start_correct_indices else None,
            "G_improved_non_target": sum(g_t_by_kind.values()),
            "same_entity_regressions": same_entity_regs_t,
            "overall_token_agreement": overall_ag,
            "global_safety_pass": global_safety_pass,
            "by_group": by_group,
        })

    final = step_trajectory[-1]
    if D == 0:
        Y_strict: Optional[float] = None
        Y_active: Optional[float] = None
    else:
        ok_final = bool(final["global_safety_pass"] and fresh_reload_passed)
        Y_strict = (final["restored_count"] / D) if ok_final else 0.0
        Y_active = (final["active_count"] / D) if ok_final else 0.0

    return {
        "denominator_D": D,
        "n_start_correct_H": len(start_correct_indices),
        "n_steps": T,
        "fresh_reload_passed": bool(fresh_reload_passed),
        "Y_strict": Y_strict,
        "Y_active": Y_active,
        "R_final": final["R_t"],
        "A_final": final["A_t"],
        "H_new_final": final["H_new_count"],
        "H_new_rate_final": final["H_new_rate"],
        "H_ever_total": final["H_ever_count"],
        "H_ever_rate": final["H_ever_rate"],
        "G_final_non_target": final["G_improved_non_target"],
        "max_Forget_t": max(s["Forget_t"] for s in step_trajectory),
        "final_global_safety_pass": final["global_safety_pass"],
        "first_harm_by_fact": {
            str(i): {
                "step": st,
                "entity_id": base_ctrls[i].get("entity_id"),
                "kind": base_ctrls[i].get("kind"),
                "question": str(base_ctrls[i].get("question", "")).splitlines()[0],
            }
            for i, st in sorted(first_harm_step.items(), key=lambda kv: kv[1])
        },
        "first_restore_step": first_restore_step,
        "first_forget_step": first_forget_step,
        "trajectory": step_trajectory,
    }
