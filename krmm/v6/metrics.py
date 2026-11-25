"""Primary endpoint (End-to-End Strict Safe Repair Yield Y) & strict per-fact locality metrics (Section 2.3)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence


def strict_per_fact_locality_audit(
    before_controls: Sequence[Dict[str, object]],
    after_controls: Sequence[Dict[str, object]],
    max_group_acc_drop: float = 0.05,
    min_token_agreement: float = 0.90,
) -> Dict[str, object]:
    """Explicit AND locality audit: zero tolerance for same-entity immutable regression; group drops & token agreement checked separately."""
    assert len(before_controls) == len(after_controls)
    by_kind: Dict[str, Dict[str, int]] = {}
    same_entity_regressions: List[Dict[str, object]] = []
    agree_count = 0

    for b, a in zip(before_controls, after_controls):
        kind = str(b.get("kind", "unknown"))
        g = by_kind.setdefault(kind, {"n": 0, "before_ok": 0, "after_ok": 0, "c2w": 0, "w2c": 0, "agree": 0})
        g["n"] += 1
        b_ok = bool(b.get("correct", False))
        a_ok = bool(a.get("correct", False))
        g["before_ok"] += int(b_ok)
        g["after_ok"] += int(a_ok)
        if b_ok and not a_ok:
            g["c2w"] += 1
            if kind == "same_entity_other_relation":
                same_entity_regressions.append({"entity_id": b.get("entity_id"), "question": b.get("question")})
        if not b_ok and a_ok:
            g["w2c"] += 1
        if b.get("pred") == a.get("pred"):
            g["agree"] += 1
            agree_count += 1

    group_summary: Dict[str, Dict[str, object]] = {}
    all_groups_ok = True
    for kind, g in by_kind.items():
        n = max(1, g["n"])
        b_acc = g["before_ok"] / n
        a_acc = g["after_ok"] / n
        drop = b_acc - a_acc
        ag = g["agree"] / n
        # For same_entity_other_relation: ZERO tolerance for any correct->wrong regression (cannot be offset by w2c!)
        if kind == "same_entity_other_relation":
            passed = (g["c2w"] == 0)
        else:
            passed = (drop <= max_group_acc_drop + 1e-9) and (ag >= min_token_agreement - 1e-9)
        if not passed:
            all_groups_ok = False
        group_summary[kind] = {
            "n": g["n"],
            "before_acc": b_acc,
            "after_acc": a_acc,
            "acc_drop": drop,
            "correct_to_wrong": g["c2w"],
            "wrong_to_correct": g["w2c"],
            "token_agreement": ag,
            "passed": passed,
        }

    overall_agree = agree_count / max(1, len(before_controls))
    strict_pass = bool(all_groups_ok and len(same_entity_regressions) == 0 and overall_agree >= min_token_agreement - 1e-9)
    return {
        "strict_locality_pass": strict_pass,
        "overall_token_agreement": overall_agree,
        "same_entity_zero_regression_pass": len(same_entity_regressions) == 0,
        "same_entity_regressions": same_entity_regressions,
        "by_group": group_summary,
    }


def compute_episode_metrics(
    truth_records: Sequence[Dict[str, object]],
    final_heldout_acc: Dict[str, float],
    step_records: Sequence[Dict[str, object]],
    locality_audit: Dict[str, object],
    fresh_reload_passed: bool,
    eval_gold_map: Dict[str, Optional[str]],
    eval_role_map: Dict[str, str],
) -> Dict[str, object]:
    """Compute Section 2.3 End-to-End Strict Safe Repair Yield (Y) and full risk vector."""
    degraded_ids = {r["unit_id"] for r in truth_records if r["is_degraded_target"]}
    healthy_ids = {r["unit_id"] for r in truth_records if r["truth_category"] == "original_mastered_healthy"}
    unknown_ids = {r["unit_id"] for r in truth_records if r["truth_category"] in ("previously_unmastered", "unverifiable")}
    n_pool = len(truth_records)
    D = len(degraded_ids)

    attempted_steps = [s for s in step_records if s["action"] == "ATTEMPT_REPAIR"]
    committed_steps = [s for s in attempted_steps if not s["rollback"]]
    committed_ids = {s["unit_id"] for s in committed_steps}

    # Global safety gate: fresh reload + strict per-fact locality
    global_safe = bool(fresh_reload_passed and locality_audit.get("strict_locality_pass", False))

    # Restored targets among D
    restored_D = [uid for uid in degraded_ids if float(final_heldout_acc.get(uid, 0.0)) >= 0.999]
    active_restored_D = [uid for uid in restored_D if uid in committed_ids]
    incidental_restored_D = [uid for uid in restored_D if uid not in committed_ids]

    # Harmed healthy units
    harmed_healthy_ids = [uid for uid in healthy_ids if float(final_heldout_acc.get(uid, 1.0)) < 0.999]

    # Risk accounting on attempted & committed writes
    n_att = len(attempted_steps)
    n_comm = len(committed_steps)
    wrong_cand_att = 0
    wrong_cand_comm = 0
    unknown_write_attempts = 0
    healthy_write_attempts = 0

    for s in attempted_steps:
        uid = s["unit_id"]
        gold = eval_gold_map.get(uid)
        role = eval_role_map.get(uid, "")
        if uid in unknown_ids or gold is None or role in ("unknown_fictional", "ambiguous"):
            unknown_write_attempts += 1
        elif uid in healthy_ids:
            healthy_write_attempts += 1
            if s["candidate"] != gold:
                wrong_cand_att += 1
                if not s["rollback"]:
                    wrong_cand_comm += 1
        else:
            if s["candidate"] != gold:
                wrong_cand_att += 1
                if not s["rollback"]:
                    wrong_cand_comm += 1

    # Primary endpoint Y: NA if D == 0; 0 safe yield flagged if global_safe is False
    if D == 0:
        Y_yield: Optional[float] = None
        Y_active: Optional[float] = None
    else:
        Y_yield = (len(restored_D) / D) if global_safe else 0.0
        Y_active = (len(active_restored_D) / D) if global_safe else 0.0

    return {
        "n_pool": n_pool,
        "denominator_D": D,
        "n_healthy_post_damage": len(healthy_ids),
        "n_unknown_or_unverifiable": len(unknown_ids),
        "global_safe_pass": global_safe,
        "fresh_reload_passed": bool(fresh_reload_passed),
        "strict_locality_pass": bool(locality_audit.get("strict_locality_pass", False)),
        "restored_in_D_raw": len(restored_D),
        "active_repaired_in_D": len(active_restored_D),
        "incidental_restored_in_D": len(incidental_restored_D),
        "end_to_end_strict_safe_yield_Y": Y_yield,
        "active_strict_safe_yield": Y_active,
        "n_attempted_writes": n_att,
        "n_committed_writes": n_comm,
        "n_rollbacks": sum(1 for s in attempted_steps if s["rollback"]),
        "n_abstentions": sum(1 for s in step_records if s["action"] == "ABSTAIN"),
        "n_request_evidence": sum(1 for s in step_records if s["action"] == "REQUEST_EVIDENCE"),
        "candidate_error_among_accepted": (wrong_cand_att / n_att) if n_att > 0 else None,
        "wrong_write_count_committed": wrong_cand_comm,
        "wrong_write_rate_attempted": (wrong_cand_att / n_att) if n_att > 0 else None,
        "wrong_write_rate_pool": wrong_cand_att / max(1, n_pool),
        "unknown_write_attempts": unknown_write_attempts,
        "healthy_write_attempts": healthy_write_attempts,
        "harmed_healthy_count": len(harmed_healthy_ids),
        "harmed_healthy_rate": len(harmed_healthy_ids) / max(1, len(healthy_ids)),
        "new_collateral_by_group": {
            k: v["correct_to_wrong"] for k, v in locality_audit.get("by_group", {}).items()
        },
        "total_train_steps": sum(int(s.get("train_steps", 0)) for s in step_records),
        "total_logical_queries": sum(int(s.get("logical_queries", 0)) for s in step_records),
        "total_physical_calls": sum(int(s.get("physical_calls", 0)) for s in step_records),
        "total_tokens": sum(int(s.get("tokens", 0)) for s in step_records),
    }
