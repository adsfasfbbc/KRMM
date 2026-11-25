#!/usr/bin/env python3
"""KRMM v0.7 P1 Writer Matrix (Section 5.3): Matched Oracle W0, W1, W2, W3, W4, W5 + T4 Public Transaction."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from krmm.v3.damage import restore_weights, snapshot_weights, state_sha256
from krmm.v3.knowledge_registry import train_retain_prompts
from krmm.v3.mastery import HELDOUT_PERM, HELDOUT_VARIANTS, evaluate_unit_target
from krmm.v6.audit import run_fresh_subprocess_audit
from krmm.v6.schemas import PublicUnit
from krmm.v7.continual_writer import fit_continual_step, fit_joint_multitarget
from krmm.v7.history_replay import build_history_replay_pairs
from krmm.v7.public_monitor import snapshot_public_monitor, verify_public_transaction
from krmm.v7.relation_retain import (
    EXTRA_GENERIC_RETAIN_PROMPTS,
    build_public_monitor_prompts,
    build_public_relation_retain_prompts,
)
from krmm.v7.state_ledger import LedgerEntry, StateLedger
from krmm.v7.transition_audit import compute_step_transition_metrics


def _eval_controls(backend, controls: List[dict]) -> List[dict]:
    out = []
    for c in controls:
        probs, mass = backend.probabilities(c["question"])
        pred = "ABCD"[int(np.argmax(probs))]
        out.append({
            "question": c["question"], "gold": c["gold"], "kind": c["kind"],
            "entity_id": c["entity_id"], "pred": pred, "correct": bool(pred == c["gold"]), "mass": float(mass),
        })
    return out


def _eval_heldout_map(backend, pub_units: List[PublicUnit], eval_map: Dict[str, dict], target_uids: set) -> Dict[str, float]:
    res: Dict[str, float] = {}
    for pu in pub_units:
        if pu.unit_id not in target_uids:
            continue
        gold = eval_map[pu.unit_id]["gold_answer"]
        if gold is None:
            res[pu.unit_id] = 0.0
            continue
        ev = evaluate_unit_target(backend, pu.to_knowledge_unit("test"), gold, variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,))
        res[pu.unit_id] = float(ev["accuracy"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="KRMM v0.7 P1 Writer Matrix (W0-W5 + T4)")
    ap.add_argument("--config", default="configs/v6/pilot.json")
    ap.add_argument("--diag-dir", default="results/v7-new/OW-replay")
    ap.add_argument("--out", default="results/v7-new/OW-writer-matrix")
    args = ap.parse_args()

    t0 = time.time()
    cfg = json.loads((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    diag_dir = REPO_ROOT / args.diag_dir
    out_dir = REPO_ROOT / args.out
    if out_dir.exists():
        raise FileExistsError(f"Output dir exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    pub_manifest = json.loads((REPO_ROOT / cfg["public_manifest"]).read_text(encoding="utf-8"))
    hidden_manifest = json.loads((REPO_ROOT / cfg["evaluator_hidden"]).read_text(encoding="utf-8"))
    pub_units = [PublicUnit(**u) for u in pub_manifest["public_units"]]
    pub_map = {u.unit_id: u for u in pub_units}
    eval_map = {r["unit_id"]: r for r in hidden_manifest["evaluator_records"]}

    m0_meta = json.loads((diag_dir / "m0_meta.json").read_text(encoding="utf-8"))
    truth_records = json.loads((diag_dir / "truth_records.json").read_text(encoding="utf-8"))
    audit_controls = json.loads((diag_dir / "audit_controls.json").read_text(encoding="utf-8"))
    m0_controls = json.loads((diag_dir / "m0_controls.json").read_text(encoding="utf-8"))

    m0_pt = diag_dir / "damaged_state_M0.pt"
    m0_state = torch.load(m0_pt, map_location="cpu", weights_only=True)
    m0_sha = state_sha256(m0_state)
    assert m0_sha == m0_meta["m0_sha256"], "M_0 SHA mismatch!"

    degraded_set = {r["unit_id"] for r in truth_records if r["is_degraded_target"]}
    oracle_targets = list(m0_meta["oracle_target_uids"])

    from krmm.model import Omni
    backend = Omni(cfg["model_path"])
    rep_cfg = {**cfg["lora"], **cfg["repair_cfg"], "model_path": cfg["model_path"], "seed": seed}
    retains = train_retain_prompts(include_multi_template=True)

    restore_weights(backend, m0_state)
    m0_retain_logits = backend.retain_logits(retains)
    m0_extra_generic_logits = backend.retain_logits(EXTRA_GENERIC_RETAIN_PROMPTS)
    m0_held_D = _eval_heldout_map(backend, pub_units, eval_map, degraded_set)

    # Pre-compute M_0 anchored relation retain teacher logits for each target (using strictly PUBLIC relation prompts!)
    rel_prompts_by_uid: Dict[str, List[str]] = {}
    rel_logits_by_uid: Dict[str, List[torch.Tensor]] = {}
    for uid in oracle_targets:
        rps = build_public_relation_retain_prompts(pub_map[uid], pub_units, max_prompts=8)
        rel_prompts_by_uid[uid] = rps
        rel_logits_by_uid[uid] = backend.retain_logits(rps)

    variants = [
        ("W0_standard_kl", False),
        ("W1_generic_matched", False),
        ("W2_relation_retain", False),
        ("W3_history_replay", False),
        ("W4_relation_plus_history", False),
        ("T4_W4_public_transaction", True),
    ]

    branch_summaries: Dict[str, dict] = {}
    branch_adapters: Dict[str, List[str]] = {}

    for label, use_tx in variants:
        wmode = "W4_relation_plus_history" if use_tx else label
        restore_weights(backend, m0_state)
        ledger = StateLedger(m0_state, rep_cfg)
        held_steps = [m0_held_D]
        ctrl_steps = [m0_controls]
        comm_steps = [set()]
        comm_set = set()
        ad_paths: List[str] = []
        step_details = []
        n_rollbacks = 0
        total_dw = 0.0
        total_train_steps = 0

        for idx, uid in enumerate(oracle_targets, 1):
            pu = pub_map[uid]
            gold = eval_map[uid]["gold_answer"]
            ku = pu.to_knowledge_unit("test")
            pre_state, pre_hash = ledger.snapshot(backend)

            # Public monitor snapshot BEFORE write (strictly disjoint from TRAIN_RETAIN & HIDDEN_AUDIT)
            mon_prompts = build_public_monitor_prompts(pu, pub_units, max_prompts=6)
            pre_mon = snapshot_public_monitor(backend, mon_prompts) if use_tx else []

            extra_p: List[str] = []
            extra_l: List[torch.Tensor] = []
            if wmode == "W1_generic_matched":
                extra_p = list(EXTRA_GENERIC_RETAIN_PROMPTS)
                extra_l = list(m0_extra_generic_logits)
            elif wmode in ("W2_relation_retain", "W4_relation_plus_history"):
                extra_p = rel_prompts_by_uid[uid]
                extra_l = rel_logits_by_uid[uid]

            hist_pairs = build_history_replay_pairs(ledger.committed_entries(), pub_map, max_pairs=8) if wmode in ("W3_history_replay", "W4_relation_plus_history") else []

            ad_dir = out_dir / label / "adapters" / f"step_{idx:02d}_{uid}"
            fit_info = fit_continual_step(
                backend, ku, gold, retains, m0_retain_logits, rep_cfg, ad_dir,
                writer_mode=wmode, extra_retains=extra_p, extra_teacher_logits=extra_l, history_pairs=hist_pairs,
            )
            total_train_steps += int(fit_info["train_steps"])

            rolled_back = False
            mon_rep = {"passed": True}
            if use_tx:
                mon_rep = verify_public_transaction(
                    backend, pu, gold, retains, m0_retain_logits, pre_mon,
                    ledger.committed_entries(), pub_map, min_retain_agreement=0.90, monitor_mode="relation_and_history",
                )
                if not mon_rep["passed"]:
                    rolled_back = True
                    n_rollbacks += 1
                    rb_ok = ledger.rollback_to_snapshot(backend, pre_state, pre_hash)
                    assert rb_ok, f"Rollback hash verification failed at step {idx}!"

            if not rolled_back:
                backend.model = backend.model.merge_and_unload()
                for p in backend.model.parameters():
                    p.requires_grad_(False)
                backend.model.eval()
                post_hash = ledger.current_hash(backend)
                ledger.entries.append(LedgerEntry(idx, uid, pu.entity_id, pu.domain, gold, True, False, fit_info["adapter_path"], fit_info["adapter_hash"], pre_hash, post_hash))
                ad_paths.append(fit_info["adapter_path"])
                comm_set = set(comm_set) | {uid}
                total_dw += float(fit_info["effective_delta_w_frobenius"])

            comm_steps.append(comm_set)
            held_steps.append(_eval_heldout_map(backend, pub_units, eval_map, degraded_set))
            ctrl_steps.append(_eval_controls(backend, audit_controls))
            step_details.append({
                "step": idx, "entity_id": pu.entity_id, "rolled_back": rolled_back,
                "dw_frob": fit_info["effective_delta_w_frobenius"], "monitor": mon_rep,
            })

        tr = compute_step_transition_metrics(degraded_set, comm_steps, held_steps, ctrl_steps, fresh_reload_passed=True)
        branch_adapters[label] = ad_paths
        branch_summaries[label] = {
            "n_committed": len(ad_paths),
            "n_rollbacks": n_rollbacks,
            "total_train_steps": total_train_steps,
            "sum_effective_delta_w_frobenius": total_dw,
            "inproc_metrics": {
                "Y_strict": tr["Y_strict"], "Y_active": tr["Y_active"],
                "R_final": tr["R_final"], "A_final": tr["A_final"],
                "restored_count": tr["trajectory"][-1]["restored_count"],
                "oracle_8_restored": sum(1 for u in oracle_targets if held_steps[-1].get(u, 0.0) >= 0.999),
                "H_new_final": tr["H_new_final"], "H_ever_total": tr["H_ever_total"],
                "G_final_non_target": tr["G_final_non_target"], "max_Forget_t": tr["max_Forget_t"],
                "final_global_safety_pass": tr["final_global_safety_pass"],
                "same_entity_regs_final": tr["trajectory"][-1]["same_entity_regressions"],
                "first_harm_by_fact": tr["first_harm_by_fact"],
            },
            "step_details": step_details,
        }
        im = branch_summaries[label]["inproc_metrics"]
        print(
            f"WRITER {label:26s} comm={len(ad_paths)}/8 rb={n_rollbacks} orc8={im['oracle_8_restored']}/8 "
            f"R={im['restored_count']}/{len(degraded_set)} H_new={im['H_new_final']} H_ever={im['H_ever_total']} "
            f"Forget={im['max_Forget_t']:.2f} Y_strict={im['Y_strict']} dw={total_dw:.2f}",
            flush=True,
        )

    # Also run W5_joint_with_relation (joint multi-target + all 8 targets' public relation retains)
    restore_weights(backend, m0_state)
    joint_pairs = [(pub_map[u].to_knowledge_unit("test"), eval_map[u]["gold_answer"]) for u in oracle_targets]
    all_rel_p = []
    all_rel_l = []
    seen_p = set()
    for uid in oracle_targets:
        for pr, lg in zip(rel_prompts_by_uid[uid], rel_logits_by_uid[uid]):
            if pr not in seen_p:
                seen_p.add(pr)
                all_rel_p.append(pr)
                all_rel_l.append(lg)
    w5_dir = out_dir / "W5_joint_with_relation" / "adapter"
    w5_fit = fit_joint_multitarget(backend, joint_pairs, retains, m0_retain_logits, rep_cfg, w5_dir, extra_retains=all_rel_p, extra_teacher_logits=all_rel_l)
    backend.model = backend.model.merge_and_unload()
    for p in backend.model.parameters():
        p.requires_grad_(False)
    backend.model.eval()
    w5_held = _eval_heldout_map(backend, pub_units, eval_map, degraded_set)
    w5_ctrl = _eval_controls(backend, audit_controls)
    w5_tr = compute_step_transition_metrics(degraded_set, [set(), set(oracle_targets)], [m0_held_D, w5_held], [m0_controls, w5_ctrl], fresh_reload_passed=True)
    branch_adapters["W5_joint_with_relation"] = [w5_fit["adapter_path"]]
    branch_summaries["W5_joint_with_relation"] = {
        "n_committed": 1, "n_rollbacks": 0, "total_train_steps": w5_fit["train_steps"],
        "sum_effective_delta_w_frobenius": w5_fit["effective_delta_w_frobenius"],
        "inproc_metrics": {
            "Y_strict": w5_tr["Y_strict"], "Y_active": w5_tr["Y_active"],
            "R_final": w5_tr["R_final"], "A_final": w5_tr["A_final"],
            "restored_count": w5_tr["trajectory"][-1]["restored_count"],
            "oracle_8_restored": sum(1 for u in oracle_targets if w5_held.get(u, 0.0) >= 0.999),
            "H_new_final": w5_tr["H_new_final"], "H_ever_total": w5_tr["H_ever_total"],
            "G_final_non_target": w5_tr["G_final_non_target"], "max_Forget_t": w5_tr["max_Forget_t"],
            "final_global_safety_pass": w5_tr["final_global_safety_pass"],
            "same_entity_regs_final": w5_tr["trajectory"][-1]["same_entity_regressions"],
            "first_harm_by_fact": w5_tr["first_harm_by_fact"],
        },
    }
    im5 = branch_summaries["W5_joint_with_relation"]["inproc_metrics"]
    print(f"WRITER {'W5_joint_with_relation':26s} orc8={im5['oracle_8_restored']}/8 R={im5['restored_count']}/{len(degraded_set)} H_new={im5['H_new_final']} Y_strict={im5['Y_strict']}", flush=True)

    # Free GPU before isolated fresh-subprocess reload of EVERY writer branch!
    del backend
    torch.cuda.empty_cache()

    eval_units_spec = [{"unit": pu.to_knowledge_unit("test").to_dict(), "gold_answer": eval_map[pu.unit_id]["gold_answer"]} for pu in pub_units]
    ow_pilot_winners = []
    for label, ad_list in branch_adapters.items():
        jp = out_dir / f"fresh_job_{label}.json"
        op = out_dir / f"fresh_out_{label}.json"
        jp.write_text(json.dumps({
            "model_path": cfg["model_path"], "backend_kind": "omni",
            "damaged_state_path": str(m0_pt.resolve()), "expected_damaged_sha256": m0_sha,
            "adapter_dirs": ad_list, "consistency_probes": [], "prob_diff_tolerance": 1e-4,
            "eval_units": eval_units_spec, "audit_controls": audit_controls, "baseline_controls": m0_controls,
            "output": str(op.resolve()),
        }, indent=2), encoding="utf-8")
        fr = run_fresh_subprocess_audit(jp, python_exe=sys.executable)
        la = fr.get("locality_audit", {})
        fh = fr.get("final_heldout_acc", {})
        orc8_fresh = sum(1 for u in oracle_targets if float(fh.get(u, 0.0)) >= 0.999)
        rest_D_fresh = sum(1 for u in degraded_set if float(fh.get(u, 0.0)) >= 0.999)
        loc_ok = bool(la.get("strict_locality_pass", False))
        fresh_ok = bool(fr.get("fresh_reload_passed", False))
        y_fresh = (rest_D_fresh / len(degraded_set)) if (loc_ok and fresh_ok) else 0.0
        se_regs = len(la.get("same_entity_regressions", []))
        branch_summaries[label]["fresh_subprocess"] = {
            "fresh_reload_passed": fresh_ok, "strict_locality_pass": loc_ok,
            "oracle_8_restored": orc8_fresh, "restored_D": rest_D_fresh,
            "same_entity_regressions_count": se_regs,
            "overall_token_agreement": la.get("overall_token_agreement"),
            "Y_strict_fresh": y_fresh,
        }
        if y_fresh > 0.0 and orc8_fresh >= 4 and se_regs == 0 and fresh_ok:
            ow_pilot_winners.append(label)
        print(f"FRESH_WRITER {label:26s} fresh={fresh_ok} loc={loc_ok} orc8={orc8_fresh}/8 restD={rest_D_fresh}/{len(degraded_set)} se_regs={se_regs} Y_fresh={y_fresh:.3f}", flush=True)

    final_out = {
        "m0_sha256": m0_sha,
        "denominator_D": len(degraded_set),
        "ow_pilot_pass": len(ow_pilot_winners) > 0,
        "ow_pilot_passing_writers": ow_pilot_winners,
        "branches": branch_summaries,
        "elapsed_seconds": time.time() - t0,
    }
    (out_dir / "writer_matrix_summary.json").write_text(json.dumps(final_out, indent=2), encoding="utf-8")
    print(f"WRITER_MATRIX_COMPLETE ow_pilot_pass={final_out['ow_pilot_pass']} winners={ow_pilot_winners} elapsed={final_out['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
