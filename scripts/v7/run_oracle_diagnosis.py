#!/usr/bin/env python3
"""KRMM v0.7 G0-Diagnosis (P0/Section 4.2): Matched Oracle D0, D1 (no-op 1/2/4/8x), D2 (single), D3 (seq step-by-step), D4 (joint)."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from krmm.v3.damage import balanced_wrong_assignments, inject_damage, restore_weights, snapshot_weights, state_sha256
from krmm.v3.knowledge_registry import AUDIT_GENERAL_CONTROLS, question_prompt, train_retain_prompts
from krmm.v3.mastery import HELDOUT_PERM, HELDOUT_VARIANTS, evaluate_unit_target, screen_clean_mastery
from krmm.v3.schemas import EvaluatorUnit
from krmm.v6.audit import run_fresh_subprocess_audit
from krmm.v6.schemas import PublicUnit
from krmm.v6.truth_audit import classify_paired_truth
from krmm.v7.continual_writer import fit_continual_step, fit_joint_multitarget
from krmm.v7.state_ledger import StateLedger, run_noop_lora_merge
from krmm.v7.transition_audit import compute_step_transition_metrics


def _eval_controls(backend, controls: List[dict]) -> List[dict]:
    out = []
    for c in controls:
        probs, mass = backend.probabilities(c["question"])
        pred = "ABCD"[int(np.argmax(probs))]
        out.append({
            "question": c["question"],
            "gold": c["gold"],
            "kind": c["kind"],
            "entity_id": c["entity_id"],
            "pred": pred,
            "correct": bool(pred == c["gold"]),
            "mass": float(mass),
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
        ku = pu.to_knowledge_unit("test")
        ev = evaluate_unit_target(backend, ku, gold, variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,))
        res[pu.unit_id] = float(ev["accuracy"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="KRMM v0.7 G0-Diagnosis (D0-D4)")
    ap.add_argument("--config", default="configs/v6/pilot.json")
    ap.add_argument("--out", default="results/v7-new/OW-replay")
    ap.add_argument("--max-targets", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    cfg = json.loads((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    out_dir = REPO_ROOT / args.out
    if out_dir.exists():
        raise FileExistsError(f"Output dir exists (never overwrite): {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    pub_manifest = json.loads((REPO_ROOT / cfg["public_manifest"]).read_text(encoding="utf-8"))
    hidden_manifest = json.loads((REPO_ROOT / cfg["evaluator_hidden"]).read_text(encoding="utf-8"))
    pub_units = [PublicUnit(**u) for u in pub_manifest["public_units"]]
    pub_map = {u.unit_id: u for u in pub_units}
    eval_map = {r["unit_id"]: r for r in hidden_manifest["evaluator_records"]}

    from krmm.model import Omni
    backend = Omni(cfg["model_path"])
    dmg_cfg = {**cfg["lora"], **cfg["damage_cfg"], "model_path": cfg["model_path"], "seed": seed}
    rep_cfg = {**cfg["lora"], **cfg["repair_cfg"], "model_path": cfg["model_path"], "seed": seed}

    retains = train_retain_prompts(include_multi_template=True)
    clean_teacher = backend.retain_logits(retains)

    # 1. Reconstruct exact v0.6 M_0 multi-target damaged checkpoint
    clean_screen: Dict[str, dict] = {}
    clean_held: Dict[str, float | None] = {}
    mastered: List[EvaluatorUnit] = []
    for pu in pub_units:
        er = eval_map[pu.unit_id]
        ku = pu.to_knowledge_unit("test")
        if er["gold_answer"] is None:
            clean_screen[pu.unit_id] = {"stably_mastered": False, "screen": {"accuracy": 0.0}}
            clean_held[pu.unit_id] = None
            continue
        ev_u = EvaluatorUnit(ku, er["gold_answer"], [], er["same_entity_controls"], er["pool_role"] != "known_real")
        m = screen_clean_mastery(backend, ev_u, min_choice_mass=0.25)
        h = evaluate_unit_target(backend, ku, er["gold_answer"], variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,))
        clean_screen[pu.unit_id] = m
        clean_held[pu.unit_id] = float(h["accuracy"])
        if bool(m["stably_mastered"]) and float(h["accuracy"]) >= 0.999 and er["pool_role"] == "known_real":
            mastered.append(ev_u)

    audit_controls: List[dict] = []
    for pu in pub_units:
        for sc in eval_map[pu.unit_id]["same_entity_controls"]:
            audit_controls.append(dict(sc))
    for g in AUDIT_GENERAL_CONTROLS:
        audit_controls.append({"question": g["question"], "gold": g["gold"], "kind": "general_commonsense", "entity_id": "__general__"})

    rng = random.Random(seed)
    by_dom: Dict[str, List[EvaluatorUnit]] = {}
    for ev_u in mastered:
        by_dom.setdefault(ev_u.unit.domain, []).append(ev_u)
    planned: List[EvaluatorUnit] = []
    for dom in sorted(by_dom.keys()):
        items = list(by_dom[dom])
        rng.shuffle(items)
        planned.extend(items[: max(1, int(round(len(items) * float(cfg.get("damage_fraction", 0.45)))))])
    rng.shuffle(planned)

    assign, _ = balanced_wrong_assignments(planned, seed=seed)
    planned_ids = {ev_u.unit.unit_id for ev_u in planned}
    for ev_u in planned:
        inject_damage(backend, ev_u.unit, ev_u.gold_answer, assign[ev_u.unit.unit_id], retains, clean_teacher, dmg_cfg, damage_type="D1_wrong_association")

    m0_state = snapshot_weights(backend, rep_cfg)
    m0_sha = state_sha256(m0_state)
    m0_pt = out_dir / "damaged_state_M0.pt"
    torch.save(m0_state, m0_pt)
    m0_retain_logits = backend.retain_logits(retains)

    # Compute exact paired truth & M_0 controls
    m0_controls = _eval_controls(backend, audit_controls)
    truth_records: List[dict] = []
    for pu in pub_units:
        er = eval_map[pu.unit_id]
        post_acc = None
        if er["gold_answer"] is not None:
            p_ev = evaluate_unit_target(backend, pu.to_knowledge_unit("test"), er["gold_answer"], variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,))
            post_acc = float(p_ev["accuracy"])
        cm = clean_screen[pu.unit_id]
        tr = classify_paired_truth(
            pu.unit_id, pu.entity_id, pu.domain, pu.source_group, er["pool_role"],
            pu.unit_id in planned_ids, pu.unit_id in planned_ids,
            bool(cm.get("stably_mastered", False)), float(cm.get("screen", {}).get("accuracy", 0.0)),
            clean_held[pu.unit_id], post_acc,
        )
        truth_records.append(tr.to_dict())

    degraded_uids = [r["unit_id"] for r in truth_records if r["is_degraded_target"]]
    degraded_set = set(degraded_uids)
    oracle_targets = degraded_uids[: args.max_targets]
    m0_held_D = _eval_heldout_map(backend, pub_units, eval_map, degraded_set)

    # Save shared M_0 evaluation state for run_writer_matrix.py reuse!
    (out_dir / "m0_meta.json").write_text(json.dumps({
        "m0_sha256": m0_sha,
        "truth_counts": dict(Counter(r["truth_category"] for r in truth_records)),
        "denominator_D": len(degraded_uids),
        "oracle_target_uids": oracle_targets,
        "oracle_target_names": [f"{pub_map[u].entity_id} ({pub_map[u].domain}, gold={eval_map[u]['gold_answer']})" for u in oracle_targets],
    }, indent=2), encoding="utf-8")
    (out_dir / "truth_records.json").write_text(json.dumps(truth_records, indent=2), encoding="utf-8")
    (out_dir / "audit_controls.json").write_text(json.dumps(audit_controls, indent=2), encoding="utf-8")
    (out_dir / "m0_controls.json").write_text(json.dumps(m0_controls, indent=2), encoding="utf-8")
    print(f"V7_M0_READY sha={m0_sha[:12]} D={len(degraded_uids)} oracle_targets={len(oracle_targets)}", flush=True)

    # --- D0: No-Update Control ---
    restore_weights(backend, m0_state)
    d0_controls = _eval_controls(backend, audit_controls)
    d0_held = _eval_heldout_map(backend, pub_units, eval_map, degraded_set)
    d0_metrics = compute_step_transition_metrics(degraded_set, [set()], [d0_held], [d0_controls], fresh_reload_passed=True)
    print(f"D0_NO_UPDATE H_new={d0_metrics['H_new_final']} agree={d0_metrics['trajectory'][0]['overall_token_agreement']:.3f}", flush=True)

    # --- D1: No-Op Merge 1x, 2x, 4x, 8x ---
    d1_results = {}
    for k_m in (1, 2, 4, 8):
        restore_weights(backend, m0_state)
        noop = run_noop_lora_merge(backend, rep_cfg, n_merges=k_m)
        c_after = _eval_controls(backend, audit_controls)
        h_after = _eval_heldout_map(backend, pub_units, eval_map, degraded_set)
        tr_m = compute_step_transition_metrics(degraded_set, [set(), set()], [m0_held_D, h_after], [m0_controls, c_after], fresh_reload_passed=True)
        d1_results[f"noop_{k_m}x"] = {**noop, "H_new_final": tr_m["H_new_final"], "overall_token_agreement": tr_m["trajectory"][-1]["overall_token_agreement"]}
        print(f"D1_NOOP_{k_m}x hash_eq={noop['hash_exact_match']} max_dw={noop['max_effective_delta_w']:.2e} max_wdiff={noop['max_abs_weight_diff']:.2e} H_new={tr_m['H_new_final']}", flush=True)

    # --- D2: Single-from-Initial (each of the 8 Oracle targets independently from M_0) ---
    d2_results = []
    for idx, uid in enumerate(oracle_targets, 1):
        restore_weights(backend, m0_state)
        pu = pub_map[uid]
        gold = eval_map[uid]["gold_answer"]
        ku = pu.to_knowledge_unit("test")
        ad_dir = out_dir / "D2_single_adapters" / f"{idx:02d}_{uid}"
        fit_info = fit_continual_step(backend, ku, gold, retains, m0_retain_logits, rep_cfg, ad_dir, writer_mode="W0_standard_kl")
        backend.model = backend.model.merge_and_unload()
        for p in backend.model.parameters():
            p.requires_grad_(False)
        backend.model.eval()
        h_s = _eval_heldout_map(backend, pub_units, eval_map, degraded_set)
        c_s = _eval_controls(backend, audit_controls)
        tr_s = compute_step_transition_metrics(degraded_set, [set(), {uid}], [m0_held_D, h_s], [m0_controls, c_s], fresh_reload_passed=True)
        d2_results.append({
            "idx": idx, "unit_id": uid, "entity_id": pu.entity_id, "domain": pu.domain, "gold": gold,
            "target_restored": h_s[uid] >= 0.999,
            "H_new_final": tr_s["H_new_final"],
            "same_entity_regs": tr_s["trajectory"][-1]["same_entity_regressions"],
            "effective_delta_w_frobenius": fit_info["effective_delta_w_frobenius"],
        })
        print(f"D2_SINGLE {idx}/8 {pu.entity_id:24s} restored={h_s[uid] >= 0.999} H_new={tr_s['H_new_final']} dw={fit_info['effective_delta_w_frobenius']:.3f}", flush=True)

    # --- D3: Sequential-v6 (step-by-step M_0 -> M_1 -> ... -> M_8) ---
    restore_weights(backend, m0_state)
    d3_held_steps = [m0_held_D]
    d3_ctrl_steps = [m0_controls]
    d3_comm_steps = [set()]
    d3_adapters: List[str] = []
    d3_step_logs = []
    comm_so_far = set()
    for idx, uid in enumerate(oracle_targets, 1):
        pu = pub_map[uid]
        gold = eval_map[uid]["gold_answer"]
        ku = pu.to_knowledge_unit("test")
        ad_dir = out_dir / "D3_seq_adapters" / f"step_{idx:02d}_{uid}"
        fit_info = fit_continual_step(backend, ku, gold, retains, m0_retain_logits, rep_cfg, ad_dir, writer_mode="W0_standard_kl")
        backend.model = backend.model.merge_and_unload()
        for p in backend.model.parameters():
            p.requires_grad_(False)
        backend.model.eval()
        d3_adapters.append(fit_info["adapter_path"])
        comm_so_far = set(comm_so_far) | {uid}
        d3_comm_steps.append(comm_so_far)
        h_t = _eval_heldout_map(backend, pub_units, eval_map, degraded_set)
        c_t = _eval_controls(backend, audit_controls)
        d3_held_steps.append(h_t)
        d3_ctrl_steps.append(c_t)
        step_tr = compute_step_transition_metrics(degraded_set, d3_comm_steps, d3_held_steps, d3_ctrl_steps, fresh_reload_passed=True)
        cur = step_tr["trajectory"][-1]
        d3_step_logs.append({"step": idx, "entity_id": pu.entity_id, "gold": gold, "dw_frob": fit_info["effective_delta_w_frobenius"], "R_t": cur["R_t"], "H_new": cur["H_new_count"], "H_ever": cur["H_ever_count"], "Forget_t": cur["Forget_t"], "se_regs": len(cur["same_entity_regressions"])})
        print(f"D3_SEQ step={idx}/8 {pu.entity_id:24s} R={cur['restored_count']}/{len(degraded_set)} H_new={cur['H_new_count']} H_ever={cur['H_ever_count']} Forget={cur['Forget_t']:.2f} se_regs={len(cur['same_entity_regressions'])}", flush=True)
    d3_full_metrics = compute_step_transition_metrics(degraded_set, d3_comm_steps, d3_held_steps, d3_ctrl_steps, fresh_reload_passed=True)

    # --- D4: Joint Multi-Target (matched 8*32 = 256 steps) ---
    restore_weights(backend, m0_state)
    joint_pairs = [(pub_map[u].to_knowledge_unit("test"), eval_map[u]["gold_answer"]) for u in oracle_targets]
    d4_ad_dir = out_dir / "D4_joint_adapter"
    d4_fit = fit_joint_multitarget(backend, joint_pairs, retains, m0_retain_logits, rep_cfg, d4_ad_dir)
    backend.model = backend.model.merge_and_unload()
    for p in backend.model.parameters():
        p.requires_grad_(False)
    backend.model.eval()
    d4_held = _eval_heldout_map(backend, pub_units, eval_map, degraded_set)
    d4_ctrl = _eval_controls(backend, audit_controls)
    d4_metrics = compute_step_transition_metrics(degraded_set, [set(), set(oracle_targets)], [m0_held_D, d4_held], [m0_controls, d4_ctrl], fresh_reload_passed=True)
    print(f"D4_JOINT R={d4_metrics['trajectory'][-1]['restored_count']}/{len(degraded_set)} H_new={d4_metrics['H_new_final']} Y_strict={d4_metrics['Y_strict']} dw={d4_fit['effective_delta_w_frobenius']:.3f}", flush=True)

    # Free GPU before isolated fresh-subprocess reload of D3 and D4!
    del backend
    torch.cuda.empty_cache()

    eval_units_spec = [{"unit": pu.to_knowledge_unit("test").to_dict(), "gold_answer": eval_map[pu.unit_id]["gold_answer"]} for pu in pub_units]
    fresh_checks = {}
    for label, ad_list in [("D3_sequential_v6", d3_adapters), ("D4_joint_multitarget", [d4_fit["adapter_path"]])]:
        jp = out_dir / f"fresh_job_{label}.json"
        op = out_dir / f"fresh_out_{label}.json"
        jp.write_text(json.dumps({
            "model_path": cfg["model_path"],
            "backend_kind": "omni",
            "damaged_state_path": str(m0_pt.resolve()),
            "expected_damaged_sha256": m0_sha,
            "adapter_dirs": ad_list,
            "consistency_probes": [],
            "prob_diff_tolerance": 1e-4,
            "eval_units": eval_units_spec,
            "audit_controls": audit_controls,
            "baseline_controls": m0_controls,
            "output": str(op.resolve()),
        }, indent=2), encoding="utf-8")
        fr = run_fresh_subprocess_audit(jp, python_exe=sys.executable)
        fresh_checks[label] = {
            "fresh_subprocess_ok": fr.get("fresh_subprocess_ok"),
            "fresh_reload_passed": fr.get("fresh_reload_passed"),
            "n_adapters_reloaded": fr.get("n_adapters_reloaded"),
            "strict_locality_pass": fr.get("locality_audit", {}).get("strict_locality_pass"),
            "same_entity_regressions": fr.get("locality_audit", {}).get("same_entity_regressions", []),
        }
        print(f"FRESH_CHECK {label}: ok={fr.get('fresh_subprocess_ok')} loc_pass={fresh_checks[label]['strict_locality_pass']} se_regs={len(fresh_checks[label]['same_entity_regressions'])}", flush=True)

    summary = {
        "m0_sha256": m0_sha,
        "denominator_D": len(degraded_uids),
        "oracle_targets": [pub_map[u].entity_id for u in oracle_targets],
        "D0_no_update": {"H_new_final": d0_metrics["H_new_final"], "overall_token_agreement": d0_metrics["trajectory"][0]["overall_token_agreement"]},
        "D1_noop_merge": d1_results,
        "D2_single_from_initial": d2_results,
        "D3_sequential_v6": {
            "Y_strict": d3_full_metrics["Y_strict"],
            "R_final": d3_full_metrics["R_final"],
            "H_new_final": d3_full_metrics["H_new_final"],
            "H_ever_total": d3_full_metrics["H_ever_total"],
            "G_final_non_target": d3_full_metrics["G_final_non_target"],
            "max_Forget_t": d3_full_metrics["max_Forget_t"],
            "first_harm_by_fact": d3_full_metrics["first_harm_by_fact"],
            "step_logs": d3_step_logs,
        },
        "D4_joint_multitarget": {
            "Y_strict": d4_metrics["Y_strict"],
            "R_final": d4_metrics["R_final"],
            "H_new_final": d4_metrics["H_new_final"],
            "G_final_non_target": d4_metrics["G_final_non_target"],
            "first_harm_by_fact": d4_metrics["first_harm_by_fact"],
            "fit_info": d4_fit,
        },
        "fresh_subprocess_checks": fresh_checks,
        "elapsed_seconds": time.time() - t0,
    }
    (out_dir / "diagnosis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"G0_DIAGNOSIS_COMPLETE elapsed={summary['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
