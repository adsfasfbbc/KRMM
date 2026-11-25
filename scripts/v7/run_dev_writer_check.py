#!/usr/bin/env python3
"""Out-of-sample DEV-V7 Oracle Writer Confirmation (OW-dev): W0 vs W1 vs W2 vs W4 + fresh-subprocess reload."""
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
from krmm.v3.knowledge_registry import AUDIT_GENERAL_CONTROLS, train_retain_prompts
from krmm.v3.mastery import HELDOUT_PERM, HELDOUT_VARIANTS, evaluate_unit_target, screen_clean_mastery
from krmm.v3.schemas import EvaluatorUnit
from krmm.v6.audit import run_fresh_subprocess_audit
from krmm.v6.truth_audit import classify_paired_truth
from krmm.v7.continual_writer import fit_continual_step
from krmm.v7.dev_catalog import build_dev_v7_catalog, verify_dev_v7_disjoint_from_all_242
from krmm.v7.history_replay import build_history_replay_pairs
from krmm.v7.relation_retain import EXTRA_GENERIC_RETAIN_PROMPTS, build_public_relation_retain_prompts
from krmm.v7.state_ledger import LedgerEntry, StateLedger
from krmm.v7.transition_audit import compute_step_transition_metrics


def _eval_controls(backend, controls: List[dict]) -> List[dict]:
    out = []
    for c in controls:
        probs, mass = backend.probabilities(c["question"])
        pred = "ABCD"[int(np.argmax(probs))]
        out.append({"question": c["question"], "gold": c["gold"], "kind": c["kind"], "entity_id": c["entity_id"], "pred": pred, "correct": bool(pred == c["gold"]), "mass": float(mass)})
    return out


def _eval_heldout_map(backend, pub_units, eval_map, target_uids: set) -> Dict[str, float]:
    res = {}
    for pu in pub_units:
        if pu.unit_id not in target_uids:
            continue
        ev = evaluate_unit_target(backend, pu.to_knowledge_unit("test"), eval_map[pu.unit_id].gold_answer, variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,))
        res[pu.unit_id] = float(ev["accuracy"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v6/pilot.json")
    ap.add_argument("--out", default="results/v7-new/OW-dev-v7")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    t0 = time.time()
    disj = verify_dev_v7_disjoint_from_all_242()
    assert disj["is_disjoint"], f"DEV-V7 overlap: {disj['overlap']}"

    cfg = json.loads((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(args.seed)
    out_dir = REPO_ROOT / args.out
    if out_dir.exists():
        raise FileExistsError(f"Output exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    pub_units, eval_recs = build_dev_v7_catalog()
    pub_map = {u.unit_id: u for u in pub_units}
    eval_map = {r.unit_id: r for r in eval_recs}

    from krmm.model import Omni
    backend = Omni(cfg["model_path"])
    dmg_cfg = {**cfg["lora"], **cfg["damage_cfg"], "model_path": cfg["model_path"], "seed": seed}
    rep_cfg = {**cfg["lora"], **cfg["repair_cfg"], "model_path": cfg["model_path"], "seed": seed}

    retains = train_retain_prompts(include_multi_template=True)
    clean_teacher = backend.retain_logits(retains)

    clean_screen, clean_held, mastered = {}, {}, []
    for pu in pub_units:
        er = eval_map[pu.unit_id]
        ku = pu.to_knowledge_unit("test")
        ev_u = EvaluatorUnit(ku, er.gold_answer, [], er.same_entity_controls, False)
        m = screen_clean_mastery(backend, ev_u, min_choice_mass=0.25)
        h = evaluate_unit_target(backend, ku, er.gold_answer, variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,))
        clean_screen[pu.unit_id] = m
        clean_held[pu.unit_id] = float(h["accuracy"])
        if bool(m["stably_mastered"]) and float(h["accuracy"]) >= 0.999:
            mastered.append(ev_u)

    audit_controls = [dict(sc) for er in eval_recs for sc in er.same_entity_controls]
    for g in AUDIT_GENERAL_CONTROLS:
        audit_controls.append({"question": g["question"], "gold": g["gold"], "kind": "general_commonsense", "entity_id": "__general__"})

    rng = random.Random(seed)
    by_dom: Dict[str, List[EvaluatorUnit]] = {}
    for ev_u in mastered:
        by_dom.setdefault(ev_u.unit.domain, []).append(ev_u)
    planned = []
    for dom in sorted(by_dom.keys()):
        items = list(by_dom[dom])
        rng.shuffle(items)
        planned.extend(items[: max(1, int(round(len(items) * 0.45)))])
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
    m0_extra_generic_logits = backend.retain_logits(EXTRA_GENERIC_RETAIN_PROMPTS)
    m0_controls = _eval_controls(backend, audit_controls)

    truth_records = []
    for pu in pub_units:
        er = eval_map[pu.unit_id]
        p_ev = evaluate_unit_target(backend, pu.to_knowledge_unit("test"), er.gold_answer, variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,))
        cm = clean_screen[pu.unit_id]
        tr = classify_paired_truth(
            pu.unit_id, pu.entity_id, pu.domain, pu.source_group, "known_real",
            pu.unit_id in planned_ids, pu.unit_id in planned_ids,
            bool(cm.get("stably_mastered", False)), float(cm.get("screen", {}).get("accuracy", 0.0)),
            clean_held[pu.unit_id], float(p_ev["accuracy"]),
        )
        truth_records.append(tr.to_dict())

    degraded_uids = [r["unit_id"] for r in truth_records if r["is_degraded_target"]]
    degraded_set = set(degraded_uids)
    oracle_targets = degraded_uids[:8]
    m0_held_D = _eval_heldout_map(backend, pub_units, eval_map, degraded_set)
    print(f"DEV7_M0_READY mastered={len(mastered)}/36 D={len(degraded_uids)} targets={len(oracle_targets)} sha={m0_sha[:12]}", flush=True)

    rel_prompts_by_uid = {u: build_public_relation_retain_prompts(pub_map[u], pub_units, max_prompts=8) for u in oracle_targets}
    rel_logits_by_uid = {u: backend.retain_logits(rel_prompts_by_uid[u]) for u in oracle_targets}

    branches = {}
    ad_map = {}
    for wmode in ("W0_standard_kl", "W1_generic_matched", "W2_relation_retain", "W4_relation_plus_history"):
        restore_weights(backend, m0_state)
        ledger = StateLedger(m0_state, rep_cfg)
        held_steps, ctrl_steps, comm_steps = [m0_held_D], [m0_controls], [set()]
        comm_set, ad_paths, total_dw = set(), [], 0.0
        for idx, uid in enumerate(oracle_targets, 1):
            pu = pub_map[uid]
            gold = eval_map[uid].gold_answer
            extra_p = list(EXTRA_GENERIC_RETAIN_PROMPTS) if wmode == "W1_generic_matched" else (rel_prompts_by_uid[uid] if wmode in ("W2_relation_retain", "W4_relation_plus_history") else [])
            extra_l = list(m0_extra_generic_logits) if wmode == "W1_generic_matched" else (rel_logits_by_uid[uid] if wmode in ("W2_relation_retain", "W4_relation_plus_history") else [])
            hist_p = build_history_replay_pairs(ledger.committed_entries(), pub_map, max_pairs=8) if wmode == "W4_relation_plus_history" else []
            ad_dir = out_dir / wmode / "adapters" / f"step_{idx:02d}_{uid}"
            fit = fit_continual_step(backend, pu.to_knowledge_unit("test"), gold, retains, m0_retain_logits, rep_cfg, ad_dir, writer_mode=wmode, extra_retains=extra_p, extra_teacher_logits=extra_l, history_pairs=hist_p)
            backend.model = backend.model.merge_and_unload()
            for p in backend.model.parameters():
                p.requires_grad_(False)
            backend.model.eval()
            ledger.entries.append(LedgerEntry(idx, uid, pu.entity_id, pu.domain, gold, True, False, fit["adapter_path"], fit["adapter_hash"]))
            ad_paths.append(fit["adapter_path"])
            comm_set = set(comm_set) | {uid}
            total_dw += float(fit["effective_delta_w_frobenius"])
            comm_steps.append(comm_set)
            held_steps.append(_eval_heldout_map(backend, pub_units, eval_map, degraded_set))
            ctrl_steps.append(_eval_controls(backend, audit_controls))
        tr = compute_step_transition_metrics(degraded_set, comm_steps, held_steps, ctrl_steps, fresh_reload_passed=True)
        ad_map[wmode] = ad_paths
        branches[wmode] = {
            "sum_dw_frob": total_dw,
            "oracle_8_restored": sum(1 for u in oracle_targets if held_steps[-1].get(u, 0.0) >= 0.999),
            "R_final": tr["R_final"], "restored_D": tr["trajectory"][-1]["restored_count"],
            "H_new": tr["H_new_final"], "H_ever": tr["H_ever_total"],
            "Forget_max": tr["max_Forget_t"], "Y_strict_inproc": tr["Y_strict"],
        }
        print(f"DEV7_WRITER {wmode:26s} orc={branches[wmode]['oracle_8_restored']}/{len(oracle_targets)} R={branches[wmode]['restored_D']}/{len(degraded_set)} H_new={tr['H_new_final']} H_ever={tr['H_ever_total']} Y={tr['Y_strict']}", flush=True)

    del backend
    torch.cuda.empty_cache()

    eval_units_spec = [{"unit": pu.to_knowledge_unit("test").to_dict(), "gold_answer": eval_map[pu.unit_id].gold_answer} for pu in pub_units]
    for wmode, ad_paths in ad_map.items():
        jp = out_dir / f"fresh_job_{wmode}.json"
        op = out_dir / f"fresh_out_{wmode}.json"
        jp.write_text(json.dumps({
            "model_path": cfg["model_path"], "backend_kind": "omni",
            "damaged_state_path": str(m0_pt.resolve()), "expected_damaged_sha256": m0_sha,
            "adapter_dirs": ad_paths, "consistency_probes": [], "prob_diff_tolerance": 1e-4,
            "eval_units": eval_units_spec, "audit_controls": audit_controls, "baseline_controls": m0_controls,
            "output": str(op.resolve()),
        }, indent=2), encoding="utf-8")
        fr = run_fresh_subprocess_audit(jp, python_exe=sys.executable)
        la = fr.get("locality_audit", {})
        fh = fr.get("final_heldout_acc", {})
        restD = sum(1 for u in degraded_set if float(fh.get(u, 0.0)) >= 0.999)
        loc_ok = bool(la.get("strict_locality_pass", False))
        fresh_ok = bool(fr.get("fresh_reload_passed", False))
        y_fr = (restD / len(degraded_set)) if (loc_ok and fresh_ok) else 0.0
        branches[wmode]["fresh_subprocess"] = {
            "fresh_ok": fresh_ok, "loc_pass": loc_ok, "restored_D": restD,
            "se_regs": len(la.get("same_entity_regressions", [])),
            "overall_agree": la.get("overall_token_agreement"), "Y_strict_fresh": y_fr,
        }
        print(f"DEV7_FRESH {wmode:26s} fresh={fresh_ok} loc={loc_ok} restD={restD}/{len(degraded_set)} se_regs={branches[wmode]['fresh_subprocess']['se_regs']} Y_fresh={y_fr:.3f}", flush=True)

    out = {
        "seed": seed, "disjoint_audit": disj, "m0_sha256": m0_sha,
        "n_mastered": len(mastered), "denominator_D": len(degraded_uids),
        "oracle_targets": [pub_map[u].entity_id for u in oracle_targets],
        "branches": branches, "elapsed_seconds": time.time() - t0,
    }
    (out_dir / "dev_writer_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"DEV7_COMPLETE elapsed={out['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
