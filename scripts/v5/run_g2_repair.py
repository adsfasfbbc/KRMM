"""Batch 2 (G2-B + G2-C): Paired Selective Repair Matrix (Direct, Generative, Structured, Evidence-Gated, Oracle) + Locality Graph."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from krmm.v3.damage import balanced_wrong_assignments, inject_damage, restore_weights, snapshot_weights
from krmm.v3.damage_audit import compare_locality, measure_controls
from krmm.v3.evidence import extract_atomic_facts
from krmm.v3.knowledge_registry import AUDIT_GENERAL_CONTROLS, question_prompt, train_retain_prompts
from krmm.v3.mastery import HELDOUT_PERM, HELDOUT_VARIANTS, evaluate_unit_target, screen_clean_mastery
from krmm.v3.metrics import paired_cluster_bootstrap_ci
from krmm.v3.parameter_repair import repair as run_parameter_repair
from krmm.v3.verify_fresh import evaluate_reloaded_adapters
from krmm.v5.locality_graph import evaluate_locality_graph
from krmm.v5.selective_repair import REPAIR_STRATEGIES_V5, select_candidate_for_strategy
from krmm.v5.support_structure import analyze_support_structure
from krmm.v5.v5_catalog import v5_known_evaluator_units, v5_ripple_map


def _choose(probs: List[float]) -> str:
    return "ABCD"[int(np.argmax(probs))]


def _build_v5_audit_controls(target_ev, all_evs) -> List[dict]:
    controls: List[dict] = []
    for sc in target_ev.same_entity_controls:
        controls.append(dict(sc))
    for ev in all_evs:
        if ev.unit.unit_id == target_ev.unit.unit_id:
            continue
        q, _, _ = question_prompt(ev.unit, variant=4)
        kind = "near_neighbor_invariant" if ev.unit.domain == target_ev.unit.domain else "other_entity_unseen_template"
        controls.append({"question": q, "gold": ev.gold_answer, "kind": kind, "entity_id": ev.unit.entity_id})
    for g in AUDIT_GENERAL_CONTROLS:
        controls.append({"question": g["question"], "gold": g["gold"], "kind": "general_commonsense", "entity_id": "__general__"})
    return controls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/root/rivermind-data/models/Qwen2.5-Omni-3B")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--limit", type=int, default=36)
    ap.add_argument("--out", default="results/v5-new/g2_repair_results.json")
    args = ap.parse_args()

    from krmm.model import Omni

    t0 = time.time()
    cfg = {
        "model_path": args.model_path,
        "last_layers": 4,
        "target_modules": ["q_proj", "v_proj", "down_proj"],
        "rank": 8,
        "alpha": 16,
        "damage_lr": 5e-4,
        "damage_steps": 64,
        "damage_retain_weight": 50.0,
        "damage_l2_weight": 1e-4,
        "repair_lr": 5e-4,
        "repair_steps": 32,
        "repair_retain_weight": 50.0,
        "repair_l2_weight": 1e-4,
        "write_strategy": "standard_kl",
        "retain_batch_size": 24,
        "seed": args.seed,
    }
    omni = Omni(args.model_path)
    clean_state = snapshot_weights(omni, cfg)
    retains = train_retain_prompts(include_multi_template=True)
    teacher_logits = omni.retain_logits(retains)

    all_evs = v5_known_evaluator_units()[: args.limit]
    ripple_map = v5_ripple_map()
    assign, assign_audit = balanced_wrong_assignments(all_evs, seed=args.seed)

    records: List[dict] = []
    n_mastered = 0
    n_dmg_eff = 0
    n_strict_elig = 0

    for idx, ev in enumerate(all_evs, 1):
        restore_weights(omni, clean_state)
        u = ev.unit
        m = screen_clean_mastery(omni, ev, min_choice_mass=0.25)
        if not m["stably_mastered"]:
            continue
        n_mastered += 1

        controls = _build_v5_audit_controls(ev, all_evs)
        clean_ctrl = measure_controls(omni, controls)

        # Inject D1 damage & audit independent locality
        inject_damage(omni, u, ev.gold_answer, assign[u.unit_id], retains, teacher_logits, cfg, damage_type="D1_wrong_association")
        dmg_held = evaluate_unit_target(omni, ev.unit, ev.gold_answer, variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,))
        dmg_ctrl = measure_controls(omni, controls)
        dmg_loc = compare_locality(clean_ctrl, dmg_ctrl, controls, u.entity_id)

        if dmg_held["accuracy"] > 0.5:
            continue
        n_dmg_eff += 1
        if not dmg_loc["independent_locality_pass"]:
            continue
        n_strict_elig += 1

        damaged_state = snapshot_weights(omni, cfg)
        dmg_teacher = omni.retain_logits(retains)

        # Extract candidates once on damaged model
        p2, _, _ = question_prompt(u, variant=2)
        direct_cand = _choose(omni.probabilities(p2)[0])
        facts, _ = extract_atomic_facts(omni, u, max_new_tokens=24)
        sup = analyze_support_structure(u.domain, facts)
        obs = "\n".join(f"- {f.attribute}: {f.raw_output or f.value}" for f in facts)
        opts = "\n".join(f"{k}: {v}" for k, v in u.options.items())
        gen_cand = _choose(omni.probabilities(f"Infer the correct category using ONLY these observations:\n{obs}\n{opts}\nReply with one letter only.")[0])

        rip_q, rip_g = ripple_map[u.unit_id]

        branch_res: Dict[str, dict] = {}
        # Cache writeback by candidate letter so identical candidates share one LoRA fit & fresh reload!
        cand_cache: Dict[str, dict] = {}
        for strat in REPAIR_STRATEGIES_V5:
            sel = select_candidate_for_strategy(strat, direct_cand, gen_cand, sup, oracle_gold=ev.gold_answer)
            if not sel["attempt"] or sel["candidate"] is None:
                branch_res[strat] = {
                    "attempted": False,
                    "candidate": None,
                    "candidate_matches_gold": False,
                    "strict_restoration": False,
                    "wrong_write": False,
                    "reason": sel["reason"],
                }
                continue

            cand = str(sel["candidate"])
            if cand not in cand_cache:
                restore_weights(omni, damaged_state)
                with tempfile.TemporaryDirectory() as td:
                    td_p = Path(td)
                    dmg_pt = td_p / "damaged_state.pt"
                    import torch
                    torch.save(damaged_state, dmg_pt)
                    ad_dir = td_p / f"adapter_{cand}"
                    run_parameter_repair(
                        omni, u, cand, retains, dmg_teacher, cfg,
                        save_dir=str(ad_dir),
                    )
                    rep_held = evaluate_unit_target(omni, ev.unit, ev.gold_answer, variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,))
                    rep_ctrl = measure_controls(omni, controls)
                    rep_loc = compare_locality(dmg_ctrl, rep_ctrl, controls, u.entity_id)
                    rip_pred = _choose(omni.probabilities(rip_q)[0])
                    rip_acc = float(rip_pred == rip_g)
                    if hasattr(omni.model, "unload"):
                        omni.model = omni.model.unload()

                    # Serialized PEFT adapter reload check from clean damaged_state
                    fresh_ok = False
                    if rep_held["accuracy"] >= 0.99 and rep_loc["independent_locality_pass"]:
                        from peft import PeftModel
                        restore_weights(omni, damaged_state)
                        omni.model = PeftModel.from_pretrained(omni.model, str(ad_dir), is_trainable=False)
                        reloaded_ev = evaluate_unit_target(
                            omni, ev.unit, ev.gold_answer,
                            variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,),
                        )
                        omni.model = omni.model.unload()
                        fresh_ok = bool(reloaded_ev["accuracy"] >= 0.99)

                    lg = evaluate_locality_graph(
                        unit_id=u.unit_id,
                        target_acc=float(rep_held["accuracy"]),
                        same_entity_before=float(rep_loc["same_entity_other_relation"]["baseline_accuracy"]),
                        same_entity_after=float(rep_loc["same_entity_other_relation"]["after_accuracy"]),
                        neighbor_before=float(rep_loc["near_neighbor_invariant"]["baseline_accuracy"]),
                        neighbor_after=float(rep_loc["near_neighbor_invariant"]["after_accuracy"]),
                        unrelated_before=float(rep_loc["other_entity_unseen_template"]["baseline_accuracy"]),
                        unrelated_after=float(rep_loc["other_entity_unseen_template"]["after_accuracy"]),
                        general_before=float(rep_loc["general_commonsense"]["baseline_accuracy"]),
                        general_after=float(rep_loc["general_commonsense"]["after_accuracy"]),
                        ripple_after=rip_acc,
                        fresh_reload_passed=fresh_ok,
                    )
                    cand_cache[cand] = {
                        "target_heldout_acc": float(rep_held["accuracy"]),
                        "independent_locality_pass": bool(rep_loc["independent_locality_pass"]),
                        "fresh_reload_passed": fresh_ok,
                        "locality_graph": lg.to_dict(),
                        "strict_restoration": bool(lg.strict_locality_pass),
                    }

            cached = cand_cache[cand]
            match = cand == ev.gold_answer
            branch_res[strat] = {
                "attempted": True,
                "candidate": cand,
                "candidate_matches_gold": match,
                "wrong_write": not match,
                "reason": sel["reason"],
                **cached,
            }

        records.append(
            {
                "unit_id": u.unit_id,
                "entity_id": u.entity_id,
                "domain": u.domain,
                "source_group": u.source_group,
                "gold_evaluator_only": ev.gold_answer,
                "support_class": sup.support_class,
                "estimated_reliability": sup.estimated_reliability,
                "branches": branch_res,
            }
        )
        print(
            f"G2BC_UNIT {idx}/{len(all_evs)} {u.entity_id:24s} cls={sup.support_class:28s} "
            f"dir={branch_res['direct']['strict_restoration']} gen={branch_res['generative']['strict_restoration']} "
            f"str={branch_res['structured']['strict_restoration']} eg={branch_res['evidence_gated']['strict_restoration']} "
            f"orc={branch_res['oracle']['strict_restoration']}",
            flush=True,
        )

    restore_weights(omni, clean_state)

    # Aggregate summary & paired bootstrap CIs
    n = len(records)
    summary: Dict[str, dict] = {}
    vecs: Dict[str, List[int]] = {}
    clusters = [r["source_group"] for r in records]
    for strat in REPAIR_STRATEGIES_V5:
        v = [int(r["branches"][strat]["strict_restoration"]) for r in records]
        att = [r for r in records if r["branches"][strat]["attempted"]]
        ww = sum(1 for r in att if r["branches"][strat]["wrong_write"])
        vecs[strat] = v
        summary[strat] = {
            "n_strict_eligible": n,
            "n_attempted": len(att),
            "n_strict_restored": sum(v),
            "strict_restoration_rate_eligible": sum(v) / max(1, n),
            "strict_restoration_rate_attempted": sum(v) / max(1, len(att)),
            "wrong_write_count": ww,
            "wrong_write_rate_eligible": ww / max(1, n),
            "abstention_rate_eligible": (n - len(att)) / max(1, n),
            "mean_same_entity_preservation": float(np.mean([r["branches"][strat]["locality_graph"]["same_entity_relation_preservation"] for r in att])) if att else 1.0,
            "mean_expected_ripple_consistency": float(np.mean([r["branches"][strat]["locality_graph"]["expected_ripple_consistency"] for r in att])) if att else 0.0,
        }

    cis = {
        "structured_vs_generative": paired_cluster_bootstrap_ci(vecs["structured"], vecs["generative"], clusters, exploratory_or_confirmatory="confirmatory_v5"),
        "evidence_gated_vs_generative": paired_cluster_bootstrap_ci(vecs["evidence_gated"], vecs["generative"], clusters, exploratory_or_confirmatory="confirmatory_v5"),
        "evidence_gated_vs_direct": paired_cluster_bootstrap_ci(vecs["evidence_gated"], vecs["direct"], clusters, exploratory_or_confirmatory="confirmatory_v5"),
    }

    out = {
        "n_catalog": len(all_evs),
        "n_mastered": n_mastered,
        "n_damage_effective": n_dmg_eff,
        "n_strict_eligible": n_strict_elig,
        "assignment_audit": assign_audit,
        "strategy_summary": summary,
        "paired_bootstrap_cis": cis,
        "per_unit": records,
        "elapsed_seconds": time.time() - t0,
    }
    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"G2BC_COMPLETE n_strict_elig={n_strict_elig} elapsed={out['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
