#!/usr/bin/env python3
"""P2 / G1: Create ONE shared multi-target damaged checkpoint M_d & run fair equal-budget blind discovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
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
from krmm.v3.schemas import EvaluatorUnit, KnowledgeUnit
from krmm.v6.budget import StrictBudgetProxy
from krmm.v6.detectors import V6_DETECTORS, evaluate_v6_discovery, run_v6_detector
from krmm.v6.schemas import PublicUnit
from krmm.v6.truth_audit import classify_paired_truth


def _load_backend(model_path: str, backend_kind: str):
    if backend_kind == "causal_lm":
        from scripts.v5.run_g3_transfer import _CausalLMWrapper
        return _CausalLMWrapper(model_path)
    from krmm.model import Omni
    return Omni(model_path)


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
            "probs": [float(p) for p in probs],
            "mass": float(mass),
        })
    return out


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="KRMM v0.6 G1 Multi-Target Damage + Fair Discovery")
    ap.add_argument("--config", default="configs/v6/pilot.json")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    t0 = time.time()
    cfg = json.loads((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    run_dir = REPO_ROOT / "results" / "v6-new" / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists (never overwrite): {run_dir}")
    eval_dir = run_dir / "evaluator_only"
    eval_dir.mkdir(parents=True, exist_ok=False)

    # Save exact config, command, provenance
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    pub_manifest = json.loads((REPO_ROOT / cfg["public_manifest"]).read_text(encoding="utf-8"))
    hidden_manifest = json.loads((REPO_ROOT / cfg["evaluator_hidden"]).read_text(encoding="utf-8"))
    (run_dir / "public_manifest.sha256").write_text(
        f"{hidden_manifest['public_manifest_sha256']}  {cfg['public_manifest']}\n", encoding="utf-8"
    )

    pub_units = [PublicUnit(**u) for u in pub_manifest["public_units"]]
    eval_map = {r["unit_id"]: r for r in hidden_manifest["evaluator_records"]}

    backend_kind = cfg.get("backend_kind", "omni")
    backend = _load_backend(cfg["model_path"], backend_kind)
    lora_cfg = {**cfg["lora"], **cfg["damage_cfg"], "model_path": cfg["model_path"], "seed": seed}
    clean_state = snapshot_weights(backend, lora_cfg)
    clean_sha = state_sha256(clean_state)

    retains = train_retain_prompts(include_multi_template=True)
    teacher_logits = backend.retain_logits(retains)

    # 1. Paired CLEAN screening + CLEAN held-out audit (fixes v0.5 bug A!)
    clean_screen_map: Dict[str, dict] = {}
    clean_heldout_map: Dict[str, float | None] = {}
    mastered_ku_list: List[EvaluatorUnit] = []

    for pu in pub_units:
        er = eval_map[pu.unit_id]
        ku = pu.to_knowledge_unit("test")
        if er["gold_answer"] is None:
            clean_screen_map[pu.unit_id] = {"stably_mastered": False, "screen": {"accuracy": 0.0}}
            clean_heldout_map[pu.unit_id] = None
            continue
        ev_u = EvaluatorUnit(ku, er["gold_answer"], [], er["same_entity_controls"], er["pool_role"] != "known_real")
        m = screen_clean_mastery(backend, ev_u, min_choice_mass=0.25)
        h_ev = evaluate_unit_target(backend, ku, er["gold_answer"], variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,))
        clean_screen_map[pu.unit_id] = m
        clean_heldout_map[pu.unit_id] = float(h_ev["accuracy"])
        if bool(m["stably_mastered"]) and float(h_ev["accuracy"]) >= 0.999 and er["pool_role"] == "known_real":
            mastered_ku_list.append(ev_u)

    print(f"V6_CLEAN_MASTERY mastered_known_with_heldout={len(mastered_ku_list)}/36", flush=True)

    # Build full hidden locality controls (same-entity for ALL known + general commonsense)
    audit_controls: List[dict] = []
    for pu in pub_units:
        er = eval_map[pu.unit_id]
        for sc in er["same_entity_controls"]:
            audit_controls.append(dict(sc))
    for g in AUDIT_GENERAL_CONTROLS:
        audit_controls.append({"question": g["question"], "gold": g["gold"], "kind": "general_commonsense", "entity_id": "__general__"})

    pre_damage_controls = _eval_controls(backend, audit_controls)

    # 2. Balanced random multi-target D1 damage injection on ONE shared checkpoint M_d
    rng = random.Random(seed)
    by_dom: Dict[str, List[EvaluatorUnit]] = {}
    for ev_u in mastered_ku_list:
        by_dom.setdefault(ev_u.unit.domain, []).append(ev_u)
    planned_targets: List[EvaluatorUnit] = []
    frac = float(cfg.get("damage_fraction", 0.45))
    for dom in sorted(by_dom.keys()):
        items = list(by_dom[dom])
        rng.shuffle(items)
        k_dom = max(1, int(round(len(items) * frac)))
        planned_targets.extend(items[:k_dom])
    rng.shuffle(planned_targets)

    assign, assign_audit = balanced_wrong_assignments(planned_targets, seed=seed)
    planned_ids = {ev_u.unit.unit_id for ev_u in planned_targets}
    completed_ids = set()
    for idx, ev_u in enumerate(planned_targets, 1):
        inject_damage(
            backend,
            ev_u.unit,
            ev_u.gold_answer,
            assign[ev_u.unit.unit_id],
            retains,
            teacher_logits,
            lora_cfg,
            damage_type=cfg.get("damage_type", "D1_wrong_association"),
        )
        completed_ids.add(ev_u.unit.unit_id)
        if idx % 4 == 0 or idx == len(planned_targets):
            print(f"V6_INJECTED {idx}/{len(planned_targets)}", flush=True)

    damaged_state = snapshot_weights(backend, lora_cfg)
    damaged_sha = state_sha256(damaged_state)
    damaged_pt = run_dir / "damaged_state.pt"
    torch.save(damaged_state, damaged_pt)

    # 3. Paired POST-damage held-out truth & post-damage hidden controls
    post_damage_controls = _eval_controls(backend, audit_controls)
    truth_records: List[dict] = []
    for pu in pub_units:
        er = eval_map[pu.unit_id]
        ku = pu.to_knowledge_unit("test")
        if er["gold_answer"] is None:
            post_acc = None
        else:
            p_ev = evaluate_unit_target(backend, ku, er["gold_answer"], variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,))
            post_acc = float(p_ev["accuracy"])
        cm = clean_screen_map[pu.unit_id]
        tr = classify_paired_truth(
            unit_id=pu.unit_id,
            entity_id=pu.entity_id,
            domain=pu.domain,
            source_group=pu.source_group,
            pool_role=er["pool_role"],
            planned_damaged=pu.unit_id in planned_ids,
            injection_completed=pu.unit_id in completed_ids,
            clean_screen_passed=bool(cm.get("stably_mastered", False)),
            clean_screen_accuracy=float(cm.get("screen", {}).get("accuracy", 0.0)),
            clean_heldout_accuracy=clean_heldout_map[pu.unit_id],
            post_heldout_accuracy=post_acc,
        )
        truth_records.append(tr.to_dict())

    cat_counts = dict(Counter(r["truth_category"] for r in truth_records))
    print(f"V6_PAIRED_TRUTH_COUNTS {cat_counts}", flush=True)

    # 4. Fair equal-budget blind discovery across all 9 detectors at B in {2N, 3N, 4N, 5N}
    n_pool = len(pub_units)
    shared_prob_cache: Dict = {}
    shared_gen_cache: Dict = {}
    budget_curves: Dict[str, Dict[str, object]] = {}
    primary_mult = int(cfg["budgets"].get("primary_discovery_multiplier", 5))
    primary_reports: Dict[str, list] = {}

    for mult in cfg["budgets"]["discovery_multipliers"]:
        B = int(mult) * n_pool
        budget_curves[f"{mult}N"] = {}
        for det in V6_DETECTORS:
            proxy = StrictBudgetProxy(
                backend=backend,
                query_budget=B,
                use_cache=True,
                shared_prob_cache=shared_prob_cache,
                shared_gen_cache=shared_gen_cache,
            )
            run_out = run_v6_detector(proxy, pub_units, method=det, seed=seed)
            ev_res = evaluate_v6_discovery(
                run_out,
                truth_records,
                fixed_k_fraction=float(cfg["budgets"].get("fixed_top_k_fraction", 0.25)),
                seed=seed,
            )
            budget_curves[f"{mult}N"][det] = ev_res
            if int(mult) == primary_mult:
                primary_reports[det] = run_out["reports"]
        print(f"V6_DISCOVERY_{mult}N_DONE", flush=True)

    # Restore clean weights before exit
    restore_weights(backend, clean_state)

    # Save evaluator_only artifacts & public metrics
    (eval_dir / "truth_records.json").write_text(json.dumps(truth_records, indent=2), encoding="utf-8")
    (eval_dir / "pre_damage_controls.json").write_text(json.dumps(pre_damage_controls, indent=2), encoding="utf-8")
    (eval_dir / "post_damage_controls.json").write_text(json.dumps(post_damage_controls, indent=2), encoding="utf-8")
    (eval_dir / "audit_controls.json").write_text(json.dumps(audit_controls, indent=2), encoding="utf-8")
    (eval_dir / "assignment_audit.json").write_text(json.dumps(assign_audit, indent=2), encoding="utf-8")

    # Public method-visible discovery rankings at primary budget (5N) — strictly NO hidden labels!
    (run_dir / "discovery_rankings_5N.json").write_text(json.dumps(primary_reports, indent=2), encoding="utf-8")

    metrics = {
        "run_id": args.run_id,
        "model_path": cfg["model_path"],
        "backend_kind": backend_kind,
        "seed": seed,
        "clean_state_sha256": clean_sha,
        "damaged_state_sha256": damaged_sha,
        "n_pool": n_pool,
        "n_mastered_known": len(mastered_ku_list),
        "n_planned_damage": len(planned_targets),
        "paired_truth_counts": cat_counts,
        "denominator_D": cat_counts.get("effective_planned_damage", 0) + cat_counts.get("collateral_damaged", 0),
        "budget_curves": budget_curves,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    runtime = {"elapsed_seconds": time.time() - t0, "gpu_launches": 1}
    (run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    prov = {
        "git_commit": _git_commit(),
        "model_path": cfg["model_path"],
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "pool_sha256": pub_manifest["pool_sha256"],
        "damaged_state_sha256": damaged_sha,
    }
    (run_dir / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")

    # Gate G0-SCIENCE check
    n_mastered = len(mastered_ku_list)
    n_eff_plan = cat_counts.get("effective_planned_damage", 0)
    g0_pass = (n_mastered / 36.0 >= 0.75) and (n_eff_plan / max(1, len(planned_targets)) >= 0.50) and (metrics["denominator_D"] >= 8)
    gate = {
        "g0_science_pass": g0_pass,
        "mastery_rate": n_mastered / 36.0,
        "planned_damage_effective_rate": n_eff_plan / max(1, len(planned_targets)),
        "denominator_D": metrics["denominator_D"],
    }
    (run_dir / "gate_assessment.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"V6_DISCOVERY_COMPLETE run_id={args.run_id} D={metrics['denominator_D']} g0_pass={g0_pass} elapsed={runtime['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
