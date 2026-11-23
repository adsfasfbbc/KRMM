"""Batch 1A (G1): Real GPU Equal-Budget Blind Damage Discovery on 52 brand-new v0.5 entities."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from krmm.v3.damage import balanced_wrong_assignments, inject_damage, restore_weights, snapshot_weights
from krmm.v3.knowledge_registry import question_prompt, train_retain_prompts
from krmm.v3.mastery import HELDOUT_PERM, HELDOUT_VARIANTS, evaluate_unit_target, screen_clean_mastery
from krmm.v4.damage_truth_audit import classify_pool_truth, stratified_damage_split
from krmm.v5.blind_discovery import CostTrackedProxy, V5_DETECTORS, evaluate_v5_discovery_metrics, run_v5_detector
from krmm.v5.v5_catalog import v5_ambiguous_evaluator_units, v5_known_evaluator_units, v5_unknown_evaluator_units


class _CachingOmniWrapper:
    """Caches identical prompt queries on the fixed post-injection model so all 7 detectors see identical outputs."""

    def __init__(self, omni) -> None:
        self.omni = omni
        self._prob_cache: Dict[str, tuple] = {}
        self._gen_cache: Dict[tuple, str] = {}

    def probabilities(self, prompt: str, media=None):
        if prompt not in self._prob_cache:
            self._prob_cache[prompt] = self.omni.probabilities(prompt, media=media)
        return self._prob_cache[prompt]

    def generate(self, prompt: str, max_new_tokens: int = 32, media=None) -> str:
        key = (prompt, int(max_new_tokens))
        if key not in self._gen_cache:
            self._gen_cache[key] = self.omni.generate(prompt, max_new_tokens=max_new_tokens, media=media)
        return self._gen_cache[key]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/root/rivermind-data/models/Qwen2.5-Omni-3B")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--damage-steps", type=int, default=64)
    ap.add_argument("--retain-weight", type=float, default=50.0)
    ap.add_argument("--out", default="results/v5-new/g1_discovery_results.json")
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
        "damage_steps": args.damage_steps,
        "damage_retain_weight": args.retain_weight,
        "damage_l2_weight": 1e-4,
        "retain_batch_size": 24,
        "seed": args.seed,
    }

    omni = Omni(args.model_path)
    clean_state = snapshot_weights(omni, cfg)
    retains = train_retain_prompts(include_multi_template=True)
    teacher_logits = omni.retain_logits(retains)

    known_evs = v5_known_evaluator_units()
    unk_evs = v5_unknown_evaluator_units()
    amb_evs = v5_ambiguous_evaluator_units()

    # 1. Clean mastery screening on brand-new v0.5 entities
    mastered_evs = []
    clean_screen_map: Dict[str, dict] = {}
    for ev in known_evs:
        m = screen_clean_mastery(omni, ev, min_choice_mass=0.25)
        clean_screen_map[ev.unit.unit_id] = m
        if m["stably_mastered"]:
            mastered_evs.append(ev)
    for ev in unk_evs + amb_evs:
        clean_screen_map[ev.unit.unit_id] = screen_clean_mastery(omni, ev, min_choice_mass=0.25)

    print(f"G1_CLEAN_MASTERY known={len(mastered_evs)}/{len(known_evs)} unk={len(unk_evs)} amb={len(amb_evs)}", flush=True)

    # 2. Stratified random 50/50 damage split + balanced wrong-target assignment
    dmg_targets, hlt_targets, split_manifest = stratified_damage_split(
        mastered_evs, damage_fraction=0.5, seed=args.seed, stratify_by_gold=True
    )
    assign, assign_audit = balanced_wrong_assignments(dmg_targets, seed=args.seed)
    planned_ids = {ev.unit.unit_id for ev in dmg_targets}

    # 3. Sequential D1 injection onto the SAME model state
    completed_ids = set()
    for idx, ev in enumerate(dmg_targets, 1):
        inject_damage(
            omni,
            ev.unit,
            ev.gold_answer,
            assign[ev.unit.unit_id],
            retains,
            teacher_logits,
            cfg,
            damage_type="D1_wrong_association",
        )
        completed_ids.add(ev.unit.unit_id)
        if idx % 5 == 0 or idx == len(dmg_targets):
            print(f"G1_INJECTED {idx}/{len(dmg_targets)}", flush=True)

    # 4. Full post-injection behavioral re-audit of ALL 52 units
    all_evs = mastered_evs + unk_evs + amb_evs
    amb_ids = {ev.unit.unit_id for ev in amb_evs}
    unk_ids = {ev.unit.unit_id for ev in unk_evs}
    truth_records: List[dict] = []
    for ev in all_evs:
        uid = ev.unit.unit_id
        post_ev = evaluate_unit_target(omni, ev.unit, ev.gold_answer, variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,))
        clean_m = clean_screen_map[uid]
        tr = classify_pool_truth(
            unit_id=uid,
            clean_mastered=bool(clean_m["stably_mastered"]) and (uid not in unk_ids) and (uid not in amb_ids),
            planned_damaged=uid in planned_ids,
            injection_completed=uid in completed_ids,
            clean_accuracy=float(clean_m.get("screen_accuracy", 0.0)),
            post_accuracy=float(post_ev["accuracy"]),
            ambiguous_or_unverifiable=uid in amb_ids,
            locality_pass=None,
        )
        truth_records.append(tr.to_dict())

    cat_counts: Dict[str, int] = {}
    for tr in truth_records:
        cat_counts[tr["pool_category"]] = cat_counts.get(tr["pool_category"], 0) + 1
    print(f"G1_POST_TRUTH_COUNTS {cat_counts}", flush=True)

    # 5. Equal-budget comparison across ALL 7 detectors at B in {2N, 3N, 4N, 5N}
    mixed_public = [ev.unit for ev in all_evs]
    n_pool = len(mixed_public)
    cached_omni = _CachingOmniWrapper(omni)
    budget_curves: Dict[str, Dict[str, object]] = {}
    per_unit_full_budget: Dict[str, list] = {}

    for mult in (2, 3, 4, 5):
        B = mult * n_pool
        budget_curves[f"{mult}N"] = {}
        for method in V5_DETECTORS:
            proxy = CostTrackedProxy(backend=cached_omni, query_budget=B)
            run_out = run_v5_detector(proxy, mixed_public, query_budget=B, method=method)
            metrics = evaluate_v5_discovery_metrics(run_out, truth_records)
            budget_curves[f"{mult}N"][method] = metrics
            if mult == 5:
                per_unit_full_budget[method] = run_out["reports"]
        print(f"G1_BUDGET_{mult}N_DONE", flush=True)

    restore_weights(omni, clean_state)
    out = {
        "model_path": args.model_path,
        "seed": args.seed,
        "n_known_catalog": len(known_evs),
        "n_mastered": len(mastered_evs),
        "n_unknown_ctrl": len(unk_evs),
        "n_ambiguous": len(amb_evs),
        "n_pool_total": n_pool,
        "split_manifest": split_manifest,
        "assignment_audit": assign_audit,
        "post_injection_category_counts": cat_counts,
        "post_injection_truth": truth_records,
        "equal_budget_curves": budget_curves,
        "per_unit_reports_5N": per_unit_full_budget,
        "elapsed_seconds": time.time() - t0,
    }
    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"G1_COMPLETE elapsed={out['elapsed_seconds']:.1f}s out={out_p}", flush=True)


if __name__ == "__main__":
    main()
