"""P1-P3 experimental pipeline: stratified M2 discovery (H1), post-injection truth, and E7 selector comparisons."""
from __future__ import annotations

import hashlib
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from .discovery import detect
from .knowledge_registry import all_evaluator_units, question_prompt
from .metrics import detection_metrics, paired_cluster_bootstrap_ci, safe_auprc, safe_auroc, safe_brier
from .repairability import (
    CandidateReliabilityPredictor,
    DecomposedPredictor,
    DirectSuccessPredictor,
    baseline_score,
)
from .schemas import CandidateReport, EvaluatorUnit


def select_stratified_damage_targets(
    mastered_evs: Sequence[EvaluatorUnit],
    damage_fraction: float = 0.5,
    seed: int = 107,
    stratify_by_gold: bool = False,
) -> Tuple[List[EvaluatorUnit], List[EvaluatorUnit], Dict[str, object]]:
    """Randomly sample balanced damaged vs healthy targets within each domain (and optionally gold class)."""
    if not (0.0 < damage_fraction < 1.0):
        raise ValueError(f"damage_fraction must be in (0, 1), got {damage_fraction}")
    rng = random.Random(int(seed))

    # Group by stratum while preserving deterministic initial sort by unit_id
    strata: Dict[str, List[EvaluatorUnit]] = {}
    for ev in sorted(mastered_evs, key=lambda e: e.unit.unit_id):
        key = f"{ev.unit.domain}::{ev.gold_answer}" if stratify_by_gold else ev.unit.domain
        strata.setdefault(key, []).append(ev)

    damaged: List[EvaluatorUnit] = []
    healthy: List[EvaluatorUnit] = []
    per_domain_counts: Dict[str, Dict[str, int]] = {}

    for skey in sorted(strata.keys()):
        items = list(strata[skey])
        rng.shuffle(items)
        n_s = len(items)
        n_dmg = int(round(n_s * damage_fraction))
        if n_s >= 2:
            n_dmg = max(1, min(n_s - 1, n_dmg))
        d_part = items[:n_dmg]
        h_part = items[n_dmg:]
        damaged.extend(d_part)
        healthy.extend(h_part)

    # Randomize injection order across domains
    rng.shuffle(damaged)
    rng.shuffle(healthy)

    # Validate strict disjointness and uniqueness of entities
    dmg_ids = [ev.unit.unit_id for ev in damaged]
    hlt_ids = [ev.unit.unit_id for ev in healthy]
    dmg_entities = [ev.unit.entity_id for ev in damaged]
    hlt_entities = [ev.unit.entity_id for ev in healthy]
    if set(dmg_ids).intersection(hlt_ids):
        raise AssertionError("Damaged and healthy unit sets overlap")
    if len(set(dmg_ids)) != len(dmg_ids) or len(set(hlt_ids)) != len(hlt_ids):
        raise AssertionError("Duplicate unit_id detected in target split")
    if len(set(dmg_entities + hlt_entities)) != len(dmg_entities) + len(hlt_entities):
        raise AssertionError("Duplicate entity_id detected across damaged/healthy sets")

    for ev in damaged:
        d = per_domain_counts.setdefault(ev.unit.domain, {"damaged": 0, "healthy": 0})
        d["damaged"] += 1
    for ev in healthy:
        d = per_domain_counts.setdefault(ev.unit.domain, {"damaged": 0, "healthy": 0})
        d["healthy"] += 1

    manifest_core = {
        "seed": int(seed),
        "damage_fraction": float(damage_fraction),
        "stratify_by_gold": bool(stratify_by_gold),
        "n_total_mastered": len(mastered_evs),
        "n_damaged": len(damaged),
        "n_healthy": len(healthy),
        "per_domain_counts": per_domain_counts,
        "damaged_injection_order": dmg_ids,
        "healthy_unit_ids": sorted(hlt_ids),
    }
    manifest_sha = hashlib.sha256(
        json.dumps(manifest_core, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_core["manifest_sha256"] = manifest_sha
    return damaged, healthy, manifest_core


def classify_post_injection_truth(
    unit_id: str,
    planned_damaged: bool,
    injection_completed: bool,
    post_accuracy: float,
    locality_pass: Optional[bool] = None,
    degraded_threshold: float = 0.5,
) -> Dict[str, object]:
    """Explicitly separate planned injection from observed post-injection behavioral degradation."""
    behaviorally_degraded = bool(float(post_accuracy) <= float(degraded_threshold))
    strict_local_damage = bool(
        planned_damaged
        and injection_completed
        and behaviorally_degraded
        and (locality_pass is True)
    )
    collateral_damage = bool((not planned_damaged) and behaviorally_degraded)
    return {
        "unit_id": unit_id,
        "planned_damaged": bool(planned_damaged),
        "injection_completed": bool(injection_completed),
        "post_accuracy": float(post_accuracy),
        "behaviorally_degraded": behaviorally_degraded,
        "strict_local_damage": strict_local_damage,
        "collateral_damage": collateral_damage,
        "locality_pass": locality_pass,
    }


def build_post_injection_truth_table(
    records: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """Aggregate post-injection behavioral truth counts across units."""
    return {
        "n_total": len(records),
        "n_planned_damaged": sum(1 for r in records if r["planned_damaged"]),
        "n_injection_completed": sum(1 for r in records if r["injection_completed"]),
        "n_behaviorally_degraded": sum(1 for r in records if r["behaviorally_degraded"]),
        "n_strict_local_damage": sum(1 for r in records if r["strict_local_damage"]),
        "n_collateral_damage": sum(1 for r in records if r["collateral_damage"]),
        "per_unit": list(records),
    }


def run_m2_discovery_eval(
    omni,
    cfg: Dict[str, object],
    split: str,
    clean_state,
    retains: Sequence[str],
    teacher_logits: Sequence[object],
    limit: int = 18,
    damage_fraction: float = 0.5,
) -> Dict[str, object]:
    """Evaluate M2 discovery with stratified randomization, post-injection truth audit, and raw log export."""
    from .damage import balanced_wrong_assignments, inject_damage, restore_weights
    from .mastery import HELDOUT_PERM, HELDOUT_VARIANTS, evaluate_unit_target, screen_clean_mastery

    ev_units = all_evaluator_units(split=split)[:limit]
    unk_units = all_evaluator_units(split="unknown_control")
    mastered_evs = []
    clean_baseline_records: Dict[str, Dict[str, object]] = {}

    restore_weights(omni, clean_state)
    for ev in ev_units:
        m = screen_clean_mastery(omni, ev, min_choice_mass=float(cfg.get("min_choice_mass", 0.25)))
        p2, _, _ = question_prompt(ev.unit, variant=2)
        probs2, _ = omni.probabilities(p2)
        clean_baseline_records[ev.unit.unit_id] = {"probs": probs2}
        if m["stably_mastered"]:
            mastered_evs.append(ev)

    for ev in unk_units:
        p2, _, _ = question_prompt(ev.unit, variant=2)
        probs2, _ = omni.probabilities(p2)
        clean_baseline_records[ev.unit.unit_id] = {"probs": probs2}

    seed = int(cfg.get("seed", 107)) + 17
    dmg_targets, healthy_targets, split_manifest = select_stratified_damage_targets(
        mastered_evs,
        damage_fraction=damage_fraction,
        seed=seed,
        stratify_by_gold=bool(cfg.get("stratify_by_gold", False)),
    )
    assign, assign_audit = balanced_wrong_assignments(dmg_targets, seed=seed)

    completed_ids = set()
    for ev in dmg_targets:
        inject_damage(
            omni,
            ev.unit,
            ev.gold_answer,
            assign[ev.unit.unit_id],
            retains,
            teacher_logits,
            cfg,
            damage_type=str(cfg.get("damage_type", "D1_wrong_association")),
        )
        completed_ids.add(ev.unit.unit_id)

    # Post-injection behavioral ground-truth re-audit on final damaged model
    planned_ids = {ev.unit.unit_id for ev in dmg_targets}
    truth_rows = []
    for ev in mastered_evs:
        post_eval = evaluate_unit_target(omni, ev, variants=HELDOUT_VARIANTS, permutations=HELDOUT_PERM)
        truth_rows.append(
            classify_post_injection_truth(
                unit_id=ev.unit.unit_id,
                planned_damaged=ev.unit.unit_id in planned_ids,
                injection_completed=ev.unit.unit_id in completed_ids,
                post_accuracy=float(post_eval["accuracy"]),
                locality_pass=None,
            )
        )
    truth_table = build_post_injection_truth_table(truth_rows)
    behavioral_degraded_ids = {r["unit_id"] for r in truth_rows if r["behaviorally_degraded"]}

    mixed_public = [ev.unit for ev in mastered_evs] + [ev.unit for ev in unk_units]
    is_planned_damaged = [1 if u.unit_id in planned_ids else 0 for u in mixed_public]
    is_behav_degraded = [1 if u.unit_id in behavioral_degraded_ids else 0 for u in mixed_public]
    is_unknown = [1 if u.split == "unknown_control" else 0 for u in mixed_public]

    budget = int(cfg.get("discovery_budget_per_unit", 6)) * len(mixed_public)
    simple_reports = detect(
        omni, mixed_public, probe_budget=budget, mode="reference_free", constraint_probe_fraction=0.0
    )
    full_rf_reports = detect(
        omni, mixed_public, probe_budget=budget, mode="reference_free", constraint_probe_fraction=0.9
    )
    full_rb_reports = detect(
        omni,
        mixed_public,
        probe_budget=budget,
        mode="reference_based",
        baseline_records=clean_baseline_records,
        constraint_probe_fraction=0.9,
    )

    restore_weights(omni, clean_state)
    s_dicts = [r.to_dict() for r in simple_reports]
    rf_dicts = [r.to_dict() for r in full_rf_reports]
    rb_dicts = [r.to_dict() for r in full_rb_reports]
    return {
        "n_mastered": len(mastered_evs),
        "n_damaged": len(dmg_targets),
        "n_healthy": len(healthy_targets),
        "n_unknown_ctrl": len(unk_units),
        "split_manifest": split_manifest,
        "assignment_audit": assign_audit,
        "post_injection_truth": truth_table,
        "simple_direct_baseline": detection_metrics(s_dicts, is_behav_degraded, is_unknown),
        "m2_reference_free": detection_metrics(rf_dicts, is_behav_degraded, is_unknown),
        "m2_reference_based": detection_metrics(rb_dicts, is_behav_degraded, is_unknown),
        "planned_label_reference": {
            "simple_direct_baseline": detection_metrics(s_dicts, is_planned_damaged, is_unknown),
            "m2_reference_free": detection_metrics(rf_dicts, is_planned_damaged, is_unknown),
            "m2_reference_based": detection_metrics(rb_dicts, is_planned_damaged, is_unknown),
        },
        "per_unit_records": {
            "unit_ids": [u.unit_id for u in mixed_public],
            "is_planned_damaged": is_planned_damaged,
            "is_behaviorally_degraded": is_behav_degraded,
            "is_unknown_ctrl": is_unknown,
            "simple_direct_baseline": s_dicts,
            "m2_reference_free": rf_dicts,
            "m2_reference_based": rb_dicts,
        },
    }


def evaluate_selector_comparisons(
    records: Sequence[dict],
    budget_fraction: float = 0.5,
    predictor: Optional[DecomposedPredictor] = None,
    branch: str = "structured",
    direct_predictor: Optional[DirectSuccessPredictor] = None,
    reliability_predictor: Optional[CandidateReliabilityPredictor] = None,
    exploratory_or_confirmatory: str = "exploratory",
) -> Dict[str, object]:
    """Compare M4 predictors and baselines with unified eligible vs attempted denominators."""
    all_eligible = [r for r in records if r.get("strict_eligible")]
    n_eligible = len(all_eligible)
    if n_eligible == 0:
        return {"n": 0, "n_strict_eligible": 0, "status": "no_valid_records"}

    valid = [r for r in all_eligible if branch in r.get("repairs", {})]
    n_attempted = len(valid)
    if n_attempted == 0:
        return {
            "n": 0,
            "n_strict_eligible": n_eligible,
            "n_attempted": 0,
            "candidate_coverage": 0.0,
            "abstention_rate_all_eligible": 1.0,
            "status": "no_attempted_records",
        }

    y_strict = [int(r["repairs"][branch]["strict_restoration"]) for r in valid]
    gold_valid = [r.get("gold_evaluator_only") for r in valid]
    y_cand_ok = []
    for r, g in zip(valid, gold_valid):
        if branch == "structured":
            c_ans = r["reconstruction"].get("candidate_answer")
        elif branch == "generative":
            c_ans = r.get("reconstruction_aux", {}).get("generative_candidate")
        else:
            c_ans = r.get("reconstruction_aux", {}).get("direct_candidate")
        y_cand_ok.append(int(c_ans == g))
    clusters = [r["unit"]["source_group"] for r in valid]

    scores_by_method: Dict[str, List[float]] = {
        "random": [],
        "direct_confidence": [],
        "entropy": [],
        "self_consistency": [],
        "candidate_change": [],
        "support_completeness": [],
    }
    prob_calibration_meta: Dict[str, Dict[str, object]] = {
        "random": {"is_calibrated_probability": False, "score_type": "ranking_random"},
        "direct_confidence": {"is_calibrated_probability": False, "score_type": "raw_model_confidence"},
        "entropy": {"is_calibrated_probability": False, "score_type": "uncalibrated_entropy"},
        "self_consistency": {"is_calibrated_probability": False, "score_type": "binary_disagreement"},
        "candidate_change": {"is_calibrated_probability": False, "score_type": "binary_indicator"},
        "support_completeness": {"is_calibrated_probability": False, "score_type": "binary_or_heuristic"},
    }

    if predictor is not None and predictor.calibration_status in {"calibrated", "fitted_uncalibrated"}:
        scores_by_method["decomposed_ere"] = []
        prob_calibration_meta["decomposed_ere"] = {
            "is_calibrated_probability": predictor.calibration_status == "calibrated",
            "calibration_status": predictor.calibration_status,
            "score_type": "factorized_probability",
        }
    if direct_predictor is not None and direct_predictor.calibration_status in {"calibrated", "fitted_uncalibrated"}:
        scores_by_method["direct_success_model"] = []
        prob_calibration_meta["direct_success_model"] = {
            "is_calibrated_probability": direct_predictor.calibration_status == "calibrated",
            "calibration_status": direct_predictor.calibration_status,
            "score_type": "direct_logistic_probability",
        }
    if reliability_predictor is not None and reliability_predictor.calibration_status in {"calibrated", "fitted_uncalibrated"}:
        scores_by_method["candidate_reliability_model"] = []
        prob_calibration_meta["candidate_reliability_model"] = {
            "is_calibrated_probability": reliability_predictor.calibration_status == "calibrated",
            "calibration_status": reliability_predictor.calibration_status,
            "score_type": "candidate_reliability_probability",
        }

    for r in valid:
        cand = CandidateReport(**r["reconstruction"])
        feats = r["diagnostic_features"]
        for m in ["random", "direct_confidence", "entropy", "self_consistency", "candidate_change", "support_completeness"]:
            scores_by_method[m].append(baseline_score(m, feats, cand))
        if "decomposed_ere" in scores_by_method:
            pred = predictor.predict(feats)
            scores_by_method["decomposed_ere"].append(float(pred["repair_success"] or 0.0))
        if "direct_success_model" in scores_by_method:
            scores_by_method["direct_success_model"].append(float(direct_predictor.predict_proba_one(feats) or 0.0))
        if "candidate_reliability_model" in scores_by_method:
            scores_by_method["candidate_reliability_model"].append(
                float(reliability_predictor.predict_proba_one(feats) or 0.0)
            )

    cap = max(1, int(round(n_attempted * budget_fraction)))
    candidate_coverage = float(n_attempted / n_eligible)
    abstention_rate_all = float((n_eligible - n_attempted) / n_eligible)

    results = {}
    selected_vectors = {}
    for m, sc in scores_by_method.items():
        ranked_idx = sorted(range(n_attempted), key=lambda i: sc[i], reverse=True)[:cap]
        sel_set = set(ranked_idx)
        sel_mask = [1 if i in sel_set else 0 for i in range(n_attempted)]
        per_unit_succ = [y_strict[i] * sel_mask[i] for i in range(n_attempted)]
        wrong_writes_sel = sum(1 for i in ranked_idx if y_cand_ok[i] == 0)
        selected_vectors[m] = per_unit_succ
        results[m] = {
            "auroc": safe_auroc(y_strict, sc),
            "auprc": safe_auprc(y_strict, sc),
            "brier": safe_brier(y_strict, [max(0.0, min(1.0, v)) for v in sc]),
            "brier_note": (
                "calibrated_probability"
                if prob_calibration_meta[m]["is_calibrated_probability"]
                else "uncalibrated_or_ranking_score"
            ),
            **prob_calibration_meta[m],
            "selected_count": cap,
            "n_attempted": n_attempted,
            "n_strict_eligible": n_eligible,
            "candidate_coverage": candidate_coverage,
            "selected_coverage_among_attempted": float(cap / n_attempted),
            "selected_coverage_among_all_eligible": float(cap / n_eligible),
            "strict_success_count_selected": int(sum(per_unit_succ)),
            "strict_success_among_selected": float(sum(per_unit_succ) / cap),
            "strict_success_among_attempted": float(sum(per_unit_succ) / n_attempted),
            "strict_success_all_eligible": float(sum(per_unit_succ) / n_eligible),
            "wrong_write_count_selected": int(wrong_writes_sel),
            "wrong_write_rate_among_selected": float(wrong_writes_sel / cap),
            "wrong_write_rate_all_eligible": float(wrong_writes_sel / n_eligible),
            "abstention_rate_all_eligible": abstention_rate_all,
        }

    if "decomposed_ere" in selected_vectors:
        results["ere_vs_candidate_change_ci"] = paired_cluster_bootstrap_ci(
            selected_vectors["decomposed_ere"],
            selected_vectors["candidate_change"],
            clusters,
            exploratory_or_confirmatory=exploratory_or_confirmatory,
        )
    if "direct_success_model" in selected_vectors:
        results["direct_model_vs_candidate_change_ci"] = paired_cluster_bootstrap_ci(
            selected_vectors["direct_success_model"],
            selected_vectors["candidate_change"],
            clusters,
            exploratory_or_confirmatory=exploratory_or_confirmatory,
        )

    return {
        "n": n_attempted,
        "n_attempted": n_attempted,
        "n_strict_eligible": n_eligible,
        "candidate_coverage": candidate_coverage,
        "abstention_rate_all_eligible": abstention_rate_all,
        "positives": int(sum(y_strict)),
        "positives_rate_attempted": float(sum(y_strict) / n_attempted),
        "positives_rate_all_eligible": float(sum(y_strict) / n_eligible),
        "methods": results,
    }
