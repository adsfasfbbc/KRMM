"""P1-P3 full experimental pipeline: M2 discovery (H1), E0-E7 matrix (H2, H3), and independent calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch

from krmm.model import Omni
from .damage import balanced_wrong_assignments, inject_damage, restore_weights, snapshot_weights, state_sha256
from .damage_audit import compare_locality, measure_controls
from .discovery import detect
from .knowledge_registry import all_evaluator_units, audit_controls_for_split, question_prompt, train_retain_prompts
from .mastery import HELDOUT_PERM, HELDOUT_VARIANTS, evaluate_unit_target, screen_clean_mastery
from .memory_recovery import ExternalMemoryStore, compensate_nonparametric
from .metrics import detection_metrics, paired_cluster_bootstrap_ci, safe_auprc, safe_auroc, safe_brier
from .parameter_repair import repair as run_parameter_repair
from .reconstruction import reconstruct
from .repairability import DIAGNOSTIC_FEATURES, DecomposedPredictor, assess_repairability, baseline_score, extract_diagnostic_features
from .repair_policy import select_for_budget
from .verify_fresh import evaluate_reloaded_adapters


def run_m2_discovery_eval(
    omni,
    cfg: Dict[str, object],
    split: str,
    clean_state: Dict[str, torch.Tensor],
    retains: Sequence[str],
    teacher_logits: Sequence[torch.Tensor],
    limit: int = 18,
    damage_fraction: float = 0.5,
) -> Dict[str, object]:
    """Evaluate M2 reference-based and reference-free discovery against simple direct anomaly baseline (H1)."""
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

    n_dmg = max(1, int(round(len(mastered_evs) * damage_fraction)))
    dmg_targets = mastered_evs[:n_dmg]
    healthy_targets = mastered_evs[n_dmg:]
    assign, _ = balanced_wrong_assignments(dmg_targets, seed=int(cfg.get("seed", 107)) + 17)

    # Inject damage sequentially onto the SAME model state for selected units (testing multi-unit discovery!)
    damaged_ids = set()
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
        damaged_ids.add(ev.unit.unit_id)

    mixed_public = [ev.unit for ev in mastered_evs] + [ev.unit for ev in unk_units]
    is_damaged = [1 if u.unit_id in damaged_ids else 0 for u in mixed_public]
    is_unknown = [1 if u.split == "unknown_control" else 0 for u in mixed_public]

    budget = int(cfg.get("discovery_budget_per_unit", 6)) * len(mixed_public)
    print(f"M2_DETECT_START total={len(mixed_public)} dmg={len(dmg_targets)} budget={budget}", flush=True)
    # 1. Simple direct-only anomaly baseline (no EvidenceGraph / constraint probing)
    simple_reports = detect(
        omni, mixed_public, probe_budget=budget, mode="reference_free", constraint_probe_fraction=0.0
    )
    print("M2_SIMPLE_DONE", flush=True)
    # 2. Full M2 reference-free with constraint graph probing
    full_rf_reports = detect(
        omni, mixed_public, probe_budget=budget, mode="reference_free", constraint_probe_fraction=0.9
    )
    print("M2_RF_DONE", flush=True)
    # 3. Full M2 reference-based
    full_rb_reports = detect(
        omni,
        mixed_public,
        probe_budget=budget,
        mode="reference_based",
        baseline_records=clean_baseline_records,
        constraint_probe_fraction=0.9,
    )
    print("M2_RB_DONE", flush=True)

    restore_weights(omni, clean_state)
    return {
        "n_mastered": len(mastered_evs),
        "n_damaged": len(dmg_targets),
        "n_healthy": len(healthy_targets),
        "n_unknown_ctrl": len(unk_units),
        "simple_direct_baseline": detection_metrics([r.to_dict() for r in simple_reports], is_damaged, is_unknown),
        "m2_reference_free": detection_metrics([r.to_dict() for r in full_rf_reports], is_damaged, is_unknown),
        "m2_reference_based": detection_metrics([r.to_dict() for r in full_rb_reports], is_damaged, is_unknown),
    }


def evaluate_selector_comparisons(
    records: Sequence[dict],
    budget_fraction: float = 0.5,
    predictor: Optional[DecomposedPredictor] = None,
    branch: str = "structured",
) -> Dict[str, object]:
    """Compare M4 predictors and baselines at matched coverage on a set of evaluated records."""
    valid = [r for r in records if r.get("strict_eligible") and branch in r.get("repairs", {})]
    if not valid:
        return {"n": 0, "status": "no_valid_records"}

    y_strict = [int(r["repairs"][branch]["strict_restoration"]) for r in valid]
    clusters = [r["unit"]["source_group"] for r in valid]

    from .schemas import CandidateReport, RepairabilityReport
    scores_by_method: Dict[str, List[float]] = {
        "random": [],
        "direct_confidence": [],
        "entropy": [],
        "self_consistency": [],
        "candidate_change": [],
        "support_completeness": [],
    }
    if predictor is not None and predictor.calibration_status == "calibrated":
        scores_by_method["decomposed_ere"] = []

    for r in valid:
        cand_d = r["reconstruction"]
        cand = CandidateReport(**cand_d)
        feats = r["diagnostic_features"]
        for m in ["random", "direct_confidence", "entropy", "self_consistency", "candidate_change", "support_completeness"]:
            scores_by_method[m].append(baseline_score(m, feats, cand))
        if "decomposed_ere" in scores_by_method:
            pred = predictor.predict(feats)
            scores_by_method["decomposed_ere"].append(float(pred["repair_success"] or 0.0))

    cap = max(1, int(round(len(valid) * budget_fraction)))
    results = {}
    selected_vectors = {}
    for m, sc in scores_by_method.items():
        ranked_idx = sorted(range(len(valid)), key=lambda i: sc[i], reverse=True)[:cap]
        sel_mask = [1 if i in set(ranked_idx) else 0 for i in range(len(valid))]
        per_unit_succ = [y_strict[i] * sel_mask[i] for i in range(len(valid))]
        selected_vectors[m] = per_unit_succ
        results[m] = {
            "auroc": safe_auroc(y_strict, sc),
            "auprc": safe_auprc(y_strict, sc),
            "brier": safe_brier(y_strict, [max(0.0, min(1.0, v)) for v in sc]),
            "selected_count": cap,
            "strict_success_among_selected": sum(per_unit_succ) / cap,
            "strict_success_all_eligible": sum(per_unit_succ) / len(valid),
        }

    if "decomposed_ere" in selected_vectors:
        results["ere_vs_candidate_change_ci"] = paired_cluster_bootstrap_ci(
            selected_vectors["decomposed_ere"],
            selected_vectors["candidate_change"],
            clusters,
        )
    return {"n": len(valid), "positives": sum(y_strict), "methods": results}
