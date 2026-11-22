"""Strictly isolated evaluator orchestrating mastery, damage, counterfactual repair, and fresh-process audit."""
from __future__ import annotations

import gc
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
from peft import PeftModel

from .damage import inject_damage, restore_weights, snapshot_weights, state_sha256
from .damage_audit import compare_locality, measure_controls
from .knowledge_registry import audit_controls_for_split, question_prompt, train_retain_prompts
from .mastery import HELDOUT_PERM, HELDOUT_VARIANTS, evaluate_unit_target, screen_clean_mastery
from .parameter_repair import repair as run_parameter_repair
from .reconstruction import reconstruct
from .repairability import assess_repairability, extract_diagnostic_features
from .schemas import EvaluatorUnit, KnowledgeUnit


def evaluate_one_unit(
    omni,
    ev_unit: EvaluatorUnit,
    wrong_target: str,
    cfg: Dict[str, object],
    out_dir: Path,
    clean_state: Dict[str, torch.Tensor],
    clean_train_retains: Sequence[str],
    clean_train_logits: Sequence[torch.Tensor],
    clean_controls_cache: Optional[Dict[str, object]] = None,
    use_fresh_subprocess: bool = True,
) -> Dict[str, object]:
    """Run full P0/M6 evaluation for one EvaluatorUnit, returning strict & operational audit outcomes."""
    t0 = time.monotonic()
    unit: KnowledgeUnit = ev_unit.unit
    split = unit.split
    audit_controls = audit_controls_for_split(split, exclude_entity=unit.entity_id)

    restore_weights(omni, clean_state)
    mastery = screen_clean_mastery(omni, ev_unit, min_choice_mass=float(cfg.get("min_choice_mass", 0.25)))

    record: Dict[str, object] = {
        "unit": unit.to_dict(),
        "gold_evaluator_only": ev_unit.gold_answer,
        "mastery": mastery,
        "operational_eligible": False,
        "strict_eligible": False,
        "exclusion_reason": mastery.get("exclusion_reason"),
    }
    if not mastery["stably_mastered"]:
        return record

    # Clean baseline on independent audit controls for this unit
    clean_ctrl_res = measure_controls(omni, audit_controls)

    # Inject localized artificial damage
    unit_dir = out_dir / unit.unit_id
    unit_dir.mkdir(parents=True, exist_ok=True)
    dmg_adapter_dir = unit_dir / "damage-adapter"
    dmg_logs = inject_damage(
        omni,
        unit,
        ev_unit.gold_answer,
        wrong_target,
        clean_train_retains,
        clean_train_logits,
        cfg,
        damage_type=str(cfg.get("damage_type", "D1_wrong_association")),
        adapter_save_dir=str(dmg_adapter_dir),
    )
    damaged_state = snapshot_weights(omni, cfg)
    damaged_sha = state_sha256(damaged_state)

    # Check target degradation on screening & heldout
    dmg_screen = evaluate_unit_target(omni, unit, ev_unit.gold_answer, variants=(0, 1))
    dmg_heldout = evaluate_unit_target(
        omni, unit, ev_unit.gold_answer, variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,)
    )
    dmg_train_logits = omni.retain_logits(list(clean_train_retains))
    dmg_train_agree = float(
        np.mean([int(a.argmax()) == int(b.argmax()) for a, b in zip(clean_train_logits, dmg_train_logits)])
    )
    dmg_ctrl_res = measure_controls(omni, audit_controls)
    dmg_locality = compare_locality(
        clean_ctrl_res,
        dmg_ctrl_res,
        audit_controls,
        unit.entity_id,
        min_token_agreement=float(cfg.get("min_locality_token_agreement", 0.90)),
        max_accuracy_drop=float(cfg.get("max_locality_accuracy_drop", 0.05)),
    )

    max_dmg_acc = float(cfg.get("max_damaged_accuracy", 0.0))
    min_dmg_drop = float(cfg.get("min_damage_drop", 0.5))
    min_ret = float(cfg.get("min_retain_agreement", 0.90))
    damage_effective = bool(
        dmg_screen["accuracy"] <= max_dmg_acc
        and (mastery["heldout_clean_audit"]["accuracy"] - dmg_heldout["accuracy"]) >= min_dmg_drop
    )
    op_eligible = bool(damage_effective and dmg_train_agree >= min_ret)
    strict_eligible = bool(
        op_eligible and mastery["heldout_clean_mastered"] and dmg_locality["independent_locality_pass"]
    )

    record.update(
        {
            "damage_type": str(cfg.get("damage_type", "D1_wrong_association")),
            "damage_logs": dmg_logs,
            "damage_screen": dmg_screen,
            "damaged_heldout": dmg_heldout,
            "damage_train_retention": dmg_train_agree,
            "damage_independent_locality": dmg_locality,
            "damage_state_sha256": damaged_sha,
            "damage_effective": damage_effective,
            "operational_eligible": op_eligible,
            "strict_eligible": strict_eligible,
        }
    )

    if not op_eligible:
        record["exclusion_reason"] = "damage_not_effective_or_train_retain_failed"
        restore_weights(omni, clean_state)
        return record
    if not strict_eligible:
        record["exclusion_reason"] = (
            "damage_independent_locality_failed"
            if not dmg_locality["independent_locality_pass"]
            else "clean_heldout_not_mastered"
        )
        # In P0 we still optionally evaluate oracle repair if configured, to diagnose writeback vs damage!
        if not bool(cfg.get("evaluate_counterfactual_even_if_damage_locality_failed", True)):
            restore_weights(omni, clean_state)
            return record

    # Save damaged state & damaged teacher logits
    dmg_state_path = unit_dir / "damaged_state.pt"
    teacher_path = unit_dir / "damaged_teacher.pt"
    torch.save(damaged_state, dmg_state_path)
    torch.save(dmg_train_logits, teacher_path)

    # Algorithm-facing diagnosis & reconstruction (ZERO access to gold_answer or clean weights)
    cand_report, aux = reconstruct(
        omni,
        unit,
        budget=int(cfg.get("probe_budget", 24)),
        max_new_tokens=int(cfg.get("max_new_tokens", 32)),
    )
    feats = extract_diagnostic_features(cand_report, None, aux)
    rep_report = assess_repairability(cand_report, None, aux, cfg)
    record["reconstruction"] = cand_report.to_dict()
    record["reconstruction_aux"] = aux
    record["diagnostic_features"] = feats
    record["repairability"] = rep_report.to_dict()

    # Counterfactual repairs across branches (M6.2 matrix)
    branches: Dict[str, tuple[str, Dict[str, object]]] = {
        "oracle": (ev_unit.gold_answer, dict(cfg)),
    }
    if bool(cfg.get("run_non_oracle_branches", True)):
        branches["direct"] = (str(aux["direct_candidate"]), dict(cfg))
        branches["generative"] = (str(aux["generative_candidate"]), dict(cfg))
        if cand_report.candidate_answer is not None:
            branches["structured"] = (cand_report.candidate_answer, dict(cfg))
        if bool(cfg.get("run_unconstrained_ablation", False)):
            u_cfg = dict(cfg, unconstrained=True)
            branches["oracle_unconstrained"] = (ev_unit.gold_answer, u_cfg)

    local_eval: Dict[str, Dict[str, object]] = {}
    adapter_paths: Dict[str, str] = {}
    attempts: Dict[str, Dict[str, object]] = {}

    for bname, (cand_letter, b_cfg) in branches.items():
        restore_weights(omni, damaged_state)
        b_dir = unit_dir / f"repair-{bname}"
        attempt = run_parameter_repair(
            omni,
            unit,
            cand_letter,
            clean_train_retains,
            dmg_train_logits,
            b_cfg,
            str(b_dir),
        )
        attempts[bname] = attempt.to_dict()
        # Local check before unloading
        ev_local = evaluate_unit_target(
            omni, unit, ev_unit.gold_answer, variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,)
        )
        local_eval[bname] = ev_local
        adapter_paths[bname] = str(b_dir.resolve())
        omni.model = omni.model.unload()

    # Fresh-process reload verification
    job_path = unit_dir / "verify-job.json"
    fresh_out_path = unit_dir / "fresh-evaluation.json"
    job = {
        "model_path": str(cfg["model_path"]),
        "unit": unit.to_dict(),
        "gold_answer": ev_unit.gold_answer,
        "damaged_accuracy": dmg_heldout["accuracy"],
        "damaged_state": str(dmg_state_path.resolve()),
        "expected_damage_sha256": damaged_sha,
        "teacher": str(teacher_path.resolve()),
        "train_retains": list(clean_train_retains),
        "audit_controls": audit_controls,
        "adapters": adapter_paths,
        "output": str(fresh_out_path.resolve()),
        "min_retain_agreement": min_ret,
        "min_locality_token_agreement": float(cfg.get("min_locality_token_agreement", 0.90)),
        "max_locality_accuracy_drop": float(cfg.get("max_locality_accuracy_drop", 0.05)),
        "success_accuracy": float(cfg.get("success_accuracy", 1.0)),
        "minimum_gain": min_dmg_drop,
    }
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")

    restore_weights(omni, clean_state)
    gc.collect()
    torch.cuda.empty_cache()

    if use_fresh_subprocess:
        with (unit_dir / "verify.log").open("w", encoding="utf-8") as logf:
            proc = subprocess.run(
                [os.sys.executable, "-m", "krmm.v3.verify_fresh", "--job", str(job_path)],
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"Fresh verification failed for {unit.unit_id}; see {unit_dir / 'verify.log'}")
        fresh = json.loads(fresh_out_path.read_text(encoding="utf-8"))
    else:
        from .verify_fresh import evaluate_reloaded_adapters
        fresh = evaluate_reloaded_adapters(omni, job)
        fresh_out_path.write_text(json.dumps(fresh, indent=2), encoding="utf-8")

    for bname, fev in fresh.items():
        local_accs = [c["canonical_probs"][ev_unit.gold_answer] for c in local_eval[bname]["checks"]]
        fresh_accs = [c["canonical_probs"][ev_unit.gold_answer] for c in fev["target_eval"]["checks"]]
        err = float(np.max(np.abs(np.asarray(local_accs) - np.asarray(fresh_accs))))
        fev["probability_reload_max_error"] = err
        fev["fresh_process_verified"] = bool(err <= 1e-4)
        fev["strict_restoration"] = bool(
            strict_eligible and fev["strict_repair_success"] and fev["fresh_process_verified"]
        )
        fev["repair_write_feasible"] = bool(
            op_eligible and fev["strict_repair_success"] and fev["fresh_process_verified"]
        )

    record["repair_attempts"] = attempts
    record["repairs"] = fresh
    record["seconds"] = time.monotonic() - t0
    restore_weights(omni, clean_state)
    gc.collect()
    torch.cuda.empty_cache()
    return record
