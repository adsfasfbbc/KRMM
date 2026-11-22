"""Fresh-process independent evaluator for serialized adapters (M5.5, M6.1)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel

from krmm.model import Omni
from .damage import restore_weights, state_sha256
from .damage_audit import compare_locality, measure_controls
from .mastery import HELDOUT_PERM, HELDOUT_VARIANTS, evaluate_unit_target
from .schemas import KnowledgeUnit


def evaluate_reloaded_adapters(omni, job: dict) -> dict:
    """Evaluate all serialized repair adapters in an isolated process from clean+damaged state."""
    unit = KnowledgeUnit.from_public_dict(job["unit"])
    damaged_state = torch.load(job["damaged_state"], map_location="cpu", weights_only=True)
    restore_weights(omni, damaged_state)

    if job.get("expected_damage_sha256"):
        actual_sha = state_sha256(damaged_state)
        if actual_sha != job["expected_damage_sha256"]:
            raise RuntimeError(f"Damaged state SHA mismatch: {actual_sha} != {job['expected_damage_sha256']}")

    teacher_logits = torch.load(job["teacher"], map_location="cpu", weights_only=True)
    retains = list(job["train_retains"])
    controls = list(job["audit_controls"])
    damaged_controls = measure_controls(omni, controls)
    min_retain = float(job.get("min_retain_agreement", 0.90))
    min_loc_agree = float(job.get("min_locality_token_agreement", 0.90))
    max_loc_drop = float(job.get("max_locality_accuracy_drop", 0.05))
    min_acc = float(job.get("success_accuracy", 1.0))
    min_gain = float(job.get("minimum_gain", 0.5))

    out = {}
    for branch, adapter_dir in job["adapters"].items():
        restore_weights(omni, damaged_state)
        omni.model = PeftModel.from_pretrained(omni.model, adapter_dir, is_trainable=False)
        target_eval = evaluate_unit_target(
            omni,
            unit,
            job["gold_answer"],
            variants=HELDOUT_VARIANTS,
            permutations=(HELDOUT_PERM,),
        )
        curr_logits = omni.retain_logits(retains)
        ret_agree = float(
            np.mean([int(a.argmax()) == int(b.argmax()) for a, b in zip(teacher_logits, curr_logits)])
        )
        post_controls = measure_controls(omni, controls)
        loc_vs_damaged = compare_locality(
            damaged_controls,
            post_controls,
            controls,
            unit.entity_id,
            min_token_agreement=min_loc_agree,
            max_accuracy_drop=max_loc_drop,
        )
        rollback_required = bool(ret_agree < min_retain or not loc_vs_damaged["independent_locality_pass"])
        target_ok = bool(
            target_eval["accuracy"] >= min_acc
            and (target_eval["accuracy"] - float(job["damaged_accuracy"])) >= min_gain
        )
        out[branch] = {
            "target_eval": target_eval,
            "accuracy": target_eval["accuracy"],
            "train_retention_agreement": ret_agree,
            "independent_locality_vs_damaged": loc_vs_damaged,
            "independent_locality_pass": bool(loc_vs_damaged["independent_locality_pass"]),
            "rollback_required": rollback_required,
            "operational_success": bool(target_ok and ret_agree >= min_retain),
            "strict_repair_success": bool(target_ok and not rollback_required),
        }
        omni.model = omni.model.unload()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    args = ap.parse_args()
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    torch.set_num_threads(8)
    omni = Omni(job["model_path"])
    results = evaluate_reloaded_adapters(omni, job)
    Path(job["output"]).write_text(json.dumps(results, indent=2), encoding="utf-8")
    Path(job["output"]).with_name("fresh-verify-resources.json").write_text(
        json.dumps(omni.state(), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
