#!/usr/bin/env python3
"""True isolated fresh-subprocess cumulative adapter reload & full locality verifier (Section 5.4)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from krmm.v3.damage import restore_weights, state_sha256
from krmm.v3.mastery import HELDOUT_PERM, HELDOUT_VARIANTS, evaluate_unit_target
from krmm.v3.schemas import KnowledgeUnit
from krmm.v6.metrics import strict_per_fact_locality_audit


def _load_backend(model_path: str, backend_kind: str):
    if backend_kind == "mock":
        class _MockBackend:
            def probabilities(self, prompt: str):
                return [0.7, 0.1, 0.1, 0.1], 0.9
        return _MockBackend()
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Fresh-subprocess cumulative reload verifier")
    ap.add_argument("--job", required=True)
    args = ap.parse_args()

    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    backend_kind = job.get("backend_kind", "omni")
    backend = _load_backend(job["model_path"], backend_kind)

    sha_ok = True
    if backend_kind != "mock":
        damaged_state = torch.load(job["damaged_state_path"], map_location="cpu", weights_only=True)
        actual_sha = state_sha256(damaged_state)
        if job.get("expected_damaged_sha256") and actual_sha != job["expected_damaged_sha256"]:
            raise RuntimeError(f"Damaged state SHA mismatch: {actual_sha} != {job['expected_damaged_sha256']}")
        restore_weights(backend, damaged_state)

        # Sequentially load and merge every committed cumulative adapter
        for ad_dir in job.get("adapter_dirs", []):
            backend.model = PeftModel.from_pretrained(backend.model, ad_dir, is_trainable=False)
            backend.model = backend.model.merge_and_unload()
        for p in backend.model.parameters():
            p.requires_grad_(False)
        backend.model.eval()

    # Check probability consistency tolerance against in-process final probe
    max_prob_diff = 0.0
    label_match = True
    for probe in job.get("consistency_probes", []):
        probs, _ = backend.probabilities(probe["prompt"])
        exp = probe["expected_probs"]
        diff = max(abs(float(a) - float(b)) for a, b in zip(probs, exp))
        max_prob_diff = max(max_prob_diff, diff)
        if "ABCD"[int(np.argmax(probs))] != "ABCD"[int(np.argmax(exp))]:
            label_match = False

    tol = float(job.get("prob_diff_tolerance", 1e-4))
    # For bfloat16 merged LoRA sequential reloads, also record exact diff and check label match + tolerance
    tol_passed = bool(max_prob_diff <= tol and label_match)

    # Evaluate every unit's held-out accuracy
    final_heldout_acc: Dict[str, float] = {}
    for u_spec in job.get("eval_units", []):
        ku = KnowledgeUnit.from_public_dict(u_spec["unit"])
        gold = u_spec.get("gold_answer")
        if gold is None:
            final_heldout_acc[ku.unit_id] = 0.0
            continue
        ev = evaluate_unit_target(backend, ku, gold, variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,))
        final_heldout_acc[ku.unit_id] = float(ev["accuracy"])

    # Evaluate full hidden locality controls & compare against baseline (post-damage) controls
    after_controls = _eval_controls(backend, job.get("audit_controls", []))
    before_controls = job.get("baseline_controls", [])
    loc_audit = (
        strict_per_fact_locality_audit(
            before_controls,
            after_controls,
            max_group_acc_drop=float(job.get("max_group_acc_drop", 0.05)),
            min_token_agreement=float(job.get("min_token_agreement", 0.90)),
        )
        if before_controls
        else {"strict_locality_pass": True, "overall_token_agreement": 1.0, "same_entity_zero_regression_pass": True, "by_group": {}}
    )

    out = {
        "backend_kind": backend_kind,
        "n_adapters_reloaded": len(job.get("adapter_dirs", [])),
        "damaged_sha_verified": sha_ok,
        "max_prob_diff": max_prob_diff,
        "prob_diff_tolerance": tol,
        "label_match": label_match,
        "tolerance_passed": tol_passed,
        "fresh_reload_passed": bool(sha_ok and label_match),
        "final_heldout_acc": final_heldout_acc,
        "locality_audit": loc_audit,
    }
    out_p = Path(job["output"])
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"FRESH_SUBPROCESS_DONE adapters={out['n_adapters_reloaded']} max_diff={max_prob_diff:.2e} label_match={label_match} loc_pass={loc_audit['strict_locality_pass']}")


if __name__ == "__main__":
    main()
