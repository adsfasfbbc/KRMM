#!/usr/bin/env python3
"""Run isolated fresh-subprocess reload audit on already-saved cumulative adapters in a run_episode directory."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from krmm.v3.damage import state_sha256
from krmm.v6.audit import run_fresh_subprocess_audit
from krmm.v6.metrics import compute_episode_metrics
from krmm.v6.schemas import PublicUnit

E2E_STRATEGIES = (
    "B0_no_repair",
    "B1_generative",
    "B2_ungated_structured",
    "B3_confidence_gate",
    "B3_fact_count_gate",
    "B3_random_gate",
    "M_evidence_gated",
    "M_full",
    "Oracle_upper_bound",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v6/pilot.json")
    ap.add_argument("--discovery-run-id", required=True)
    ap.add_argument("--episode-run-id", required=True)
    args = ap.parse_args()

    t0 = time.time()
    cfg = json.loads((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    disc_dir = REPO_ROOT / "results" / "v6-new" / args.discovery_run_id
    run_dir = REPO_ROOT / "results" / "v6-new" / args.episode_run_id

    pub_manifest = json.loads((REPO_ROOT / cfg["public_manifest"]).read_text(encoding="utf-8"))
    hidden_manifest = json.loads((REPO_ROOT / cfg["evaluator_hidden"]).read_text(encoding="utf-8"))
    pub_units = [PublicUnit(**u) for u in pub_manifest["public_units"]]
    eval_map = {r["unit_id"]: r for r in hidden_manifest["evaluator_records"]}

    damaged_pt = disc_dir / "damaged_state.pt"
    damaged_state = torch.load(damaged_pt, map_location="cpu", weights_only=True)
    damaged_sha = state_sha256(damaged_state)

    truth_records = json.loads((disc_dir / "evaluator_only" / "truth_records.json").read_text(encoding="utf-8"))
    audit_controls = json.loads((disc_dir / "evaluator_only" / "audit_controls.json").read_text(encoding="utf-8"))
    post_damage_controls = json.loads((disc_dir / "evaluator_only" / "post_damage_controls.json").read_text(encoding="utf-8"))

    eval_units_spec = []
    eval_gold_map = {}
    eval_role_map = {}
    for pu in pub_units:
        er = eval_map[pu.unit_id]
        ku = pu.to_knowledge_unit("test")
        eval_units_spec.append({"unit": ku.to_dict(), "gold_answer": er["gold_answer"]})
        eval_gold_map[pu.unit_id] = er["gold_answer"]
        eval_role_map[pu.unit_id] = er["pool_role"]

    all_metrics: Dict[str, dict] = {}
    for strat in E2E_STRATEGIES:
        strat_dir = run_dir / strat
        if not strat_dir.exists():
            continue
        steps: List[dict] = []
        act_f = strat_dir / "method_actions.jsonl"
        if act_f.exists():
            for line in act_f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    steps.append(json.loads(line))
        committed_adapters = [
            s["adapter_path"] for s in steps
            if s.get("action") == "ATTEMPT_REPAIR" and not s.get("rollback") and s.get("adapter_path")
        ]
        job_path = strat_dir / "fresh_audit_job.json"
        out_path = strat_dir / "fresh_audit_result.json"
        job = {
            "model_path": cfg["model_path"],
            "backend_kind": cfg.get("backend_kind", "omni"),
            "damaged_state_path": str(damaged_pt.resolve()),
            "expected_damaged_sha256": damaged_sha,
            "adapter_dirs": committed_adapters,
            "consistency_probes": [],
            "prob_diff_tolerance": float(cfg["locality_gates"].get("prob_diff_tolerance", 1e-4)),
            "eval_units": eval_units_spec,
            "audit_controls": audit_controls,
            "baseline_controls": post_damage_controls,
            "max_group_acc_drop": float(cfg["locality_gates"].get("max_group_acc_drop", 0.05)),
            "min_token_agreement": float(cfg["locality_gates"].get("min_token_agreement", 0.90)),
            "output": str(out_path.resolve()),
        }
        job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        fresh_res = run_fresh_subprocess_audit(job_path, python_exe=sys.executable)
        if not fresh_res.get("fresh_subprocess_ok"):
            print(f"FRESH_SUBPROCESS_FAILED strat={strat}: {fresh_res}", flush=True)
            all_metrics[strat] = {"error": fresh_res}
            continue
        ep_m = compute_episode_metrics(
            truth_records=truth_records,
            final_heldout_acc=fresh_res["final_heldout_acc"],
            step_records=steps,
            locality_audit=fresh_res["locality_audit"],
            fresh_reload_passed=bool(fresh_res["fresh_reload_passed"]),
            eval_gold_map=eval_gold_map,
            eval_role_map=eval_role_map,
        )
        ep_m["fresh_reload_n_adapters"] = fresh_res["n_adapters_reloaded"]
        all_metrics[strat] = ep_m
        print(
            f"V6_E2E_FRESH_VERIFIED strat={strat:22s} Y={ep_m['end_to_end_strict_safe_yield_Y']} "
            f"restored={ep_m['restored_in_D_raw']}/{ep_m['denominator_D']} "
            f"active={ep_m['active_repaired_in_D']} comm={ep_m['n_committed_writes']} "
            f"ww={ep_m['wrong_write_count_committed']} unk_att={ep_m['unknown_write_attempts']} "
            f"harm={ep_m['harmed_healthy_count']} loc={ep_m['strict_locality_pass']}",
            flush=True,
        )

    (run_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    (run_dir / "runtime.json").write_text(json.dumps({"fresh_audit_elapsed_seconds": time.time() - t0}, indent=2), encoding="utf-8")
    print(f"V6_FRESH_AUDIT_COMPLETE elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
