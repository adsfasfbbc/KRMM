#!/usr/bin/env python3
"""P2 / G1: TRUE End-to-End Sequential Blind Repair Episodes from ONE shared damaged checkpoint M_d + Fresh-Subprocess Audit."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from krmm.v3.damage import restore_weights, snapshot_weights, state_sha256
from krmm.v3.evidence import extract_atomic_facts
from krmm.v3.knowledge_registry import question_prompt, train_retain_prompts
from krmm.v3.schemas import KnowledgeUnit
from krmm.v5.support_structure import analyze_support_structure
from krmm.v6.audit import run_fresh_subprocess_audit
from krmm.v6.budget import StrictBudgetProxy
from krmm.v6.metrics import compute_episode_metrics
from krmm.v6.policy import select_action
from krmm.v6.schemas import EpisodeStepRecord, PublicUnit
from krmm.v6.writer import sequential_write_and_public_verify

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


def _load_backend(model_path: str, backend_kind: str):
    if backend_kind == "causal_lm":
        from scripts.v5.run_g3_transfer import _CausalLMWrapper
        return _CausalLMWrapper(model_path)
    from krmm.model import Omni
    return Omni(model_path)


def _argmax_letter(probs: List[float]) -> str:
    return "ABCD"[int(np.argmax(probs))]


def main() -> None:
    ap = argparse.ArgumentParser(description="KRMM v0.6 TRUE End-to-End Sequential Blind Repair")
    ap.add_argument("--config", default="configs/v6/pilot.json")
    ap.add_argument("--discovery-run-id", required=True, help="Existing run_id from run_discovery.py providing M_d")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--strategies", default=",".join(E2E_STRATEGIES))
    args = ap.parse_args()

    t0 = time.time()
    cfg = json.loads((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    disc_dir = REPO_ROOT / "results" / "v6-new" / args.discovery_run_id
    run_dir = REPO_ROOT / "results" / "v6-new" / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"Run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    pub_manifest = json.loads((REPO_ROOT / cfg["public_manifest"]).read_text(encoding="utf-8"))
    hidden_manifest = json.loads((REPO_ROOT / cfg["evaluator_hidden"]).read_text(encoding="utf-8"))
    pub_units = [PublicUnit(**u) for u in pub_manifest["public_units"]]
    pub_map = {u.unit_id: u for u in pub_units}
    eval_map = {r["unit_id"]: r for r in hidden_manifest["evaluator_records"]}

    # Load shared damaged checkpoint M_d & blind discovery rankings
    damaged_pt = disc_dir / "damaged_state.pt"
    damaged_state = torch.load(damaged_pt, map_location="cpu", weights_only=True)
    damaged_sha = state_sha256(damaged_state)
    rankings_5n = json.loads((disc_dir / "discovery_rankings_5N.json").read_text(encoding="utf-8"))

    # Evaluator-only files (loaded ONLY for post-hoc evaluation & separate Oracle trajectory)
    truth_records = json.loads((disc_dir / "evaluator_only" / "truth_records.json").read_text(encoding="utf-8"))
    audit_controls = json.loads((disc_dir / "evaluator_only" / "audit_controls.json").read_text(encoding="utf-8"))
    post_damage_controls = json.loads((disc_dir / "evaluator_only" / "post_damage_controls.json").read_text(encoding="utf-8"))

    backend_kind = cfg.get("backend_kind", "omni")
    backend = _load_backend(cfg["model_path"], backend_kind)
    lora_cfg = {**cfg["lora"], **cfg["repair_cfg"], "model_path": cfg["model_path"], "seed": seed}
    retains = train_retain_prompts(include_multi_template=True)

    restore_weights(backend, damaged_state)
    initial_retain_logits = backend.retain_logits(retains)

    max_writes = int(cfg["budgets"].get("max_writes_per_episode", 8))
    min_ret_agree = float(cfg["repair_cfg"].get("min_public_retain_agreement", 0.90))
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    episode_artifacts: Dict[str, dict] = {}

    for strat in strategies:
        restore_weights(backend, damaged_state)
        strat_dir = run_dir / strat
        strat_dir.mkdir(parents=True, exist_ok=True)
        adapters_dir = strat_dir / "adapters"
        adapters_dir.mkdir(exist_ok=True)

        committed_adapters: List[str] = []
        steps: List[dict] = []
        writes_attempted = 0
        step_id = 0

        if strat == "Oracle_upper_bound":
            # Separate isolated Oracle trajectory: knows degraded targets & gold answers, uses SAME writer & B_w cap
            degraded_uids = [r["unit_id"] for r in truth_records if r["is_degraded_target"]]
            for rank_i, uid in enumerate(degraded_uids[:max_writes], 1):
                step_id += 1
                pu = pub_map[uid]
                er = eval_map[uid]
                gold = er["gold_answer"]
                if gold is None:
                    continue
                ku = pu.to_knowledge_unit("test")
                ad_dir = adapters_dir / f"step_{step_id:02d}_{uid}"
                wr = sequential_write_and_public_verify(
                    backend, ku, gold, retains, initial_retain_logits, lora_cfg, ad_dir, min_ret_agree
                )
                writes_attempted += 1
                if not wr["rollback"]:
                    committed_adapters.append(wr["adapter_path"])
                rec = EpisodeStepRecord(
                    episode_id=f"{args.run_id}:{strat}", model_id=cfg["model_path"], seed=seed,
                    pool_sha256=pub_manifest["pool_sha256"], initial_state_hash=damaged_sha,
                    step_id=step_id, unit_id=uid, source_group=pu.source_group, rank=rank_i,
                    pre_hash=wr["pre_hash"], post_hash=wr["post_hash"], candidate=gold,
                    support_class="oracle_upper_bound", score=1.0, action="ATTEMPT_REPAIR",
                    reason="oracle_separate_trajectory", logical_queries=0, physical_calls=0,
                    tokens=0, train_steps=wr["train_steps"], adapter_path=wr["adapter_path"],
                    adapter_hash=wr["adapter_hash"], public_selfcheck=wr["public_selfcheck"],
                    rollback=wr["rollback"], elapsed=wr["elapsed"], failure_reason=wr["failure_reason"],
                )
                steps.append(rec.to_dict())
        elif strat != "B0_no_repair":
            # Strictly BLIND method trajectory: uses ONLY public discovery ranking + CURRENT model M!
            det_name = "full_m2" if strat == "M_full" else "m2_reference_free"
            ranked_list = rankings_5n[det_name]
            # Consider top suspicious units (pred_state == "damaged" or top-16 ranked) — includes healthy/unknown false positives!
            suspicious = [r for r in ranked_list if r["pred_state"] == "damaged"]
            if len(suspicious) < max_writes:
                seen = {r["unit_id"] for r in suspicious}
                for r in ranked_list:
                    if r["unit_id"] not in seen:
                        suspicious.append(r)
                        seen.add(r["unit_id"])
                    if len(suspicious) >= max_writes * 2:
                        break

            for r in suspicious:
                if writes_attempted >= max_writes:
                    break
                step_id += 1
                uid = r["unit_id"]
                pu = pub_map[uid]
                ku = pu.to_knowledge_unit("test")

                # Reconstruct directly on the CURRENT cumulative model state M!
                proxy = StrictBudgetProxy(backend=backend, query_budget=16, use_cache=False)
                t_s = time.perf_counter()
                p2, _, _ = question_prompt(ku, variant=2)
                probs2, _ = proxy.probabilities(p2)
                dir_cand = _argmax_letter(probs2)
                dir_conf = float(max(probs2))
                facts, _ = extract_atomic_facts(proxy, ku, max_new_tokens=24)
                valid_facts = [f for f in facts if f.value != "unparsed"]
                sup = analyze_support_structure(ku.domain, facts)

                obs = "\n".join(f"- {f.attribute}: {f.raw_output or f.value}" for f in facts)
                opts = "\n".join(f"{k}: {v}" for k, v in ku.options.items())
                gen_probs, _ = proxy.probabilities(
                    f"Infer the correct category using ONLY these observations:\n{obs}\n{opts}\nReply with one letter only."
                )
                gen_cand = _argmax_letter(gen_probs)

                dec = select_action(
                    strategy=strat,
                    unit_id=uid,
                    direct_candidate=dir_cand,
                    direct_confidence=dir_conf,
                    generative_candidate=gen_cand,
                    structured_candidate=sup.candidate_answer,
                    support_class=sup.support_class,
                    n_valid_facts=len(valid_facts),
                    n_minimal_sets=sup.n_minimal_sets,
                    heuristic_support_score=sup.estimated_reliability,
                    remaining_write_budget=max_writes - writes_attempted,
                    seed=seed,
                )
                acc = proxy.accounting()
                cur_hash = state_sha256(snapshot_weights(backend, lora_cfg))

                if dec["action"] == "ATTEMPT_REPAIR" and dec["candidate"] is not None:
                    ad_dir = adapters_dir / f"step_{step_id:02d}_{uid}"
                    wr = sequential_write_and_public_verify(
                        backend, ku, str(dec["candidate"]), retains, initial_retain_logits, lora_cfg, ad_dir, min_ret_agree
                    )
                    writes_attempted += 1
                    if not wr["rollback"]:
                        committed_adapters.append(wr["adapter_path"])
                    rec = EpisodeStepRecord(
                        episode_id=f"{args.run_id}:{strat}", model_id=cfg["model_path"], seed=seed,
                        pool_sha256=pub_manifest["pool_sha256"], initial_state_hash=damaged_sha,
                        step_id=step_id, unit_id=uid, source_group=pu.source_group, rank=int(r["rank"]),
                        pre_hash=wr["pre_hash"], post_hash=wr["post_hash"], candidate=str(dec["candidate"]),
                        support_class=sup.support_class, score=float(dec["score"]), action="ATTEMPT_REPAIR",
                        reason=str(dec["reason"]), logical_queries=int(acc["logical_queries"]),
                        physical_calls=int(acc["physical_calls"]), tokens=int(acc["total_tokens"]),
                        train_steps=wr["train_steps"], adapter_path=wr["adapter_path"],
                        adapter_hash=wr["adapter_hash"], public_selfcheck=wr["public_selfcheck"],
                        rollback=wr["rollback"], elapsed=time.perf_counter() - t_s, failure_reason=wr["failure_reason"],
                    )
                else:
                    rec = EpisodeStepRecord(
                        episode_id=f"{args.run_id}:{strat}", model_id=cfg["model_path"], seed=seed,
                        pool_sha256=pub_manifest["pool_sha256"], initial_state_hash=damaged_sha,
                        step_id=step_id, unit_id=uid, source_group=pu.source_group, rank=int(r["rank"]),
                        pre_hash=cur_hash, post_hash=cur_hash, candidate=None,
                        support_class=sup.support_class, score=float(dec["score"]), action=dec["action"],
                        reason=str(dec["reason"]), logical_queries=int(acc["logical_queries"]),
                        physical_calls=int(acc["physical_calls"]), tokens=int(acc["total_tokens"]),
                        train_steps=0, adapter_path=None, adapter_hash=None,
                        public_selfcheck={"skipped": True}, rollback=False,
                        elapsed=time.perf_counter() - t_s, failure_reason=None,
                    )
                steps.append(rec.to_dict())

        # Record 3 consistency probes on the final in-process model M_f before unloading
        probes = []
        for pu in pub_units[:3]:
            ku = pu.to_knowledge_unit("test")
            pr_txt, _, _ = question_prompt(ku, variant=4)
            pr_p, _ = backend.probabilities(pr_txt)
            probes.append({"prompt": pr_txt, "expected_probs": [float(x) for x in pr_p]})

        # Write method_actions.jsonl
        with (strat_dir / "method_actions.jsonl").open("w", encoding="utf-8") as fh:
            for s in steps:
                fh.write(json.dumps(s) + "\n")

        episode_artifacts[strat] = {
            "steps": steps,
            "committed_adapters": committed_adapters,
            "consistency_probes": probes,
        }
        print(f"V6_E2E_INPROC_DONE strat={strat:22s} steps={len(steps)} committed={len(committed_adapters)}", flush=True)

    # Free GPU VRAM completely before launching true isolated fresh-subprocess reload audits!
    del backend
    torch.cuda.empty_cache()

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
    for strat in strategies:
        strat_dir = run_dir / strat
        art = episode_artifacts[strat]
        job_path = strat_dir / "fresh_audit_job.json"
        out_path = strat_dir / "fresh_audit_result.json"
        job = {
            "model_path": cfg["model_path"],
            "backend_kind": backend_kind,
            "damaged_state_path": str(damaged_pt.resolve()),
            "expected_damaged_sha256": damaged_sha,
            "adapter_dirs": art["committed_adapters"],
            "consistency_probes": art["consistency_probes"],
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
            step_records=art["steps"],
            locality_audit=fresh_res["locality_audit"],
            fresh_reload_passed=bool(fresh_res["fresh_reload_passed"]),
            eval_gold_map=eval_gold_map,
            eval_role_map=eval_role_map,
        )
        ep_m["fresh_reload_max_prob_diff"] = fresh_res["max_prob_diff"]
        ep_m["fresh_reload_tolerance_passed"] = fresh_res["tolerance_passed"]
        ep_m["fresh_reload_n_adapters"] = fresh_res["n_adapters_reloaded"]
        all_metrics[strat] = ep_m
        print(
            f"V6_E2E_FRESH_VERIFIED strat={strat:22s} Y={ep_m['end_to_end_strict_safe_yield_Y']} "
            f"restored={ep_m['restored_in_D_raw']}/{ep_m['denominator_D']} "
            f"comm={ep_m['n_committed_writes']} ww={ep_m['wrong_write_count_committed']} "
            f"unk_att={ep_m['unknown_write_attempts']} harm={ep_m['harmed_healthy_count']} "
            f"loc={ep_m['strict_locality_pass']} max_diff={fresh_res['max_prob_diff']:.2e}",
            flush=True,
        )

    (run_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    (run_dir / "runtime.json").write_text(json.dumps({"elapsed_seconds": time.time() - t0}, indent=2), encoding="utf-8")
    print(f"V6_E2E_COMPLETE run_id={args.run_id} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
