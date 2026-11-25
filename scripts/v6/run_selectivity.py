#!/usr/bin/env python3
"""P3 / G2-A: Fixed-Candidate Selectivity, Risk-Coverage (AURC, 25/50/75%) & Full-Policy Interventions A-F."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from krmm.v3.damage import restore_weights
from krmm.v3.evidence import extract_atomic_facts
from krmm.v3.knowledge_registry import question_prompt
from krmm.v3.schemas import AtomicFact, KnowledgeUnit
from krmm.v5.support_structure import analyze_support_structure
from krmm.v6.policy import compute_risk_coverage_curve, select_action
from krmm.v6.schemas import PublicUnit
from krmm.v6.v6_catalog import TWIN_WORLD_PAIRS


def _argmax_letter(probs: List[float]) -> str:
    return "ABCD"[int(np.argmax(probs))]


def _counterfactual_fact(domain: str, f: AtomicFact) -> AtomicFact:
    ent = getattr(f, "entity", "unknown")
    if domain == "vertebrate_class":
        return AtomicFact(ent, "body_covering", "feathers", "cf", "text", 1.0, "cf", "feathers") if f.value != "feathers" else AtomicFact(ent, "body_covering", "scales", "cf", "text", 1.0, "cf", "scales")
    if domain == "everyday_material":
        return AtomicFact(ent, "material_class", "metal", "cf", "text", 1.0, "cf", "metal") if f.value != "metal" else AtomicFact(ent, "material_class", "wood", "cf", "text", 1.0, "cf", "wood")
    return AtomicFact(ent, "sound_source", "vibrating air column", "cf", "text", 1.0, "cf", "air column") if "air" not in str(f.value) else AtomicFact(ent, "sound_source", "stretched strings", "cf", "text", 1.0, "cf", "strings")


def main() -> None:
    ap = argparse.ArgumentParser(description="KRMM v0.6 G2-A Fixed-Candidate Selectivity & Interventions A-F")
    ap.add_argument("--config", default="configs/v6/pilot.json")
    ap.add_argument("--discovery-run-id", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    t0 = time.time()
    cfg = json.loads((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    disc_dir = REPO_ROOT / "results" / "v6-new" / args.discovery_run_id
    run_dir = REPO_ROOT / "results" / "v6-new" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    pub_manifest = json.loads((REPO_ROOT / cfg["public_manifest"]).read_text(encoding="utf-8"))
    hidden_manifest = json.loads((REPO_ROOT / cfg["evaluator_hidden"]).read_text(encoding="utf-8"))
    truth_records = json.loads((disc_dir / "evaluator_only" / "truth_records.json").read_text(encoding="utf-8"))

    pub_map = {u["unit_id"]: PublicUnit(**u) for u in pub_manifest["public_units"]}
    eval_map = {r["unit_id"]: r for r in hidden_manifest["evaluator_records"]}

    from krmm.model import Omni
    backend = Omni(cfg["model_path"])
    damaged_state = torch.load(disc_dir / "damaged_state.pt", map_location="cpu", weights_only=True)
    restore_weights(backend, damaged_state)

    # Evaluate on ALL degraded targets D + all healthy/unknown controls for full-pool selectivity!
    degraded_records = [r for r in truth_records if r["is_degraded_target"]]
    fixed_cand_rows: List[dict] = []
    intervention_rows: List[dict] = []

    for r in degraded_records:
        uid = r["unit_id"]
        pu = pub_map[uid]
        er = eval_map[uid]
        gold = er["gold_answer"]
        ku = pu.to_knowledge_unit("test")

        p2, _, _ = question_prompt(ku, variant=2)
        probs2, _ = backend.probabilities(p2)
        dir_cand = _argmax_letter(probs2)
        dir_conf = float(max(probs2))

        facts, _ = extract_atomic_facts(backend, ku, max_new_tokens=24)
        valid_facts = [f for f in facts if f.value != "unparsed"]
        sup = analyze_support_structure(ku.domain, facts)

        obs = "\n".join(f"- {f.attribute}: {f.raw_output or f.value}" for f in facts)
        opts = "\n".join(f"{k}: {v}" for k, v in ku.options.items())
        gen_probs, _ = backend.probabilities(f"Infer the correct category using ONLY these observations:\n{obs}\n{opts}\nReply with one letter only.")
        gen_cand = _argmax_letter(gen_probs)

        # Shared fixed structured candidate for selector comparison (Section 6.1)
        cand = sup.candidate_answer
        is_wrong = (cand != gold) if cand is not None else True
        rand_score = int(hashlib.sha256(f"rc_rand:{seed}:{uid}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF

        fixed_cand_rows.append({
            "unit_id": uid,
            "entity_id": pu.entity_id,
            "domain": pu.domain,
            "support_class": sup.support_class,
            "candidate": cand,
            "generative_candidate": gen_cand,
            "gold_evaluator_only": gold,
            "candidate_is_wrong": is_wrong,
            "generative_is_wrong": gen_cand != gold,
            "score_random": rand_score,
            "score_direct_confidence": dir_conf,
            "score_candidate_change": float(cand is not None and cand != dir_cand),
            "score_fact_count": float(len(valid_facts)),
            "score_minset_count": float(sup.n_minimal_sets),
            "score_evidence_gate": float(sup.estimated_reliability),
        })

        # Controlled Interventions A-E acting on FULL GATING POLICY ACTION (Section 6.2)
        sham = AtomicFact(pu.entity_id, "storage_tag_color", "gray", "sham", "text", 1.0, "sham", "gray")
        # Hitting set removal for B: remove 1 fact from EACH minimal support set so uniqueness truly vanishes
        hit_indices = {ms["indices"][0] for ms in sup.minimal_support_sets if ms.get("indices")} if sup.minimal_support_sets else ({0} if valid_facts else set())
        facts_A = list(valid_facts) + [sham]
        facts_B = [f for i, f in enumerate(valid_facts) if i not in hit_indices] + [sham]
        cf_f = _counterfactual_fact(ku.domain, valid_facts[min(hit_indices)] if valid_facts and hit_indices else sham)
        facts_C = [cf_f if i in hit_indices else f for i, f in enumerate(valid_facts)]
        facts_D = list(valid_facts)  # sham removed
        facts_E = [cf_f, cf_f]  # correlated wrong duplicates

        def _gate_act(fset: List[AtomicFact]) -> Dict[str, object]:
            s_obj = analyze_support_structure(ku.domain, fset)
            dec = select_action(
                strategy="M_evidence_gated", unit_id=uid, direct_candidate=dir_cand,
                direct_confidence=dir_conf, generative_candidate=gen_cand,
                structured_candidate=s_obj.candidate_answer, support_class=s_obj.support_class,
                n_valid_facts=len([x for x in fset if x.value != "unparsed"]),
                n_minimal_sets=s_obj.n_minimal_sets, heuristic_support_score=s_obj.estimated_reliability,
                remaining_write_budget=8, seed=seed,
            )
            return {"action": dec["action"], "candidate": dec["candidate"], "support_class": s_obj.support_class}

        gA, gB, gC, gD, gE = _gate_act(facts_A), _gate_act(facts_B), _gate_act(facts_C), _gate_act(facts_D), _gate_act(facts_E)
        intervention_rows.append({
            "unit_id": uid,
            "entity_id": pu.entity_id,
            "A_complete": gA,
            "B_hitting_set_removed": gB,
            "C_counterfactual_replaced": gC,
            "D_sham_removed": gD,
            "E_correlated_wrong_dup": gE,
            "sham_action_invariant": (gD["action"] == gA["action"] and gD["candidate"] == gA["candidate"]),
            "hitting_set_stops_accept": (gA["action"] == "ATTEMPT_REPAIR" and gB["action"] != "ATTEMPT_REPAIR"),
            "cf_changes_action_or_cand": (gC["action"] != gA["action"] or gC["candidate"] != gA["candidate"]),
            "correlated_wrong_fooled_gate": (gE["action"] == "ATTEMPT_REPAIR" and gE["candidate"] != gold),
        })

    # Intervention F: Twin-World Unidentifiable Negative Controls (Section 2.2 & 6.2)
    twin_results = []
    for tw in TWIN_WORLD_PAIRS:
        tw_facts = [
            AtomicFact(tw["pair_id"], f["attribute"], f["value"], "tw", "text", 1.0, "tw", f["value"])
            for f in tw["shared_visible_facts"]
        ]
        s_tw = analyze_support_structure(tw["domain"], tw_facts)
        dec_tw = select_action(
            strategy="M_evidence_gated", unit_id=tw["pair_id"], direct_candidate="A",
            direct_confidence=0.5, generative_candidate="A", structured_candidate=s_tw.candidate_answer,
            support_class=s_tw.support_class, n_valid_facts=len(tw_facts),
            n_minimal_sets=s_tw.n_minimal_sets, heuristic_support_score=s_tw.estimated_reliability,
            remaining_write_budget=8, seed=seed,
        )
        twin_results.append({
            "pair_id": tw["pair_id"],
            "support_class": s_tw.support_class,
            "action": dec_tw["action"],
            "correctly_abstained_or_requested": dec_tw["action"] in ("ABSTAIN", "REQUEST_EVIDENCE"),
        })

    # Compute Risk-Coverage curves across all 6 selectors on the EXACT SAME fixed structured candidate set
    rc_curves = {}
    for s_key in ("score_random", "score_direct_confidence", "score_candidate_change", "score_fact_count", "score_minset_count", "score_evidence_gate"):
        rc_curves[s_key] = compute_risk_coverage_curve(fixed_cand_rows, score_key=s_key, seed=seed)

    accepted_A = [x for x in intervention_rows if x["A_complete"]["action"] == "ATTEMPT_REPAIR"]
    int_summary = {
        "n_degraded_D": len(intervention_rows),
        "n_accepted_at_A": len(accepted_A),
        "sham_action_invariance_rate": float(np.mean([x["sham_action_invariant"] for x in intervention_rows])) if intervention_rows else None,
        "hitting_set_removal_abstain_rate_given_A_accept": (
            float(np.mean([x["hitting_set_stops_accept"] for x in accepted_A])) if accepted_A else None
        ),
        "cf_action_or_cand_change_rate": float(np.mean([x["cf_changes_action_or_cand"] for x in intervention_rows])) if intervention_rows else None,
        "correlated_wrong_fooled_gate_rate": float(np.mean([x["correlated_wrong_fooled_gate"] for x in intervention_rows])) if intervention_rows else None,
        "twin_world_negative_control_pass_rate": float(np.mean([x["correctly_abstained_or_requested"] for x in twin_results])),
    }

    out = {
        "run_id": args.run_id,
        "discovery_run_id": args.discovery_run_id,
        "n_degraded_D": len(fixed_cand_rows),
        "candidate_generation_comparison": {
            "generative_accuracy": float(np.mean([not r["generative_is_wrong"] for r in fixed_cand_rows])) if fixed_cand_rows else None,
            "structured_accuracy_overall": float(np.mean([not r["candidate_is_wrong"] for r in fixed_cand_rows])) if fixed_cand_rows else None,
            "structured_coverage": float(np.mean([r["candidate"] is not None for r in fixed_cand_rows])) if fixed_cand_rows else None,
        },
        "risk_coverage_curves": rc_curves,
        "full_policy_intervention_summary": int_summary,
        "twin_world_results": twin_results,
        "per_unit_fixed_candidate": fixed_cand_rows,
        "per_unit_interventions": intervention_rows,
        "elapsed_seconds": time.time() - t0,
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"V6_SELECTIVITY_COMPLETE D={len(fixed_cand_rows)} sham_inv={int_summary['sham_action_invariance_rate']} hit_stop={int_summary['hitting_set_removal_abstain_rate_given_A_accept']} twin_pass={int_summary['twin_world_negative_control_pass_rate']}", flush=True)


if __name__ == "__main__":
    main()
