"""Batch 1B (G2-A): Real GPU Evidence Structure, Strong Baselines, and Model-Level Causal Interventions A-E."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from krmm.v3.damage import balanced_wrong_assignments, inject_damage, restore_weights, snapshot_weights
from krmm.v3.evidence import extract_atomic_facts
from krmm.v3.knowledge_registry import question_prompt, train_retain_prompts
from krmm.v3.mastery import HELDOUT_PERM, HELDOUT_VARIANTS, evaluate_unit_target, screen_clean_mastery
from krmm.v3.schemas import AtomicFact, KnowledgeUnit
from krmm.v5.support_structure import analyze_support_structure
from krmm.v5.v5_catalog import v5_known_evaluator_units


def _choose(probs: List[float]) -> str:
    return "ABCD"[int(np.argmax(probs))]


def _prompt_with_facts(unit: KnowledgeUnit, facts: List[AtomicFact], include_entity: bool = False) -> str:
    obs = "\n".join(f"- {f.attribute}: {f.raw_output or f.value}" for f in facts)
    opts = "\n".join(f"{k}: {v}" for k, v in unit.options.items())
    header = (
        f"Subject: {unit.entity_id}\nInfer the correct category using these observations:\n"
        if include_entity
        else "Infer the correct category using ONLY these observations:\n"
    )
    return f"{header}{obs}\n{opts}\nReply with one letter only."


def _counterfactual_opposite(domain: str, fact: AtomicFact) -> AtomicFact:
    opposites = {
        ("vertebrate_class", "covering"): ("feathers", "scales") if fact.value != "scales" else ("scales", "feathers"),
        ("vertebrate_class", "respiration"): ("gills", "lungs") if fact.value == "gills" else ("lungs", "gills"),
        ("vertebrate_class", "thermoregulation"): ("warm_blooded", "cold_blooded") if fact.value == "warm_blooded" else ("cold_blooded", "warm_blooded"),
        ("everyday_material", "conductivity"): ("conductor", "insulator") if fact.value == "conductor" else ("insulator", "conductor"),
        ("everyday_material", "origin"): ("inorganic", "biological_plant") if fact.value == "inorganic" else ("biological_plant", "inorganic"),
        ("sound_source_mechanism", "oscillator"): ("stretched_string", "air_column") if fact.value == "stretched_string" else ("air_column", "stretched_string"),
    }
    new_val = opposites.get((domain, fact.attribute), (fact.value, "gills" if domain == "vertebrate_class" else "insulator"))[1]
    return AtomicFact(
        entity=fact.entity,
        attribute=fact.attribute,
        value=new_val,
        source="counterfactual_intervention",
        modality="text",
        confidence=1.0,
        provenance_group="counterfactual_intervention",
        raw_output=f"The {fact.attribute} is {new_val.replace('_', ' ')}.",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/root/rivermind-data/models/Qwen2.5-Omni-3B")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--damage-steps", type=int, default=64)
    ap.add_argument("--out", default="results/v5-new/g2_mechanism_results.json")
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
        "damage_retain_weight": 50.0,
        "damage_l2_weight": 1e-4,
        "retain_batch_size": 24,
        "seed": args.seed,
    }
    omni = Omni(args.model_path)
    clean_state = snapshot_weights(omni, cfg)
    retains = train_retain_prompts(include_multi_template=True)
    teacher_logits = omni.retain_logits(retains)

    evs = v5_known_evaluator_units()
    assign, assign_audit = balanced_wrong_assignments(evs, seed=args.seed)

    unit_rows: List[dict] = []
    for idx, ev in enumerate(evs, 1):
        restore_weights(omni, clean_state)
        u = ev.unit
        m = screen_clean_mastery(omni, ev, min_choice_mass=0.25)
        if not m["stably_mastered"]:
            continue

        # Single-target D1 damage
        inject_damage(omni, u, ev.gold_answer, assign[u.unit_id], retains, teacher_logits, cfg, damage_type="D1_wrong_association")
        post_ev = evaluate_unit_target(omni, ev.unit, ev.gold_answer, variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,))
        if post_ev["accuracy"] > 0.5:
            continue

        # Extract real atomic facts from damaged model
        facts, _ = extract_atomic_facts(omni, u, max_new_tokens=24)
        valid_facts = [f for f in facts if f.value != "unparsed"]
        sup = analyze_support_structure(u.domain, facts)

        # Strong baselines (Section 5.4) on the damaged model
        p2, _, _ = question_prompt(u, variant=2)
        p3, _, _ = question_prompt(u, variant=3)
        p_perm, d2c, _ = question_prompt(u, variant=2, permutation=("D", "C", "B", "A"))
        probs2, _ = omni.probabilities(p2)
        probs3, _ = omni.probabilities(p3)
        raw_pp, _ = omni.probabilities(p_perm)
        direct_cand = _choose(probs2)
        # Self-consistency majority over 3 direct probes
        votes = [_choose(probs2), _choose(probs3), d2c[_choose(raw_pp)]]
        sc_cand = max(set(votes), key=votes.count)
        # Entity-name-only prompt
        ent_only_p = f"Entity: {u.entity_id}\n" + "\n".join(f"{k}: {v}" for k, v in u.options.items()) + "\nReply with one letter only."
        ent_only_cand = _choose(omni.probabilities(ent_only_p)[0])
        # Generic CoT (generate brief reasoning then classify)
        cot_prompt = f"Think briefly step-by-step about what {u.entity_id} is, then pick A, B, C, or D:\n" + "\n".join(f"{k}: {v}" for k, v in u.options.items())
        cot_text = omni.generate(cot_prompt, max_new_tokens=48)
        cot_cls_p = f"{cot_prompt}\nReasoning: {cot_text}\nFinal answer letter only:"
        cot_cand = _choose(omni.probabilities(cot_cls_p)[0])
        # Generative reconstruction from extracted facts
        gen_probs, _ = omni.probabilities(_prompt_with_facts(u, facts, include_entity=False))
        gen_cand = _choose(gen_probs)
        # Rule-only (zero model facts -> always unidentifiable)
        rule_only_cand = analyze_support_structure(u.domain, []).candidate_answer

        # REAL MODEL-LEVEL CAUSAL INTERVENTIONS A-E (Section 5.3)
        sham = AtomicFact(u.entity_id, "unrelated_container_color", "brown", "sham", "text", 1.0, "sham", "The storage container is brown.")
        key_idx = sup.minimal_support_sets[0]["indices"][0] if sup.minimal_support_sets else 0
        key_fact = valid_facts[key_idx] if valid_facts and key_idx < len(valid_facts) else (facts[0] if facts else sham)
        cf_fact = _counterfactual_opposite(u.domain, key_fact)

        p_A = omni.probabilities(_prompt_with_facts(u, valid_facts + [sham]))[0]
        p_B = omni.probabilities(_prompt_with_facts(u, [f for i, f in enumerate(valid_facts) if i != key_idx] + [sham]))[0]
        p_C = omni.probabilities(_prompt_with_facts(u, [cf_fact if i == key_idx else f for i, f in enumerate(valid_facts)]))[0]
        p_D = omni.probabilities(_prompt_with_facts(u, valid_facts))[0]
        p_E = omni.probabilities(_prompt_with_facts(u, [cf_fact, cf_fact]))[0]

        cand_A, cand_B, cand_C, cand_D, cand_E = _choose(p_A), _choose(p_B), _choose(p_C), _choose(p_D), _choose(p_E)
        gold_idx = "ABCD".index(ev.gold_answer)

        unit_rows.append(
            {
                "unit_id": u.unit_id,
                "entity_id": u.entity_id,
                "domain": u.domain,
                "gold_evaluator_only": ev.gold_answer,
                "support_class": sup.support_class,
                "n_minimal_sets": sup.n_minimal_sets,
                "dependency_overlap": sup.dependency_overlap,
                "estimated_reliability": sup.estimated_reliability,
                "recommended_action": sup.recommended_action,
                "baselines": {
                    "rule_only": rule_only_cand == ev.gold_answer,
                    "entity_name_only": ent_only_cand == ev.gold_answer,
                    "direct_probing": direct_cand == ev.gold_answer,
                    "self_consistency": sc_cand == ev.gold_answer,
                    "generic_cot": cot_cand == ev.gold_answer,
                    "generative_reconstruction": gen_cand == ev.gold_answer,
                    "existing_evidence_graph_correct": sup.candidate_answer == ev.gold_answer,
                    "existing_evidence_graph_covered": sup.candidate_answer is not None,
                    "evidence_gated_accepted": sup.recommended_action == "ATTEMPT_PARAMETRIC_REPAIR",
                    "evidence_gated_correct": (sup.recommended_action == "ATTEMPT_PARAMETRIC_REPAIR" and sup.candidate_answer == ev.gold_answer),
                },
                "model_level_interventions": {
                    "A_complete_pred": cand_A,
                    "B_key_removal_pred": cand_B,
                    "C_counterfactual_pred": cand_C,
                    "D_sham_removal_pred": cand_D,
                    "E_correlated_wrong_pred": cand_E,
                    "A_gold_prob": float(p_A[gold_idx]),
                    "B_gold_prob": float(p_B[gold_idx]),
                    "C_gold_prob": float(p_C[gold_idx]),
                    "D_gold_prob": float(p_D[gold_idx]),
                    "E_gold_prob": float(p_E[gold_idx]),
                    "key_removal_changed_pred": cand_B != cand_A,
                    "counterfactual_changed_pred": cand_C != cand_A,
                    "sham_removal_invariant_pred": cand_D == cand_A,
                    "key_prob_drop": float(p_A[gold_idx] - p_B[gold_idx]),
                    "cf_prob_drop": float(p_A[gold_idx] - p_C[gold_idx]),
                    "sham_prob_abs_diff": float(abs(p_A[gold_idx] - p_D[gold_idx])),
                },
            }
        )
        print(f"G2A_UNIT {idx}/{len(evs)} {u.entity_id:24s} cls={sup.support_class:28s} ok={sup.candidate_answer == ev.gold_answer} cf_chg={cand_C != cand_A} sham_inv={cand_D == cand_A}", flush=True)

    restore_weights(omni, clean_state)

    # Aggregate summary
    n = len(unit_rows)
    by_class: Dict[str, Dict[str, int]] = {}
    for r in unit_rows:
        b = by_class.setdefault(r["support_class"], {"total": 0, "correct": 0, "accepted": 0})
        b["total"] += 1
        b["correct"] += int(r["baselines"]["existing_evidence_graph_correct"])
        b["accepted"] += int(r["baselines"]["evidence_gated_accepted"])

    base_summary = {
        k: float(np.mean([r["baselines"][k] for r in unit_rows])) if n else 0.0
        for k in (
            "rule_only", "entity_name_only", "direct_probing", "self_consistency",
            "generic_cot", "generative_reconstruction", "existing_evidence_graph_correct",
            "existing_evidence_graph_covered", "evidence_gated_accepted", "evidence_gated_correct",
        )
    }
    n_eg_acc = sum(int(r["baselines"]["evidence_gated_accepted"]) for r in unit_rows)
    base_summary["evidence_gated_precision_when_accepted"] = (
        sum(int(r["baselines"]["evidence_gated_correct"]) for r in unit_rows) / max(1, n_eg_acc)
    )
    int_summary = {
        "key_removal_pred_change_rate": float(np.mean([r["model_level_interventions"]["key_removal_changed_pred"] for r in unit_rows])) if n else 0.0,
        "counterfactual_pred_change_rate": float(np.mean([r["model_level_interventions"]["counterfactual_changed_pred"] for r in unit_rows])) if n else 0.0,
        "sham_removal_invariant_rate": float(np.mean([r["model_level_interventions"]["sham_removal_invariant_pred"] for r in unit_rows])) if n else 0.0,
        "mean_key_gold_prob_drop": float(np.mean([r["model_level_interventions"]["key_prob_drop"] for r in unit_rows])) if n else 0.0,
        "mean_cf_gold_prob_drop": float(np.mean([r["model_level_interventions"]["cf_prob_drop"] for r in unit_rows])) if n else 0.0,
        "mean_sham_gold_prob_abs_diff": float(np.mean([r["model_level_interventions"]["sham_prob_abs_diff"] for r in unit_rows])) if n else 0.0,
    }

    out = {
        "n_damaged_evaluated": n,
        "by_support_class": by_class,
        "baseline_summary": base_summary,
        "model_level_intervention_summary": int_summary,
        "assignment_audit": assign_audit,
        "per_unit": unit_rows,
        "elapsed_seconds": time.time() - t0,
    }
    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"G2A_COMPLETE n={n} elapsed={out['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
