"""CPU-only synthetic mechanism suite: 8 evidence structures + 5 causal interventions + shortcut baselines (P2)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from krmm.v3.schemas import AtomicFact, KnowledgeUnit
from krmm.v4.evidence_graph import EvidenceGraphAnalyzer
from krmm.v4.selective_policy import decide_repair


def _f(entity: str, attr: str, val: str, source: str = "model_probe", prov: str = "damaged_model_internal") -> AtomicFact:
    return AtomicFact(entity=entity, attribute=attr, value=val, source=source, modality="text", confidence=0.9, provenance_group=prov)


def run_synthetic_mechanism_suite() -> Dict[str, object]:
    """Run all 8 required evidence-structure conditions and 5 causal interventions purely on CPU."""
    analyzer_int = EvidenceGraphAnalyzer("vertebrate_class", study_condition="internal_only")
    analyzer_aug = EvidenceGraphAnalyzer("vertebrate_class", study_condition="evidence_augmented")
    dmg_suspected = {"status": "suspected", "anomaly_score": 0.9}
    budget = {"remaining_repairs": 5, "allow_nonparametric": True}

    # 8 stratified conditions (Section 5.2)
    conditions = {
        "1_no_decisive_evidence": (
            analyzer_int,
            [_f("anon_ent_1", "thermoregulation", "warm_blooded")],
            "A",
        ),
        "2_single_decisive": (
            analyzer_int,
            [_f("anon_ent_2", "covering", "feathers")],
            "B",
        ),
        "3_single_chain_multi_premise": (
            analyzer_int,
            [_f("anon_ent_3", "thermoregulation", "cold_blooded"), _f("anon_ent_3", "respiration", "lungs")],
            "C",
        ),
        "4_same_source_spurious_redundancy": (
            analyzer_int,
            [
                _f("anon_ent_4", "covering", "feathers", source="probe_v1", prov="damaged_model_internal"),
                _f("anon_ent_4", "covering", "feathers", source="probe_v2", prov="damaged_model_internal"),
            ],
            "B",
        ),
        "5_distinct_semantic_paths_same_model": (
            analyzer_int,
            [
                _f("anon_ent_5", "covering", "feathers", prov="damaged_model_internal"),
                _f("anon_ent_5", "thermoregulation", "warm_blooded", prov="damaged_model_internal"),
                _f("anon_ent_5", "reproduction", "hard_shelled_eggs", prov="damaged_model_internal"),
            ],
            "B",
        ),
        "6_independent_source_support": (
            analyzer_aug,
            [
                _f("anon_ent_6", "covering", "feathers", source="text_probe", prov="damaged_model_internal"),
                _f("anon_ent_6", "thermoregulation", "warm_blooded", source="external_kb", prov="external_verified_source"),
                _f("anon_ent_6", "reproduction", "hard_shelled_eggs", source="external_kb", prov="external_verified_source"),
            ],
            "B",
        ),
        "7_correlated_wrong_consensus": (
            analyzer_int,
            # Gold is D (fish), but model hallucinates two consistent mammal-like paraphrases of the same attribute
            [
                _f("anon_ent_7", "covering", "fur_or_hair", source="probe_v1", prov="damaged_model_internal"),
                _f("anon_ent_7", "covering", "fur_or_hair", source="probe_v2", prov="damaged_model_internal"),
            ],
            "D",
        ),
        "8_conflicting_premises": (
            analyzer_int,
            [
                _f("anon_ent_8", "thermoregulation", "warm_blooded"),
                _f("anon_ent_8", "respiration", "gills"),
            ],
            "A",
        ),
    }

    cond_results = {}
    for cname, (an, facts, gold) in conditions.items():
        assess = an.assess(facts)
        dec = decide_repair(dmg_suspected, assess, budget)
        cond_results[cname] = {
            "evidence_category": assess.evidence_category,
            "identifiable": assess.identifiable,
            "candidate_answer": assess.candidate_answer,
            "gold_synthetic": gold,
            "candidate_matches_synthetic_gold": assess.candidate_answer == gold,
            "n_minimal_sets": assess.number_of_support_paths,
            "dependency_overlap": assess.dependency_overlap,
            "provenance_groups": assess.provenance_groups,
            "estimated_reliability": assess.estimated_candidate_reliability,
            "policy_action": dec["action"],
            "policy_reason": dec["reason"],
        }

    # 5 causal interventions (Section 5.3) on a corroborated target (bird: feathers + warm_blooded + hard_shelled_eggs)
    base_facts = [
        _f("anon_bird", "covering", "feathers", prov="damaged_model_internal"),
        _f("anon_bird", "thermoregulation", "warm_blooded", prov="damaged_model_internal"),
        _f("anon_bird", "reproduction", "hard_shelled_eggs", prov="damaged_model_internal"),
        _f("anon_bird", "container_color", "brown", source="sham", prov="sham"),
    ]
    interventions = {
        "complete_support": base_facts,
        "key_removal": [base_facts[1], base_facts[2], base_facts[3]],  # drops feathers -> becomes single_chain!
        "counterfactual_replacement": [
            _f("anon_bird", "covering", "scales", prov="damaged_model_internal"),
            base_facts[1],
            base_facts[2],
        ],  # conflict!
        "sham_removal": [base_facts[0], base_facts[1], base_facts[2]],  # invariant!
        "correlated_wrong_support": [
            _f("anon_bird", "covering", "fur_or_hair", source="p1", prov="damaged_model_internal"),
            _f("anon_bird", "covering", "fur_or_hair", source="p2", prov="damaged_model_internal"),
        ],  # redundant_same_source -> discounted & blocked from parametric write!
    }
    int_results = {}
    for iname, ifacts in interventions.items():
        a = analyzer_int.assess(ifacts)
        d = decide_repair(dmg_suspected, a, budget)
        int_results[iname] = {
            "evidence_category": a.evidence_category,
            "candidate_answer": a.candidate_answer,
            "identifiable": a.identifiable,
            "conflict_status": a.conflict_status,
            "estimated_reliability": a.estimated_candidate_reliability,
            "policy_action": d["action"],
        }

    return {
        "scope_warning": "SYNTHETIC_CPU_MECHANISM_ONLY — NOT real neural model repair performance.",
        "stratified_conditions": cond_results,
        "causal_interventions": int_results,
    }


def main() -> None:
    res = run_synthetic_mechanism_suite()
    out_p = REPO_ROOT / "results/v4-new/cpu_mechanism_results.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
