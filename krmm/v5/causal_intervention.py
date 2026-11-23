"""Counterfactual causal interventions A-E (Section 5.3) with explicit rule-level vs model-level scope tag."""
from __future__ import annotations

from typing import Dict, List, Sequence

from krmm.v3.schemas import AtomicFact
from .support_structure import analyze_support_structure


def run_rule_level_interventions(
    domain: str,
    complete_facts: Sequence[AtomicFact],
    key_index: int,
    counterfactual_fact: AtomicFact,
    sham_fact: AtomicFact,
    correlated_wrong_facts: Sequence[AtomicFact],
) -> Dict[str, object]:
    """Execute all 5 required interventions (A-E) at rule-engine level; explicitly tagged NOT model-level."""
    base_with_sham = list(complete_facts) + [sham_fact]
    key_removed = [f for i, f in enumerate(base_with_sham) if i != key_index]
    cf_replaced = [counterfactual_fact if i == key_index else f for i, f in enumerate(complete_facts)]
    sham_removed = list(complete_facts)

    a_complete = analyze_support_structure(domain, base_with_sham)
    b_key_rem = analyze_support_structure(domain, key_removed)
    c_cf_rep = analyze_support_structure(domain, cf_replaced)
    d_sham_rem = analyze_support_structure(domain, sham_removed)
    e_corr_wrong = analyze_support_structure(domain, correlated_wrong_facts)

    return {
        "intervention_level": "rule_engine_only_NOT_model_causal",
        "A_complete_support": a_complete.to_dict(),
        "B_key_removal": b_key_rem.to_dict(),
        "C_counterfactual_replacement": c_cf_rep.to_dict(),
        "D_sham_removal": d_sham_rem.to_dict(),
        "E_correlated_wrong_support": e_corr_wrong.to_dict(),
        "sensitivity_checks": {
            "key_removal_changed_reliability_or_candidate": (
                b_key_rem.candidate_answer != a_complete.candidate_answer
                or b_key_rem.estimated_reliability < a_complete.estimated_reliability
            ),
            "counterfactual_changed_or_conflicted": (
                c_cf_rep.conflict_status or c_cf_rep.candidate_answer != a_complete.candidate_answer
            ),
            "sham_removal_invariant": (
                d_sham_rem.candidate_answer == a_complete.candidate_answer
                and abs(d_sham_rem.estimated_reliability - a_complete.estimated_reliability) < 1e-9
            ),
            "correlated_wrong_blocked_from_parametric": (
                e_corr_wrong.recommended_action != "ATTEMPT_PARAMETRIC_REPAIR"
            ),
        },
    }
