"""Verify all 5 causal support interventions (Section 5.3) behave as predicted."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.v4.run_cpu_mechanism import run_synthetic_mechanism_suite


def test_five_causal_interventions_and_eight_conditions():
    res = run_synthetic_mechanism_suite()
    ci = res["causal_interventions"]
    assert ci["complete_support"]["evidence_category"] == "redundant_distinct_paths"
    assert ci["complete_support"]["policy_action"] == "ATTEMPT_PARAMETRIC_REPAIR"

    # Key removal drops feathers -> degrades to single_chain -> lowers reliability and stops blind parametric repair
    assert ci["key_removal"]["evidence_category"] == "single_chain"
    assert ci["key_removal"]["estimated_reliability"] < ci["complete_support"]["estimated_reliability"]
    assert ci["key_removal"]["policy_action"] != "ATTEMPT_PARAMETRIC_REPAIR"

    # Counterfactual replacement -> conflict -> ABSTAIN
    assert ci["counterfactual_replacement"]["conflict_status"] is True
    assert ci["counterfactual_replacement"]["policy_action"] == "ABSTAIN"

    # Sham removal -> invariant
    assert ci["sham_removal"]["candidate_answer"] == ci["complete_support"]["candidate_answer"]
    assert ci["sham_removal"]["estimated_reliability"] == ci["complete_support"]["estimated_reliability"]

    # Correlated wrong support -> discounted & blocked from parametric writeback
    assert ci["correlated_wrong_support"]["evidence_category"] == "redundant_same_source"
    assert ci["correlated_wrong_support"]["policy_action"] != "ATTEMPT_PARAMETRIC_REPAIR"
