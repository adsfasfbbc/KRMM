"""Verify diagnostic feature extraction & assessment reject hidden evaluator/post-repair keys."""
from __future__ import annotations

import pytest
from krmm.v3.repairability import FORBIDDEN_LEAKAGE_KEYS, assess_repairability, extract_diagnostic_features
from krmm.v3.schemas import CandidateReport


def _dummy_cand() -> CandidateReport:
    return CandidateReport(
        unit_id="u1",
        candidate_answer="A",
        candidate_source_type="structured_single_hop",
        atomic_facts=[],
        support_paths=[],
        ambiguity=0.0,
        conflict=False,
        key_intervention_result={"sensitive": True},
        sham_intervention_result={"stable": True},
        abstain_reason=None,
        query_cost=4,
    )


@pytest.mark.parametrize("bad_key", sorted(FORBIDDEN_LEAKAGE_KEYS))
def test_extract_features_rejects_forbidden_keys(bad_key):
    with pytest.raises(ValueError):
        extract_diagnostic_features(_dummy_cand(), None, {bad_key: "A"})


@pytest.mark.parametrize("bad_key", sorted(FORBIDDEN_LEAKAGE_KEYS))
def test_assess_repairability_rejects_forbidden_keys(bad_key):
    with pytest.raises(ValueError):
        assess_repairability(_dummy_cand(), None, {bad_key: True}, {})
