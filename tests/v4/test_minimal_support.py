"""Verify exact minimal support-set enumeration and 7-way v0.4 evidence classification."""
from __future__ import annotations

from krmm.v3.schemas import AtomicFact
from krmm.v4.evidence_graph import EvidenceGraphAnalyzer


def _f(attr, val, prov="damaged_model_internal", src="probe"):
    return AtomicFact("anon", attr, val, src, "text", 0.9, prov)


def test_all_seven_evidence_categories():
    an = EvidenceGraphAnalyzer("vertebrate_class", study_condition="internal_only")
    an_aug = EvidenceGraphAnalyzer("vertebrate_class", study_condition="evidence_augmented")

    assert an.assess([_f("thermoregulation", "warm_blooded")]).evidence_category == "insufficient_support"
    assert an.assess([_f("covering", "feathers")]).evidence_category == "single_decisive"
    assert an.assess([_f("thermoregulation", "cold_blooded"), _f("respiration", "lungs")]).evidence_category == "single_chain"
    assert an.assess([_f("covering", "feathers", src="p1"), _f("covering", "feathers", src="p2")]).evidence_category == "redundant_same_source"
    assert an.assess([_f("covering", "feathers"), _f("thermoregulation", "warm_blooded"), _f("reproduction", "hard_shelled_eggs")]).evidence_category == "redundant_distinct_paths"
    assert an_aug.assess([_f("covering", "feathers", prov="m1"), _f("thermoregulation", "warm_blooded", prov="ext"), _f("reproduction", "hard_shelled_eggs", prov="ext")]).evidence_category == "independent_source_support"
    assert an.assess([_f("thermoregulation", "warm_blooded"), _f("respiration", "gills")]).evidence_category == "conflicting_support"


def test_identifiable_never_auto_assigns_reliability_one():
    an = EvidenceGraphAnalyzer("vertebrate_class")
    a = an.assess([_f("covering", "feathers")])
    assert a.identifiable is True
    assert 0.0 < a.estimated_candidate_reliability < 1.0
