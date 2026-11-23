"""Verify correlated same-source redundancy and wrong consensus are discounted and blocked from blind writeback."""
from __future__ import annotations

from krmm.v3.schemas import AtomicFact
from krmm.v4.evidence_graph import EvidenceGraphAnalyzer
from krmm.v4.selective_policy import decide_repair


def test_spurious_same_source_redundancy_discounted_and_not_written():
    an = EvidenceGraphAnalyzer("vertebrate_class")
    # Two consistent but correlated wrong paraphrases of the same attribute from same model
    facts = [
        AtomicFact("shark", "covering", "fur_or_hair", "probe_v1", "text", 0.95, "damaged_model_internal"),
        AtomicFact("shark", "covering", "fur_or_hair", "probe_v2", "text", 0.95, "damaged_model_internal"),
    ]
    assess = an.assess(facts)
    assert assess.evidence_category == "redundant_same_source"
    assert assess.estimated_candidate_reliability < 0.65
    dec = decide_repair({"status": "suspected"}, assess, {"remaining_repairs": 5})
    assert dec["action"] != "ATTEMPT_PARAMETRIC_REPAIR"
    assert dec["action"] == "REQUEST_EXTERNAL_EVIDENCE"


def test_distinct_semantic_paths_notes_same_model_non_independence():
    an = EvidenceGraphAnalyzer("vertebrate_class")
    facts = [
        AtomicFact("eagle", "covering", "feathers", "p1", "text", 0.9, "damaged_model_internal"),
        AtomicFact("eagle", "thermoregulation", "warm_blooded", "p2", "text", 0.9, "damaged_model_internal"),
        AtomicFact("eagle", "reproduction", "hard_shelled_eggs", "p3", "text", 0.9, "damaged_model_internal"),
    ]
    assess = an.assess(facts)
    assert assess.evidence_category == "redundant_distinct_paths"
    assert assess.provenance_groups == ["damaged_model_internal"]
    assert any("NOT statistically independent" in n for n in assess.notes)
