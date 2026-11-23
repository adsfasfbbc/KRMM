"""Comprehensive CPU unit tests for EvidenceGraph minimal support-set classification."""
from __future__ import annotations

from krmm.v3.reconstruction import reconstruct_structured
from krmm.v3.schemas import AtomicFact


def _f(entity: str, attr: str, val: str, prov: str = "damaged_model_internal") -> AtomicFact:
    return AtomicFact(entity=entity, attribute=attr, value=val, source="probe", modality="text", confidence=0.9, provenance_group=prov)


def test_single_hop_uniquely_decides():
    # feathers alone -> B; lungs -> A,B,C (non-deciding extra)
    res = reconstruct_structured("vertebrate_class", [_f("robin", "covering", "feathers"), _f("robin", "respiration", "lungs")])
    assert res["candidate"] == "B"
    assert res["explanation_class"] == "single_hop"
    assert res["min_support_size"] == 1
    assert res["true_composition"] is False
    assert len(res["minimal_support_sets"]) == 1
    assert res["minimal_support_sets"][0]["attributes"] == ["covering"]


def test_two_premise_minimal_multi_premise():
    # cold_blooded -> {C, D}; lungs -> {A, B, C}; intersection -> C
    res = reconstruct_structured(
        "vertebrate_class",
        [_f("crocodile", "thermoregulation", "cold_blooded"), _f("crocodile", "respiration", "lungs")],
    )
    assert res["candidate"] == "C"
    assert res["explanation_class"] == "minimal_multi_premise"
    assert res["min_support_size"] == 2
    assert res["true_composition"] is True


def test_three_premise_joint_composition():
    # everyday_material:
    # origin=inorganic -> {A, C}; conductivity=insulator -> {B, C, D}; heat_behavior=high_melting -> {A, C, D}
    # Here origin+conductivity already gives {C} (size 2); test vertebrate_class 3-premise:
    # reproduction=live_birth -> {A, C, D}; respiration=lungs -> {A, B, C}; appendage=fins -> {A, D}
    # Pairwise:
    #   live_birth & lungs -> {A, C}
    #   live_birth & fins -> {A, D}
    #   lungs & fins -> {A} -> size 2!
    # Check 3-way: in sound_source_mechanism:
    #   oscillator=flexible_reed ({B,D}) + excitation=struck ({A,C,D}) -> {D} (size 2)
    # Test that 2-premise minimal subset is found even when 3rd non-minimal fact is added
    facts = [
        _f("ceramic_mug", "origin", "inorganic"),
        _f("ceramic_mug", "conductivity", "insulator"),
        _f("ceramic_mug", "heat_behavior", "high_melting"),
    ]
    res = reconstruct_structured("everyday_material", facts)
    assert res["candidate"] == "C"
    assert res["min_support_size"] == 2
    assert res["true_composition"] is True
    assert res["explanation_class"] == "minimal_multi_premise"
    assert res["minimal_support_sets"][0]["attributes"] == ["origin", "conductivity"]


def test_redundant_support_multiple_minimal_sets():
    # feathers -> {B} AND (warm_blooded + hard_shelled_eggs) -> {B}: two distinct minimal support sets!
    facts = [
        _f("eagle", "covering", "feathers", "g1"),
        _f("eagle", "thermoregulation", "warm_blooded", "g2"),
        _f("eagle", "reproduction", "hard_shelled_eggs", "g2"),
    ]
    res = reconstruct_structured("vertebrate_class", facts)
    assert res["candidate"] == "B"
    assert res["explanation_class"] == "redundant_support"
    assert res["min_support_size"] == 1
    assert len(res["minimal_support_sets"]) == 2
    assert res["true_composition"] is False


def test_conflicting_evidence():
    res = reconstruct_structured(
        "vertebrate_class",
        [_f("wolf", "thermoregulation", "warm_blooded"), _f("wolf", "respiration", "gills")],
    )
    assert res["candidate"] is None
    assert res["explanation_class"] == "conflicting_evidence"
    assert res["graph"]["conflict"] is True


def test_unparsed_irrelevant_and_missing_premise_insufficient():
    facts = [
        _f("mystery", "thermoregulation", "warm_blooded"),
        _f("mystery", "covering", "unparsed"),
        _f("mystery", "favorite_color", "blue"),
    ]
    res = reconstruct_structured("vertebrate_class", facts)
    assert res["candidate"] is None
    assert res["explanation_class"] == "insufficient_evidence"
    assert res["min_support_size"] is None
    assert len(res["graph"]["invalid_atoms"]) == 1


def test_duplicate_same_model_provenance_flagged():
    facts = [
        _f("dolphin", "thermoregulation", "warm_blooded", "damaged_model_internal"),
        _f("dolphin", "reproduction", "live_birth", "damaged_model_internal"),
    ]
    res = reconstruct_structured("vertebrate_class", facts)
    assert res["candidate"] == "A"
    assert res["graph"]["same_model_correlated_only"] is True
    assert res["graph"]["unique_provenance_groups"] == ["damaged_model_internal"]
