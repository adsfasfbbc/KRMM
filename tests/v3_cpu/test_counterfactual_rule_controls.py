"""Verify rule-level counterfactual controls (key-premise deletion & sham-premise invariance)."""
from __future__ import annotations

from krmm.v3.reconstruction import reconstruct_structured
from krmm.v3.schemas import AtomicFact


def test_key_premise_deletion_breaks_candidate_and_sham_is_invariant():
    f1 = AtomicFact("crocodile", "thermoregulation", "cold_blooded", "p", "text", 0.9, "g1")
    f2 = AtomicFact("crocodile", "respiration", "lungs", "p", "text", 0.9, "g2")
    sham = AtomicFact("crocodile", "unrelated_container_color", "brown", "sham_control", "text", 1.0, "sham")

    full = reconstruct_structured("vertebrate_class", [f1, f2])
    drop_f1 = reconstruct_structured("vertebrate_class", [f2])
    drop_f2 = reconstruct_structured("vertebrate_class", [f1])
    with_sham = reconstruct_structured("vertebrate_class", [f1, f2, sham])

    assert full["candidate"] == "C"
    assert drop_f1["candidate"] is None
    assert drop_f2["candidate"] is None
    assert with_sham["candidate"] == "C"
    assert with_sham["explanation_class"] == full["explanation_class"]
