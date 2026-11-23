"""Verify v0.4 domain- and class-stratified damage randomization and behavioral pool truth."""
from __future__ import annotations

from krmm.v3.knowledge_registry import all_evaluator_units
from krmm.v4.damage_truth_audit import classify_pool_truth, stratified_damage_split


def test_stratified_damage_split_and_pool_truth_categories():
    evs = all_evaluator_units(split="calibration")
    d1, h1, m1 = stratified_damage_split(evs, damage_fraction=0.5, seed=2026, stratify_by_gold=True)
    d2, h2, m2 = stratified_damage_split(evs, damage_fraction=0.5, seed=2026, stratify_by_gold=True)
    d3, h3, m3 = stratified_damage_split(evs, damage_fraction=0.5, seed=2027, stratify_by_gold=True)

    assert m1["manifest_sha256"] == m2["manifest_sha256"]
    assert m1["manifest_sha256"] != m3["manifest_sha256"]
    assert {e.unit.unit_id for e in d1}.isdisjoint({e.unit.unit_id for e in h1})

    # Verify all 5 pool categories
    assert classify_pool_truth("u1", True, True, True, 1.0, 0.0, locality_pass=True).pool_category == "damaged_mastered"
    assert classify_pool_truth("u2", True, False, False, 1.0, 1.0).pool_category == "healthy_mastered"
    assert classify_pool_truth("u3", False, False, False, 0.25, 0.25).pool_category == "originally_unknown"
    assert classify_pool_truth("u4", True, False, False, 1.0, 0.25).pool_category == "collateral_damaged"
    assert classify_pool_truth("u5", True, False, False, 1.0, 1.0, ambiguous_or_unverifiable=True).pool_category == "ambiguous_or_unverifiable"
