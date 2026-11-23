"""Verify stratified M2 damage target selection: reproducibility, seed variation, balance, disjointness."""
from __future__ import annotations

from krmm.v3.knowledge_registry import all_evaluator_units
from krmm.v3.pipeline_p1_p3 import select_stratified_damage_targets


def test_stratified_randomization_reproducible_and_seed_sensitive():
    evs = all_evaluator_units(split="dev")
    d1, h1, m1 = select_stratified_damage_targets(evs, damage_fraction=0.5, seed=107)
    d2, h2, m2 = select_stratified_damage_targets(evs, damage_fraction=0.5, seed=107)
    d3, h3, m3 = select_stratified_damage_targets(evs, damage_fraction=0.5, seed=2026)

    # Identical seed -> exact reproduction including injection order & manifest hash
    assert [e.unit.unit_id for e in d1] == [e.unit.unit_id for e in d2]
    assert m1["manifest_sha256"] == m2["manifest_sha256"]

    # Different seed -> different assignment/order
    assert [e.unit.unit_id for e in d1] != [e.unit.unit_id for e in d3]
    assert m1["manifest_sha256"] != m3["manifest_sha256"]

    # Strict domain balance (dev has 12 per domain -> 6 damaged, 6 healthy each)
    for dom, cnts in m1["per_domain_counts"].items():
        assert cnts["damaged"] == 6, f"Domain {dom} damaged={cnts['damaged']}"
        assert cnts["healthy"] == 6, f"Domain {dom} healthy={cnts['healthy']}"

    # Strict disjointness and zero duplicate entities
    d_ids = {e.unit.unit_id for e in d1}
    h_ids = {e.unit.unit_id for e in h1}
    assert d_ids.isdisjoint(h_ids)
    assert len(d_ids) + len(h_ids) == len(evs)
    all_ents = [e.unit.entity_id for e in d1] + [e.unit.entity_id for e in h1]
    assert len(set(all_ents)) == len(all_ents)
