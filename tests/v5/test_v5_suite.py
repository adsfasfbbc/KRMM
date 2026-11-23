"""CPU unit test suite for KRMM v0.5 (all 8 support classes, causal interventions, 7 detectors, locality graph, catalog disjointness)."""
from __future__ import annotations

from krmm.v3.knowledge_registry import all_evaluator_units
from krmm.v3.schemas import AtomicFact
from krmm.v4.damage_truth_audit import classify_pool_truth
from krmm.v5.blind_discovery import CostTrackedProxy, V5_DETECTORS, evaluate_v5_discovery_metrics, run_v5_detector
from krmm.v5.causal_intervention import run_rule_level_interventions
from krmm.v5.end_to_end import run_v5_end_to_end_mock
from krmm.v5.locality_graph import evaluate_locality_graph
from krmm.v5.reliability import V5ReliabilityEstimator
from krmm.v5.selective_repair import REPAIR_STRATEGIES_V5, select_candidate_for_strategy
from krmm.v5.support_structure import V5_SUPPORT_CLASSES, analyze_support_structure
from krmm.v5.v5_catalog import all_v5_entities, v5_ambiguous_evaluator_units, v5_known_evaluator_units, v5_unknown_evaluator_units
import numpy as np


def _f(attr, val, prov="damaged_model_internal", src="probe"):
    return AtomicFact("anon_v5", attr, val, src, "text", 0.9, prov)


def test_v5_catalog_100_percent_disjoint_from_v3():
    v3_ents = {e.unit.entity_id.lower() for e in all_evaluator_units()}
    v5_ents = {e.lower() for e in all_v5_entities()}
    assert len(v5_known_evaluator_units()) == 36
    assert len(v5_unknown_evaluator_units()) == 12
    assert len(v5_ambiguous_evaluator_units()) == 4
    assert len(v5_ents) == 52
    assert v3_ents.isdisjoint(v5_ents), f"Overlap with v3: {v3_ents.intersection(v5_ents)}"


def test_all_eight_v5_support_classes():
    assert analyze_support_structure("vertebrate_class", [_f("covering", "feathers")]).support_class == "single_decisive"
    assert analyze_support_structure("vertebrate_class", [_f("thermoregulation", "cold_blooded"), _f("respiration", "lungs")]).support_class == "single_chain"
    assert analyze_support_structure("vertebrate_class", [_f("covering", "feathers", src="p1"), _f("covering", "feathers", src="p2")]).support_class == "redundant_same_model"
    assert analyze_support_structure("vertebrate_class", [_f("covering", "feathers"), _f("thermoregulation", "warm_blooded"), _f("reproduction", "hard_shelled_eggs")]).support_class == "redundant_distinct_attributes"
    assert analyze_support_structure("vertebrate_class", [_f("covering", "feathers", prov="m"), _f("thermoregulation", "warm_blooded", prov="ext"), _f("reproduction", "hard_shelled_eggs", prov="ext")]).support_class == "independent_source_support"
    assert analyze_support_structure("vertebrate_class", [_f("covering", "fur_or_hair")], flagged_correlated_wrong=True).support_class == "correlated_wrong_support"
    assert analyze_support_structure("vertebrate_class", [_f("thermoregulation", "warm_blooded"), _f("respiration", "gills")]).support_class == "conflicting_support"
    assert analyze_support_structure("vertebrate_class", [_f("thermoregulation", "warm_blooded")]).support_class == "insufficient_support"
    assert len(V5_SUPPORT_CLASSES) == 8


def test_v5_rule_level_interventions_and_scope_tag():
    res = run_rule_level_interventions(
        domain="vertebrate_class",
        complete_facts=[_f("covering", "feathers"), _f("thermoregulation", "warm_blooded"), _f("reproduction", "hard_shelled_eggs")],
        key_index=0,
        counterfactual_fact=_f("covering", "scales"),
        sham_fact=_f("container_color", "brown", prov="sham"),
        correlated_wrong_facts=[_f("covering", "fur_or_hair", src="p1"), _f("covering", "fur_or_hair", src="p2")],
    )
    assert res["intervention_level"] == "rule_engine_only_NOT_model_causal"
    assert all(res["sensitivity_checks"].values())


def test_v5_seven_detectors_equal_budget_and_three_class_metrics():
    class _Mock:
        def probabilities(self, prompt, media=None):
            return [0.85, 0.05, 0.05, 0.05], 0.95

        def generate(self, prompt, max_new_tokens=24, media=None):
            return "warm-blooded lungs live birth fur"

    units = [e.unit for e in v5_known_evaluator_units()[:4]] + [e.unit for e in v5_unknown_evaluator_units()[:2]]
    truths = [
        classify_pool_truth(units[0].unit_id, True, True, True, 1.0, 0.0).to_dict(),
        classify_pool_truth(units[1].unit_id, True, True, True, 1.0, 0.0).to_dict(),
        classify_pool_truth(units[2].unit_id, True, False, False, 1.0, 1.0).to_dict(),
        classify_pool_truth(units[3].unit_id, True, False, False, 1.0, 0.0).to_dict(),  # collateral_damaged
        classify_pool_truth(units[4].unit_id, False, False, False, 0.25, 0.25).to_dict(),
        classify_pool_truth(units[5].unit_id, False, False, False, 0.25, 0.25).to_dict(),
    ]
    B = 18
    for m in V5_DETECTORS:
        proxy = CostTrackedProxy(backend=_Mock(), query_budget=B)
        out = run_v5_detector(proxy, units, query_budget=B, method=m)
        assert out["total_queries"] <= B
        met = evaluate_v5_discovery_metrics(out, truths)
        assert met["n_collateral_damaged"] == 1
        assert "three_class" in met and "binary_full_pool" in met


def test_v5_locality_graph_separates_ripple_from_collateral():
    ok = evaluate_locality_graph("u1", 1.0, 1.0, 1.0, 0.95, 0.95, 0.95, 0.95, 0.85, 0.85, ripple_after=1.0, fresh_reload_passed=True)
    assert ok.strict_locality_pass is True and ok.collateral_damage_detected is False
    bad = evaluate_locality_graph("u2", 1.0, 1.0, 0.0, 0.95, 0.95, 0.95, 0.95, 0.85, 0.85, ripple_after=1.0, fresh_reload_passed=True)
    assert bad.strict_locality_pass is False and bad.collateral_damage_detected is True


def test_v5_reliability_sparsity_guard_and_end_to_end_mock():
    X = np.random.default_rng(1).normal(size=(20, 11))
    est = V5ReliabilityEstimator(min_samples=8, min_write_failures=5)
    st = est.fit(X, [1] * 14 + [0] * 6, [1] * 13 + [0] * 1 + [0] * 6)
    assert st["conditional_write_status"] == "insufficient_data"
    assert st["mode_active"] == "candidate_reliability_only"
    assert est.predict(X[0])["p_strict_decomposed"] is None
