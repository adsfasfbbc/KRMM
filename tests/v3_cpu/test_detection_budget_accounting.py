"""Verify hard query-budget enforcement and per-unit query accounting in evaluate_detection_at_budget."""
from __future__ import annotations

from krmm.v3.discovery import detect, evaluate_detection_at_budget
from krmm.v3.knowledge_registry import public_units


class _DeterministicMockBackend:
    def probabilities(self, prompt: str, media=None):
        return [0.85, 0.05, 0.05, 0.05], 0.95

    def generate(self, prompt: str, max_new_tokens: int = 32, media=None):
        return "warm-blooded lungs live birth fur"


def test_equal_budget_hard_stop_and_accounting():
    units = public_units(split="dev")[:6]
    budget = 12  # 2 queries/unit cap
    res = evaluate_detection_at_budget(
        detector=lambda proxy, us, b: detect(proxy, us, probe_budget=b, mode="reference_free", constraint_probe_fraction=0.9),
        query_budget=budget,
        units=units,
        backend=_DeterministicMockBackend(),
    )
    assert res["total_queries"] <= budget
    assert res["query_budget"] == budget
    assert len(res["reports"]) == len(units)
    assert sum(res["per_unit_queries"].values()) == res["total_queries"]


def test_strict_stop_when_detector_tries_to_overspend():
    units = public_units(split="dev")[:4]

    def greedy_detector(proxy, us, b):
        for _ in range(b + 5):
            proxy.probabilities("test")
        return []

    res = evaluate_detection_at_budget(
        detector=greedy_detector,
        query_budget=5,
        units=units,
        backend=_DeterministicMockBackend(),
    )
    assert res["total_queries"] == 5
    assert res["budget_exhausted"] is True
