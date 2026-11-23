"""Verify all 5 v0.4 blind discovery comparators respect identical hard query budgets."""
from __future__ import annotations

from krmm.v3.discovery import BudgetedQueryProxy
from krmm.v3.knowledge_registry import public_units
from krmm.v4.blind_discovery import run_blind_detector


class _MockBackend:
    def probabilities(self, prompt: str, media=None):
        return [0.80, 0.10, 0.05, 0.05], 0.92

    def generate(self, prompt: str, max_new_tokens: int = 24, media=None):
        return "warm-blooded lungs live birth fur"


def test_all_five_comparators_stay_within_equal_budget():
    units = public_units(split="dev")[:5]
    budget = 10
    for method in ("direct_confidence", "self_consistency", "direct_paraphrases", "constraint_conflict_only", "full_m2"):
        proxy = BudgetedQueryProxy(backend=_MockBackend(), query_budget=budget)
        res = run_blind_detector(proxy, units, query_budget=budget, method=method)
        assert res["total_queries"] <= budget, f"{method} exceeded budget: {res['total_queries']} > {budget}"
        assert len(res["reports"]) == len(units)
