"""Budget-constrained repair decision policy (M4.5)."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple
from .schemas import CandidateReport, RepairabilityReport


def select_for_budget(
    items: Sequence[Tuple[str, float, CandidateReport, RepairabilityReport]],
    budget_fraction: float,
) -> Dict[str, str]:
    """Select up to round(budget_fraction * N) eligible items ranked by priority score."""
    n = len(items)
    cap = max(0, int(round(n * budget_fraction)))
    ranked = sorted(items, key=lambda x: x[1], reverse=True)
    decisions: Dict[str, str] = {}
    selected = 0
    for uid, _score, cand, rep in ranked:
        if cand.candidate_answer is None or cand.conflict:
            decisions[uid] = "abstain"
        elif rep.recommended_action == "nonparametric_recovery":
            decisions[uid] = "nonparametric_recovery"
        elif selected < cap and rep.recommended_action == "attempt_repair":
            decisions[uid] = "attempt_repair"
            selected += 1
        else:
            decisions[uid] = "abstain"
    return decisions
