"""Selective repair strategies (Direct, Generative, Structured, Evidence-Gated, Oracle) for G2-B (Section 6.2)."""
from __future__ import annotations

from typing import Dict, Optional

from .support_structure import SupportStructureAssessment

REPAIR_STRATEGIES_V5 = ("direct", "generative", "structured", "evidence_gated", "oracle")


def select_candidate_for_strategy(
    strategy: str,
    direct_candidate: Optional[str],
    generative_candidate: Optional[str],
    support: SupportStructureAssessment,
    min_reliability: float = 0.65,
    oracle_gold: Optional[str] = None,
) -> Dict[str, object]:
    """Return candidate and attempt decision for each of the 5 G2-B strategies."""
    if strategy == "direct":
        return {"attempt": direct_candidate is not None, "candidate": direct_candidate, "reason": "direct_unfiltered"}
    if strategy == "generative":
        return {"attempt": generative_candidate is not None, "candidate": generative_candidate, "reason": "generative_unfiltered"}
    if strategy == "structured":
        ok = support.candidate_answer is not None and not support.conflict_status
        return {"attempt": ok, "candidate": support.candidate_answer if ok else None, "reason": support.support_class}
    if strategy == "evidence_gated":
        ok = (
            support.identifiable
            and not support.conflict_status
            and support.recommended_action == "ATTEMPT_PARAMETRIC_REPAIR"
            and support.estimated_reliability >= min_reliability
        )
        return {
            "attempt": ok,
            "candidate": support.candidate_answer if ok else None,
            "reason": f"evidence_gated::{support.support_class}::{support.recommended_action}",
        }
    if strategy == "oracle":
        return {"attempt": oracle_gold is not None, "candidate": oracle_gold, "reason": "oracle_upper_bound"}
    raise ValueError(f"Unsupported strategy: {strategy}")
