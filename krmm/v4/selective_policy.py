"""Selective repair policy decide_repair(...) with abstention and nonparametric fallback (Section 6.2)."""
from __future__ import annotations

from typing import Dict, Optional

from .schemas import EvidenceAssessment, assert_method_boundary


def decide_repair(
    damage_report: Dict[str, object],
    evidence_assessment: EvidenceAssessment,
    repair_budget: Dict[str, object],
    calibrated_policy: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Select one of ATTEMPT_PARAMETRIC_REPAIR / ABSTAIN / REQUEST_EXTERNAL_EVIDENCE / NONPARAMETRIC_COMPENSATION."""
    assert_method_boundary(damage_report)
    policy = calibrated_policy or {}
    min_rel = float(policy.get("min_reliability_for_parametric", 0.65))
    min_rel_nonparam = float(policy.get("min_reliability_for_nonparametric", 0.50))
    remaining_repairs = int(repair_budget.get("remaining_repairs", 1))
    allow_nonparam = bool(repair_budget.get("allow_nonparametric", True))

    dmg_status = str(damage_report.get("status", "unknown"))
    if dmg_status != "suspected":
        return {
            "action": "ABSTAIN",
            "reason": f"damage_status_{dmg_status}",
            "estimated_reliability": evidence_assessment.estimated_candidate_reliability,
        }

    if evidence_assessment.conflict_status:
        return {
            "action": "ABSTAIN",
            "reason": "conflicting_support",
            "estimated_reliability": 0.0,
        }

    if not evidence_assessment.identifiable or evidence_assessment.candidate_answer is None:
        return {
            "action": "REQUEST_EXTERNAL_EVIDENCE",
            "reason": "insufficient_support_unidentifiable",
            "estimated_reliability": 0.0,
        }

    # Correlated same-source duplication or single uncorroborated chain should NOT blindly trigger writeback
    rel = float(evidence_assessment.estimated_candidate_reliability)
    if evidence_assessment.evidence_category in {"redundant_same_source", "single_chain"} and rel < min_rel:
        return {
            "action": "REQUEST_EXTERNAL_EVIDENCE",
            "reason": f"low_reliability_{evidence_assessment.evidence_category}",
            "estimated_reliability": rel,
        }

    if rel >= min_rel:
        if remaining_repairs > 0:
            return {
                "action": "ATTEMPT_PARAMETRIC_REPAIR",
                "reason": f"verified_structure_{evidence_assessment.evidence_category}",
                "estimated_reliability": rel,
            }
        if allow_nonparam:
            return {
                "action": "NONPARAMETRIC_COMPENSATION",
                "reason": "parametric_budget_exhausted",
                "estimated_reliability": rel,
            }
        return {"action": "ABSTAIN", "reason": "repair_budget_exhausted", "estimated_reliability": rel}

    if allow_nonparam and rel >= min_rel_nonparam:
        return {
            "action": "NONPARAMETRIC_COMPENSATION",
            "reason": "moderate_reliability_nonparametric_fallback",
            "estimated_reliability": rel,
        }

    return {
        "action": "ABSTAIN",
        "reason": "below_reliability_threshold",
        "estimated_reliability": rel,
    }
