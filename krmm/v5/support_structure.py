"""Support structure taxonomy (8 v0.5 classes) and minimal support-set analysis (Section 5.1)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from krmm.v3.schemas import AtomicFact
from krmm.v4.minimal_support import enumerate_minimal_support_sets
from .correlated_evidence import assess_evidence_correlation

V5_SUPPORT_CLASSES = frozenset(
    {
        "single_decisive",
        "single_chain",
        "redundant_same_model",
        "redundant_distinct_attributes",
        "independent_source_support",
        "correlated_wrong_support",
        "conflicting_support",
        "insufficient_support",
    }
)

# Conservative empirical reliability priors (identifiable=True NEVER implies 1.0)
V5_STRUCTURE_PRIORS: Dict[str, float] = {
    "independent_source_support": 0.90,
    "redundant_distinct_attributes": 0.84,
    "single_decisive": 0.72,
    "redundant_same_model": 0.48,
    "single_chain": 0.38,
    "correlated_wrong_support": 0.25,
    "conflicting_support": 0.0,
    "insufficient_support": 0.0,
}


@dataclass
class SupportStructureAssessment:
    candidate_answer: Optional[str]
    compatible_candidates: List[str]
    support_class: str
    minimal_support_sets: List[Dict[str, object]]
    n_minimal_sets: int
    min_support_size: Optional[int]
    dependency_overlap: float
    shared_premises: List[str]
    provenance_groups: List[str]
    same_model_only: bool
    conflict_status: bool
    identifiable: bool
    estimated_reliability: float
    recommended_action: str
    rule_trace: List[Dict[str, object]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def analyze_support_structure(
    domain: str,
    facts: Sequence[AtomicFact],
    min_repair_reliability: float = 0.65,
    flagged_correlated_wrong: bool = False,
) -> SupportStructureAssessment:
    """Classify atomic facts into one of the 8 v0.5 support classes without accessing evaluator truth."""
    min_res = enumerate_minimal_support_sets(domain, facts)
    valid_facts: List[AtomicFact] = min_res["valid_facts"]
    corr = assess_evidence_correlation(
        min_res["minimal_support_sets"],
        valid_facts,
        conflict_status=bool(min_res["conflict_status"]),
        identifiable=bool(min_res["identifiable"]),
        flagged_correlated_wrong=flagged_correlated_wrong,
    )
    s_class = str(corr["support_class"])
    overlap = float(corr["dependency_overlap"])
    base_rel = float(V5_STRUCTURE_PRIORS.get(s_class, 0.0))
    if s_class in {"redundant_distinct_attributes", "independent_source_support"} and overlap > 0.0:
        est_rel = max(0.0, min(0.95, base_rel * (1.0 - 0.25 * overlap)))
    else:
        est_rel = max(0.0, min(0.95, base_rel))

    if min_res["conflict_status"]:
        rec_action = "ABSTAIN"
    elif not min_res["identifiable"]:
        rec_action = "REQUEST_EXTERNAL_EVIDENCE"
    elif s_class in {"correlated_wrong_support", "redundant_same_model", "single_chain"} and est_rel < min_repair_reliability:
        rec_action = "REQUEST_EXTERNAL_EVIDENCE"
    elif est_rel >= min_repair_reliability:
        rec_action = "ATTEMPT_PARAMETRIC_REPAIR"
    else:
        rec_action = "ABSTAIN"

    return SupportStructureAssessment(
        candidate_answer=min_res["candidate"],
        compatible_candidates=list(min_res["compatible_candidates"]),
        support_class=s_class,
        minimal_support_sets=list(min_res["minimal_support_sets"]),
        n_minimal_sets=len(min_res["minimal_support_sets"]),
        min_support_size=min_res["min_support_size"],
        dependency_overlap=overlap,
        shared_premises=list(corr["shared_premises"]),
        provenance_groups=list(corr["provenance_groups"]),
        same_model_only=bool(corr["same_model_only"]),
        conflict_status=bool(min_res["conflict_status"]),
        identifiable=bool(min_res["identifiable"]),
        estimated_reliability=float(est_rel),
        recommended_action=rec_action,
        rule_trace=list(min_res["full_eval"]["rule_trace"]),
        warnings=list(corr["warnings"]),
    )
