"""EvidenceGraph analyzer producing structured EvidenceAssessment (P2, Section 5.1)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from krmm.v3.schemas import AtomicFact
from .evidence_dependence import analyze_evidence_dependence
from .minimal_support import enumerate_minimal_support_sets
from .schemas import EvidenceAssessment, STUDY_CONDITIONS_V4

# Conservative empirical priors (NEVER 1.0 just because identifiable=True!)
DEFAULT_STRUCTURE_RELIABILITY_PRIORS: Dict[str, float] = {
    "independent_source_support": 0.90,
    "redundant_distinct_paths": 0.85,
    "single_decisive": 0.72,
    "redundant_same_source": 0.50,
    "single_chain": 0.40,
    "conflicting_support": 0.0,
    "insufficient_support": 0.0,
}


class EvidenceGraphAnalyzer:
    """Evaluates extracted atomic facts and produces an EvidenceAssessment with conservative reliability."""

    def __init__(
        self,
        domain: str,
        study_condition: str = "internal_only",
        reliability_priors: Optional[Dict[str, float]] = None,
    ) -> None:
        if study_condition not in STUDY_CONDITIONS_V4:
            raise ValueError(f"Unsupported study_condition: {study_condition}")
        self.domain = domain
        self.study_condition = study_condition
        self.priors = dict(DEFAULT_STRUCTURE_RELIABILITY_PRIORS)
        if reliability_priors:
            self.priors.update(reliability_priors)

    def assess(
        self,
        facts: Sequence[AtomicFact],
        min_repair_reliability: float = 0.65,
    ) -> EvidenceAssessment:
        min_res = enumerate_minimal_support_sets(self.domain, facts)
        valid_facts: List[AtomicFact] = min_res["valid_facts"]
        dep = analyze_evidence_dependence(
            min_res["minimal_support_sets"],
            valid_facts,
            conflict_status=bool(min_res["conflict_status"]),
            identifiable=bool(min_res["identifiable"]),
        )
        cat = str(dep["evidence_category"])
        base_rel = float(self.priors.get(cat, 0.0))

        # Penalize high premise overlap among redundant paths so shared-premise duplication is discounted
        overlap = float(dep["dependency_overlap"])
        if cat in {"redundant_distinct_paths", "independent_source_support"} and overlap > 0.0:
            est_rel = max(0.0, min(0.95, base_rel * (1.0 - 0.25 * overlap)))
        else:
            est_rel = max(0.0, min(0.95, base_rel))

        notes: List[str] = []
        if cat == "redundant_distinct_paths" and not dep["has_independent_sources"]:
            notes.append("Distinct semantic attributes from same model; NOT statistically independent sources.")
        elif cat == "redundant_same_source":
            notes.append("Repeated/correlated same-source support; discounted against spurious redundancy.")
        elif cat == "single_chain":
            notes.append("Uncorroborated single multi-premise chain; elevated error risk.")

        if min_res["conflict_status"]:
            rec_action = "abstain"
        elif not min_res["identifiable"]:
            rec_action = "request_evidence"
        elif est_rel >= min_repair_reliability:
            rec_action = "repair"
        else:
            rec_action = "request_evidence"

        return EvidenceAssessment(
            compatible_candidates=list(min_res["compatible_candidates"]),
            minimal_support_sets=list(min_res["minimal_support_sets"]),
            number_of_support_paths=len(min_res["minimal_support_sets"]),
            dependency_overlap=overlap,
            provenance_groups=list(dep["provenance_groups"]),
            conflict_status=bool(min_res["conflict_status"]),
            identifiable=bool(min_res["identifiable"]),
            estimated_candidate_reliability=float(est_rel),
            recommended_action=rec_action,
            evidence_category=cat,
            candidate_answer=min_res["candidate"],
            min_support_size=min_res["min_support_size"],
            shared_premises=list(dep["shared_premises"]),
            rule_trace=list(min_res["full_eval"]["rule_trace"]),
            study_condition=self.study_condition,
            notes=notes,
        )
