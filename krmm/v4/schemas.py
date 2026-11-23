"""Typed schemas and strict information-boundary dataclasses for KRMM v0.4."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

EVIDENCE_CATEGORIES_V4 = frozenset(
    {
        "single_decisive",
        "single_chain",
        "redundant_same_source",
        "redundant_distinct_paths",
        "independent_source_support",
        "conflicting_support",
        "insufficient_support",
    }
)

POOL_CATEGORIES_V4 = frozenset(
    {
        "damaged_mastered",
        "healthy_mastered",
        "originally_unknown",
        "collateral_damaged",
        "ambiguous_or_unverifiable",
    }
)

DECISION_ACTIONS_V4 = frozenset(
    {
        "ATTEMPT_PARAMETRIC_REPAIR",
        "ABSTAIN",
        "REQUEST_EXTERNAL_EVIDENCE",
        "NONPARAMETRIC_COMPENSATION",
    }
)

STUDY_CONDITIONS_V4 = frozenset({"internal_only", "evidence_augmented"})

FORBIDDEN_METHOD_KEYS = frozenset(
    {
        "gold_answer",
        "gold_evaluator_only",
        "wrong_target",
        "clean_state",
        "damage_delta",
        "injected_target_list",
        "hidden_audit_labels",
        "pool_truth_category",
    }
)


def assert_method_boundary(mapping: Optional[Dict[str, object]]) -> None:
    if not isinstance(mapping, dict):
        return
    leaked = FORBIDDEN_METHOD_KEYS.intersection(mapping.keys())
    if leaked:
        raise ValueError(f"Information boundary violation: forbidden evaluator keys {sorted(leaked)}")


@dataclass
class EvidenceAssessment:
    compatible_candidates: List[str]
    minimal_support_sets: List[Dict[str, object]]
    number_of_support_paths: int
    dependency_overlap: float
    provenance_groups: List[str]
    conflict_status: bool
    identifiable: bool
    estimated_candidate_reliability: float
    recommended_action: str
    evidence_category: str = "insufficient_support"
    candidate_answer: Optional[str] = None
    min_support_size: Optional[int] = None
    shared_premises: List[str] = field(default_factory=list)
    rule_trace: List[Dict[str, object]] = field(default_factory=list)
    study_condition: str = "internal_only"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class DamageTruthRecord:
    unit_id: str
    pool_category: str
    planned_damaged: bool
    injection_completed: bool
    clean_accuracy: float
    post_accuracy: float
    behaviorally_degraded: bool
    locality_pass: Optional[bool] = None
    strict_local_damage: bool = False
    collateral_damage: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class SideEffectReport:
    unit_id: str
    target_restored: bool
    same_entity_preserved: bool
    neighbor_knowledge_preserved: bool
    general_capability_preserved: bool
    fresh_process_reload_passed: bool
    entailed_updates_passed: bool = True
    strict_parametric_recovery: bool = False
    details: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
