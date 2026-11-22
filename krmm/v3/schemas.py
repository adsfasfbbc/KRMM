"""Unified public and evaluator-only data schemas for KRMM v0.3."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

VALID_MODALITIES = {"text", "image", "audio", "mixed"}
VALID_SPLITS = {"dev", "calibration", "test", "unknown_control", "anchor"}
VALID_DAMAGE_STATUS = {"suspected", "no_evidence_of_damage", "unknown"}
VALID_ACTIONS = {"attempt_repair", "abstain", "request_evidence", "nonparametric_recovery"}
FORBIDDEN_PUBLIC_KEYS = {
    "gold",
    "gold_answer",
    "answer",
    "damage_target",
    "wrong_target",
    "hidden_questions",
    "clean_weights",
    "damage_delta",
}


@dataclass(frozen=True)
class KnowledgeUnit:
    """Public knowledge unit exposed to algorithm/repair modules. Never contains ground truth."""

    unit_id: str
    entity_id: str
    relation_id: str
    modality: str
    source_group: str
    probe_template_ids: List[int]
    split: str
    domain: str
    options: Dict[str, str]

    def __post_init__(self) -> None:
        if self.modality not in VALID_MODALITIES:
            raise ValueError(f"Invalid modality: {self.modality}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Invalid split: {self.split}")
        if set(self.options.keys()) != {"A", "B", "C", "D"}:
            raise ValueError(f"Options must have exact keys A/B/C/D, got {set(self.options.keys())}")
        for key in FORBIDDEN_PUBLIC_KEYS:
            if hasattr(self, key):
                raise AssertionError(f"Forbidden key {key} leaked onto KnowledgeUnit")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_public_dict(cls, data: Dict[str, Any]) -> "KnowledgeUnit":
        leaked = FORBIDDEN_PUBLIC_KEYS.intersection(data.keys())
        if leaked:
            raise ValueError(f"Public KnowledgeUnit dict contains forbidden evaluator keys: {sorted(leaked)}")
        return cls(**data)


@dataclass(frozen=True)
class EvaluatorUnit:
    """Evaluator-only record holding gold answer, hidden controls, and supporting truth."""

    unit: KnowledgeUnit
    gold_answer: str
    supporting_attributes: Dict[str, str]
    same_entity_controls: List[Dict[str, Any]] = field(default_factory=list)
    is_originally_unknown_control: bool = False

    def __post_init__(self) -> None:
        if self.gold_answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"Invalid gold_answer: {self.gold_answer}")


@dataclass
class DamageReport:
    """Output of M2 damage discovery."""

    unit_id: str
    anomaly_score: float
    suspected_paths: List[str]
    supporting_observations: List[Dict[str, Any]]
    conflicts: List[Dict[str, Any]]
    reference_mode: str
    uncertainty: float
    query_cost: int
    status: str

    def __post_init__(self) -> None:
        if self.status not in VALID_DAMAGE_STATUS:
            raise ValueError(f"Invalid damage status: {self.status}")
        if self.reference_mode not in {"reference_based", "reference_free"}:
            raise ValueError(f"Invalid reference_mode: {self.reference_mode}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AtomicFact:
    """Structured atomic observation extracted from a model or evidence source."""

    entity: str
    attribute: str
    value: str
    source: str
    modality: str
    confidence: float
    provenance_group: str
    raw_output: str = ""
    matched_rule: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateReport:
    """Output of M3 residual knowledge reconstruction."""

    unit_id: str
    candidate_answer: Optional[str]
    candidate_source_type: str
    atomic_facts: List[Dict[str, Any]]
    support_paths: List[Dict[str, Any]]
    ambiguity: float
    conflict: bool
    key_intervention_result: Dict[str, Any]
    sham_intervention_result: Dict[str, Any]
    abstain_reason: Optional[str]
    query_cost: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RepairabilityReport:
    """Output of M4 decomposed repairability assessment."""

    unit_id: str
    candidate_reliability: Optional[float]
    predicted_repair_success: Optional[float]
    estimated_write_risk: Optional[float]
    recommended_action: str
    calibration_status: str
    limitations: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.recommended_action not in VALID_ACTIONS:
            raise ValueError(f"Invalid recommended_action: {self.recommended_action}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RepairAttempt:
    """Output of M5 repair attempt."""

    unit_id: str
    pre_target_behavior: Dict[str, Any]
    post_target_behavior: Dict[str, Any]
    training_log: List[Dict[str, Any]]
    weight_update_info: Dict[str, Any]
    adapter_path: Optional[str]
    adapter_sha256: Optional[str]
    retention_check: Dict[str, Any]
    persistent: bool
    rolled_back: bool
    failure_reason: Optional[str] = None
    recovery_mode: str = "parametric_lora"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
