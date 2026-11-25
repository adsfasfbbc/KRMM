"""Strictly isolated Public vs Evaluator-Only schemas for KRMM v0.6."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional

ActionType = Literal["ATTEMPT_REPAIR", "ABSTAIN", "REQUEST_EVIDENCE"]


@dataclass(frozen=True)
class PublicUnit:
    """Method-visible knowledge unit. Strictly excludes gold answers and pool categories."""
    unit_id: str
    entity_id: str
    domain: str
    relation: str
    options: Dict[str, str]
    source_group: str

    def to_public_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_knowledge_unit(self, split: str = "test"):
        from krmm.v3.schemas import KnowledgeUnit
        return KnowledgeUnit(
            unit_id=self.unit_id,
            entity_id=self.entity_id,
            relation_id=self.relation,
            modality="text",
            source_group=self.source_group,
            probe_template_ids=[0, 1, 2, 3],
            split=split,
            domain=self.domain,
            options=dict(self.options),
        )


@dataclass(frozen=True)
class EvaluatorRecord:
    """Evaluator-only hidden metadata. Never passed to non-Oracle method processes."""
    unit_id: str
    entity_id: str
    domain: str
    source_group: str
    gold_answer: Optional[str]  # None for fictional unknown or ambiguous/unverifiable
    pool_role: str  # "known_real" | "unknown_real" | "unknown_fictional" | "ambiguous"
    same_entity_controls: List[Dict[str, str]]
    ripple_probe: Optional[Dict[str, str]]
    aliases: List[str] = field(default_factory=list)
    provenance: str = "curated_v6_verifiable_fact"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class EpisodeStepRecord:
    """Structured per-step log required by Section 5.2 of KRMM v0.6 protocol."""
    episode_id: str
    model_id: str
    seed: int
    pool_sha256: str
    initial_state_hash: str
    step_id: int
    unit_id: str
    source_group: str
    rank: int
    pre_hash: str
    post_hash: str
    candidate: Optional[str]
    support_class: str
    score: float
    action: ActionType
    reason: str
    logical_queries: int
    physical_calls: int
    tokens: int
    train_steps: int
    adapter_path: Optional[str]
    adapter_hash: Optional[str]
    public_selfcheck: Dict[str, object]
    rollback: bool
    elapsed: float
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
