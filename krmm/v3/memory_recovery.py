"""Non-parametric functional compensation fallback (M5.4). Never counted as parametric repair."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np

from .knowledge_registry import question_prompt
from .schemas import CandidateReport, KnowledgeUnit, RepairAttempt


class ExternalMemoryStore:
    """Explicit external memory mapping scoped to verified unit_ids only."""

    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, object]] = {}

    def put(self, unit: KnowledgeUnit, cand: CandidateReport) -> None:
        if cand.candidate_answer is None:
            raise ValueError("Cannot store empty candidate in external memory")
        self._entries[unit.unit_id] = {
            "entity_id": unit.entity_id,
            "relation_id": unit.relation_id,
            "candidate_answer": cand.candidate_answer,
            "support_paths": list(cand.support_paths),
        }

    def get(self, unit_id: str) -> Optional[Dict[str, object]]:
        return self._entries.get(unit_id)


def compensate_nonparametric(
    omni,
    unit: KnowledgeUnit,
    cand: CandidateReport,
    store: ExternalMemoryStore,
) -> RepairAttempt:
    """Apply explicit evidence-assisted routing tagged strictly as functional_compensation."""
    if cand.candidate_answer is None or cand.conflict:
        return RepairAttempt(
            unit_id=unit.unit_id,
            pre_target_behavior={},
            post_target_behavior={},
            training_log=[],
            weight_update_info={"method": "none"},
            adapter_path=None,
            adapter_sha256=None,
            retention_check={"passed": True, "zero_weight_drift": True},
            persistent=False,
            rolled_back=False,
            failure_reason="no_valid_candidate",
            recovery_mode="functional_compensation",
        )

    store.put(unit, cand)
    entry = store.get(unit.unit_id)
    hint = (
        f"Verified external memory note for {unit.entity_id} ({unit.relation_id}): "
        f"consistent evidence supports option {entry['candidate_answer']}: {unit.options[entry['candidate_answer']]}."
    )
    post: Dict[str, object] = {}
    for v in (4, 5):
        p, d2c, _ = question_prompt(unit, variant=v)
        probs, mass = omni.probabilities(hint + "\n" + p)
        disp = "ABCD"[int(np.argmax(probs))]
        post[f"v{v}"] = {"displayed_pred": disp, "canonical_pred": d2c[disp], "choice_mass": float(mass)}

    return RepairAttempt(
        unit_id=unit.unit_id,
        pre_target_behavior={},
        post_target_behavior=post,
        training_log=[],
        weight_update_info={"method": "external_memory_hint", "parameter_delta_norm": 0.0},
        adapter_path=None,
        adapter_sha256=None,
        retention_check={"passed": True, "zero_weight_drift": True},
        persistent=False,
        rolled_back=False,
        failure_reason=None,
        recovery_mode="functional_compensation",
    )
