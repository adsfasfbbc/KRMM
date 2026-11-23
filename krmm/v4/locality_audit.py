"""Strict 4-category independent locality & entailed-update side-effect auditor (Section 6.4)."""
from __future__ import annotations

from typing import Dict, Sequence

from .schemas import SideEffectReport


def audit_side_effects(
    unit_id: str,
    target_heldout_accuracy: float,
    same_entity_invariant_acc_before: float,
    same_entity_invariant_acc_after: float,
    neighbor_acc_before: float,
    neighbor_acc_after: float,
    general_acc_before: float,
    general_acc_after: float,
    fresh_process_reload_passed: bool,
    entailed_updates_passed: bool = True,
    max_allowed_drop: float = 0.05,
    min_target_acc: float = 0.99,
) -> SideEffectReport:
    """Evaluate all 5 strict persistence & locality gates; distinguishes invariant vs entailed updates."""
    target_ok = float(target_heldout_accuracy) >= float(min_target_acc)
    same_entity_ok = (float(same_entity_invariant_acc_before) - float(same_entity_invariant_acc_after)) <= 0.0
    neighbor_ok = (float(neighbor_acc_before) - float(neighbor_acc_after)) <= float(max_allowed_drop)
    general_ok = (float(general_acc_before) - float(general_acc_after)) <= float(max_allowed_drop)
    reload_ok = bool(fresh_process_reload_passed)
    strict_ok = bool(target_ok and same_entity_ok and neighbor_ok and general_ok and reload_ok and entailed_updates_passed)

    return SideEffectReport(
        unit_id=unit_id,
        target_restored=target_ok,
        same_entity_preserved=same_entity_ok,
        neighbor_knowledge_preserved=neighbor_ok,
        general_capability_preserved=general_ok,
        fresh_process_reload_passed=reload_ok,
        entailed_updates_passed=bool(entailed_updates_passed),
        strict_parametric_recovery=strict_ok,
        details={
            "target_heldout_accuracy": float(target_heldout_accuracy),
            "same_entity_drop": float(same_entity_invariant_acc_before - same_entity_invariant_acc_after),
            "neighbor_drop": float(neighbor_acc_before - neighbor_acc_after),
            "general_drop": float(general_acc_before - general_acc_after),
        },
    )
