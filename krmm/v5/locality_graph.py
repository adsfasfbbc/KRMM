"""G2-C Same-entity relation locality graph distinguishing expected ripple vs collateral damage (Section 7)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class LocalityGraphReport:
    unit_id: str
    target_restored: bool
    same_entity_relation_preservation: float
    neighbor_relation_preservation: float
    unrelated_knowledge_preservation: float
    expected_ripple_consistency: float
    general_capability_retention: float
    fresh_process_reload_passed: bool
    strict_locality_pass: bool
    collateral_damage_detected: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def evaluate_locality_graph(
    unit_id: str,
    target_acc: float,
    same_entity_before: float,
    same_entity_after: float,
    neighbor_before: float,
    neighbor_after: float,
    unrelated_before: float,
    unrelated_after: float,
    general_before: float,
    general_after: float,
    ripple_after: float,
    fresh_reload_passed: bool,
    max_drop: float = 0.05,
) -> LocalityGraphReport:
    """Evaluate all 5 G2-C locality metrics; ripple is tracked separately from collateral damage."""
    same_pres = 1.0 - max(0.0, same_entity_before - same_entity_after)
    neigh_pres = 1.0 - max(0.0, neighbor_before - neighbor_after)
    unrel_pres = 1.0 - max(0.0, unrelated_before - unrelated_after)
    gen_ret = 1.0 - max(0.0, general_before - general_after)

    same_ok = (same_entity_before - same_entity_after) <= 0.0
    neigh_ok = (neighbor_before - neighbor_after) <= max_drop
    unrel_ok = (unrelated_before - unrelated_after) <= max_drop
    gen_ok = (general_before - general_after) <= max_drop
    target_ok = float(target_acc) >= 0.99
    collateral = not (same_ok and neigh_ok and unrel_ok and gen_ok)
    strict_ok = bool(target_ok and (not collateral) and fresh_reload_passed)

    return LocalityGraphReport(
        unit_id=unit_id,
        target_restored=target_ok,
        same_entity_relation_preservation=float(same_pres),
        neighbor_relation_preservation=float(neigh_pres),
        unrelated_knowledge_preservation=float(unrel_pres),
        expected_ripple_consistency=float(ripple_after),
        general_capability_retention=float(gen_ret),
        fresh_process_reload_passed=bool(fresh_reload_passed),
        strict_locality_pass=strict_ok,
        collateral_damage_detected=collateral,
    )
