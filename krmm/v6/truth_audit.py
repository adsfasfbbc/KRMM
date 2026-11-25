"""Paired pre/post held-out behavioral truth classification & blind pool shuffling (fixes v0.5 P0 bugs A & B)."""
from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .schemas import PublicUnit


@dataclass(frozen=True)
class PairedTruthRecord:
    unit_id: str
    entity_id: str
    domain: str
    source_group: str
    pool_role: str
    planned_damaged: bool
    injection_completed: bool
    clean_screen_passed: bool
    clean_screen_accuracy: float
    clean_heldout_accuracy: Optional[float]
    post_heldout_accuracy: Optional[float]
    heldout_delta: Optional[float]
    truth_category: str
    # 6-way: original_mastered_healthy | effective_planned_damage | collateral_damaged |
    #        previously_unmastered | unverifiable | insufficient_observation
    is_degraded_target: bool  # True iff effective_planned_damage or collateral_damaged

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def classify_paired_truth(
    unit_id: str,
    entity_id: str,
    domain: str,
    source_group: str,
    pool_role: str,
    planned_damaged: bool,
    injection_completed: bool,
    clean_screen_passed: bool,
    clean_screen_accuracy: float,
    clean_heldout_accuracy: Optional[float],
    post_heldout_accuracy: Optional[float],
) -> PairedTruthRecord:
    """Compute true paired pre/post held-out truth without v0.5 clean_accuracy=0 bug."""
    if pool_role in ("ambiguous", "unknown_fictional"):
        return PairedTruthRecord(
            unit_id=unit_id,
            entity_id=entity_id,
            domain=domain,
            source_group=source_group,
            pool_role=pool_role,
            planned_damaged=planned_damaged,
            injection_completed=injection_completed,
            clean_screen_passed=False,
            clean_screen_accuracy=float(clean_screen_accuracy),
            clean_heldout_accuracy=clean_heldout_accuracy,
            post_heldout_accuracy=post_heldout_accuracy,
            heldout_delta=None,
            truth_category="unverifiable",
            is_degraded_target=False,
        )

    if clean_heldout_accuracy is None or post_heldout_accuracy is None:
        return PairedTruthRecord(
            unit_id=unit_id,
            entity_id=entity_id,
            domain=domain,
            source_group=source_group,
            pool_role=pool_role,
            planned_damaged=planned_damaged,
            injection_completed=injection_completed,
            clean_screen_passed=bool(clean_screen_passed),
            clean_screen_accuracy=float(clean_screen_accuracy),
            clean_heldout_accuracy=clean_heldout_accuracy,
            post_heldout_accuracy=post_heldout_accuracy,
            heldout_delta=None,
            truth_category="insufficient_observation",
            is_degraded_target=False,
        )

    clean_h = float(clean_heldout_accuracy)
    post_h = float(post_heldout_accuracy)
    delta = clean_h - post_h

    # Must pass both screen AND clean held-out == 1.0 to count as originally mastered
    originally_mastered = bool(clean_screen_passed and clean_h >= 0.999)
    if not originally_mastered:
        cat = "previously_unmastered"
        degraded = False
    else:
        # Pre-registered paired degradation rule: post <= 0.5 AND delta >= 0.5
        is_deg = bool(post_h <= 0.5001 and delta >= 0.4999)
        if is_deg and planned_damaged and injection_completed:
            cat = "effective_planned_damage"
            degraded = True
        elif is_deg:
            cat = "collateral_damaged"
            degraded = True
        else:
            cat = "original_mastered_healthy"
            degraded = False

    return PairedTruthRecord(
        unit_id=unit_id,
        entity_id=entity_id,
        domain=domain,
        source_group=source_group,
        pool_role=pool_role,
        planned_damaged=planned_damaged,
        injection_completed=injection_completed,
        clean_screen_passed=bool(clean_screen_passed),
        clean_screen_accuracy=float(clean_screen_accuracy),
        clean_heldout_accuracy=clean_h,
        post_heldout_accuracy=post_h,
        heldout_delta=delta,
        truth_category=cat,
        is_degraded_target=degraded,
    )


def shuffle_public_pool(units: Sequence[PublicUnit], seed: int) -> Tuple[List[PublicUnit], str]:
    """Deterministically shuffle public units so list position never leaks pool_role."""
    items = list(units)
    rng = random.Random(int(seed))
    rng.shuffle(items)
    h = hashlib.sha256()
    for u in items:
        h.update(f"{u.unit_id}|{u.entity_id}|{u.domain}\n".encode("utf-8"))
    return items, h.hexdigest()


def deterministic_tie_key(unit_id: str, seed: int) -> str:
    """Tie-breaker hash independent of labels or original category ordering."""
    return hashlib.sha256(f"{seed}:{unit_id}".encode("utf-8")).hexdigest()
