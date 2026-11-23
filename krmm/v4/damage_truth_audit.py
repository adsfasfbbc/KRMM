"""Evaluator-only stratified damage assignment & post-injection behavioral truth audit (Section 4.2, 4.3)."""
from __future__ import annotations

import hashlib
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple

from krmm.v3.schemas import EvaluatorUnit
from .schemas import DamageTruthRecord


def stratified_damage_split(
    mastered_evs: Sequence[EvaluatorUnit],
    damage_fraction: float = 0.5,
    seed: int = 107,
    stratify_by_gold: bool = True,
) -> Tuple[List[EvaluatorUnit], List[EvaluatorUnit], Dict[str, object]]:
    """Domain- and class-stratified random target selection with randomized injection order and SHA256 manifest."""
    if not (0.0 < damage_fraction < 1.0):
        raise ValueError(f"damage_fraction must be in (0, 1), got {damage_fraction}")
    rng = random.Random(int(seed))
    strata: Dict[str, List[EvaluatorUnit]] = {}
    for ev in sorted(mastered_evs, key=lambda e: e.unit.unit_id):
        key = f"{ev.unit.domain}::{ev.gold_answer}" if stratify_by_gold else ev.unit.domain
        strata.setdefault(key, []).append(ev)

    damaged: List[EvaluatorUnit] = []
    healthy: List[EvaluatorUnit] = []
    for skey in sorted(strata.keys()):
        items = list(strata[skey])
        rng.shuffle(items)
        n_s = len(items)
        n_d = int(round(n_s * damage_fraction))
        if n_s >= 2:
            n_d = max(1, min(n_s - 1, n_d))
        damaged.extend(items[:n_d])
        healthy.extend(items[n_d:])

    rng.shuffle(damaged)
    rng.shuffle(healthy)

    d_ids = [e.unit.unit_id for e in damaged]
    h_ids = [e.unit.unit_id for e in healthy]
    if set(d_ids).intersection(h_ids):
        raise AssertionError("Overlap between damaged and healthy splits")
    all_ents = [e.unit.entity_id for e in damaged] + [e.unit.entity_id for e in healthy]
    if len(set(all_ents)) != len(all_ents):
        raise AssertionError("Duplicate entity across damaged/healthy splits")

    per_domain: Dict[str, Dict[str, int]] = {}
    for e in damaged:
        per_domain.setdefault(e.unit.domain, {"damaged": 0, "healthy": 0})["damaged"] += 1
    for e in healthy:
        per_domain.setdefault(e.unit.domain, {"damaged": 0, "healthy": 0})["healthy"] += 1

    manifest = {
        "seed": int(seed),
        "damage_fraction": float(damage_fraction),
        "stratify_by_gold": bool(stratify_by_gold),
        "per_domain_counts": per_domain,
        "damaged_injection_order": d_ids,
        "healthy_unit_ids": sorted(h_ids),
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return damaged, healthy, manifest


def classify_pool_truth(
    unit_id: str,
    clean_mastered: bool,
    planned_damaged: bool,
    injection_completed: bool,
    clean_accuracy: float,
    post_accuracy: float,
    ambiguous_or_unverifiable: bool = False,
    locality_pass: Optional[bool] = None,
    degraded_threshold: float = 0.5,
) -> DamageTruthRecord:
    """Assign each unit in the mixed pool to one of the 5 v0.4 behavioral ground-truth categories."""
    behaviorally_degraded = bool(clean_mastered and float(post_accuracy) <= float(degraded_threshold))
    if ambiguous_or_unverifiable:
        pool_cat = "ambiguous_or_unverifiable"
    elif not clean_mastered:
        pool_cat = "originally_unknown"
    elif planned_damaged and behaviorally_degraded:
        pool_cat = "damaged_mastered"
    elif (not planned_damaged) and behaviorally_degraded:
        pool_cat = "collateral_damaged"
    else:
        pool_cat = "healthy_mastered"

    strict_local = bool(
        pool_cat == "damaged_mastered" and injection_completed and (locality_pass is True)
    )
    return DamageTruthRecord(
        unit_id=unit_id,
        pool_category=pool_cat,
        planned_damaged=bool(planned_damaged),
        injection_completed=bool(injection_completed),
        clean_accuracy=float(clean_accuracy),
        post_accuracy=float(post_accuracy),
        behaviorally_degraded=behaviorally_degraded,
        locality_pass=locality_pass,
        strict_local_damage=strict_local,
        collateral_damage=(pool_cat == "collateral_damaged"),
    )
