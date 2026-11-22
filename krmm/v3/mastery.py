"""Training-free clean mastery screening with phrasing and option-permutation checks (M0.3)."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np

from .knowledge_registry import question_prompt
from .schemas import EvaluatorUnit, KnowledgeUnit

SCREEN_VARIANTS: Tuple[int, ...] = (0, 1, 2, 3)
PERM_CHECKS: Tuple[Tuple[int, Tuple[str, ...]], ...] = (
    (0, ("D", "C", "B", "A")),
    (1, ("B", "D", "A", "C")),
)
HELDOUT_VARIANTS: Tuple[int, ...] = (4, 5)
HELDOUT_PERM: Tuple[int, Tuple[str, ...]] = (4, ("C", "A", "D", "B"))


def _argmax_letter(probs: Sequence[float]) -> str:
    return "ABCD"[int(np.argmax(probs))]


def evaluate_unit_target(
    omni,
    unit: KnowledgeUnit,
    gold_answer: str,
    variants: Sequence[int],
    permutations: Sequence[Tuple[int, Tuple[str, ...]]] = (),
) -> Dict[str, object]:
    """Evaluate a model on canonical and permuted-option prompts, mapping predictions back to canonical options."""
    records: List[Dict[str, object]] = []
    for v in variants:
        prompt, d2c, _ = question_prompt(unit, variant=v)
        probs, mass = omni.probabilities(prompt)
        disp_pred = _argmax_letter(probs)
        canon_pred = d2c[disp_pred]
        canon_probs = {d2c[k]: float(probs[i]) for i, k in enumerate("ABCD")}
        records.append(
            {
                "variant": v,
                "permutation": ["A", "B", "C", "D"],
                "displayed_pred": disp_pred,
                "canonical_pred": canon_pred,
                "correct": bool(canon_pred == gold_answer),
                "choice_mass": float(mass),
                "canonical_probs": canon_probs,
            }
        )
    for v, perm in permutations:
        prompt, d2c, _ = question_prompt(unit, variant=v, permutation=perm)
        probs, mass = omni.probabilities(prompt)
        disp_pred = _argmax_letter(probs)
        canon_pred = d2c[disp_pred]
        canon_probs = {d2c[k]: float(probs[i]) for i, k in enumerate("ABCD")}
        records.append(
            {
                "variant": v,
                "permutation": list(perm),
                "displayed_pred": disp_pred,
                "canonical_pred": canon_pred,
                "correct": bool(canon_pred == gold_answer),
                "choice_mass": float(mass),
                "canonical_probs": canon_probs,
            }
        )
    acc = float(np.mean([r["correct"] for r in records])) if records else 0.0
    return {"accuracy": acc, "checks": records}


def screen_clean_mastery(omni, ev_unit: EvaluatorUnit, min_choice_mass: float = 0.25) -> Dict[str, object]:
    """Screen whether the clean model stably masters the unit across 4 phrasings + 2 option permutations."""
    screen = evaluate_unit_target(
        omni,
        ev_unit.unit,
        ev_unit.gold_answer,
        variants=SCREEN_VARIANTS,
        permutations=PERM_CHECKS,
    )
    heldout = evaluate_unit_target(
        omni,
        ev_unit.unit,
        ev_unit.gold_answer,
        variants=HELDOUT_VARIANTS,
        permutations=(HELDOUT_PERM,),
    )
    min_mass = min((c["choice_mass"] for c in screen["checks"]), default=0.0)
    stably_mastered = bool(screen["accuracy"] == 1.0 and min_mass >= min_choice_mass)
    exclusion_reason = None
    if screen["accuracy"] < 1.0:
        exclusion_reason = "failed_screen_accuracy"
    elif min_mass < min_choice_mass:
        exclusion_reason = "low_choice_mass"
    return {
        "unit_id": ev_unit.unit.unit_id,
        "entity_id": ev_unit.unit.entity_id,
        "domain": ev_unit.unit.domain,
        "stably_mastered": stably_mastered,
        "exclusion_reason": exclusion_reason,
        "screen": screen,
        "heldout_clean_audit": heldout,
        "heldout_clean_mastered": bool(heldout["accuracy"] == 1.0),
    }


def summarize_mastery(records: List[Dict[str, object]]) -> Dict[str, object]:
    total = len(records)
    mastered = sum(1 for r in records if r["stably_mastered"])
    heldout_mastered = sum(1 for r in records if r["stably_mastered"] and r["heldout_clean_mastered"])
    exclusions: Dict[str, int] = {}
    for r in records:
        if not r["stably_mastered"]:
            reason = str(r.get("exclusion_reason") or "unknown")
            exclusions[reason] = exclusions.get(reason, 0) + 1
    return {
        "total_candidates": total,
        "stably_mastered": mastered,
        "clean_mastery_rate": mastered / max(1, total),
        "heldout_also_mastered": heldout_mastered,
        "heldout_stable_rate": heldout_mastered / max(1, total),
        "exclusion_counts": exclusions,
    }
