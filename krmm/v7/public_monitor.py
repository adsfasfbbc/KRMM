"""Public-only stability monitor & transactional commit/rollback gate (Section 6.1 & 6.2)."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch

from krmm.v3.knowledge_registry import question_prompt
from krmm.v6.schemas import PublicUnit
from .state_ledger import LedgerEntry


def snapshot_public_monitor(backend, prompts: Sequence[str]) -> List[Dict[str, object]]:
    """Capture pre-write model predictions & confidence on PUBLIC_MONITOR prompts (no gold!)."""
    out = []
    for p in prompts:
        probs, mass = backend.probabilities(p)
        pred = "ABCD"[int(np.argmax(probs))]
        conf = float(max(probs))
        out.append({"prompt": p, "pred": pred, "conf": conf, "stable": bool(conf >= 0.70 and mass >= 0.25)})
    return out


def verify_public_transaction(
    backend,
    target_unit: PublicUnit,
    candidate: str,
    public_retain_prompts: Sequence[str],
    initial_retain_logits: Sequence[torch.Tensor],
    pre_monitor_snapshot: Sequence[Dict[str, object]],
    committed_entries: Sequence[LedgerEntry],
    pub_map: Dict[str, PublicUnit],
    min_retain_agreement: float = 0.90,
    monitor_mode: str = "relation_and_history",  # "generic_only" | "relation_and_history"
) -> Dict[str, object]:
    """Verify trial state strictly on PUBLIC observables (zero hidden audit!)."""
    # 1. Generic public retain agreement
    with torch.inference_mode():
        curr_logits = backend.retain_logits(list(public_retain_prompts))
    ret_agree = float(
        np.mean([int(a.argmax()) == int(b.argmax()) for a, b in zip(initial_retain_logits, curr_logits)])
    )
    generic_ok = ret_agree >= min_retain_agreement

    # 2. Target public paraphrase self-check (variant 3, unseen during variant 0/1 training)
    ku = target_unit.to_knowledge_unit("test")
    p3, _, _ = question_prompt(ku, variant=3)
    p3_probs, _ = backend.probabilities(p3)
    target_self_ok = ("ABCD"[int(np.argmax(p3_probs))] == candidate)

    # 3. Relation-specific public monitor (zero flip on pre-write stable public invariants)
    rel_flips = 0
    rel_checked = 0
    if monitor_mode == "relation_and_history":
        for rec in pre_monitor_snapshot:
            if not rec["stable"]:
                continue
            rel_checked += 1
            pr, _ = backend.probabilities(rec["prompt"])
            if "ABCD"[int(np.argmax(pr))] != rec["pred"]:
                rel_flips += 1

    # 4. Provisional history consistency check (previously committed candidates stay consistent)
    hist_flips = 0
    hist_checked = 0
    if monitor_mode == "relation_and_history":
        for e in committed_entries:
            if e.conflicted or e.unit_id not in pub_map:
                continue
            hist_checked += 1
            hku = pub_map[e.unit_id].to_knowledge_unit("test")
            hp, _, _ = question_prompt(hku, variant=3)
            hpr, _ = backend.probabilities(hp)
            if "ABCD"[int(np.argmax(hpr))] != e.candidate:
                hist_flips += 1

    rel_ok = (rel_flips == 0)
    hist_ok = (hist_flips == 0)
    passed = bool(generic_ok and target_self_ok and rel_ok and hist_ok)

    reasons = []
    if not generic_ok:
        reasons.append(f"generic_retain_drop({ret_agree:.3f})")
    if not target_self_ok:
        reasons.append("target_public_paraphrase_mismatch")
    if not rel_ok:
        reasons.append(f"public_relation_monitor_flip({rel_flips}/{rel_checked})")
    if not hist_ok:
        reasons.append(f"provisional_history_flip({hist_flips}/{hist_checked})")

    return {
        "passed": passed,
        "monitor_mode": monitor_mode,
        "public_retain_agreement": ret_agree,
        "target_self_ok": target_self_ok,
        "rel_checked": rel_checked,
        "rel_flips": rel_flips,
        "hist_checked": hist_checked,
        "hist_flips": hist_flips,
        "reasons": reasons,
    }
