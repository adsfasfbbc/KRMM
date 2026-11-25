"""Sequential cumulative standard-KL LoRA writer with public retain self-check & rollback (Section 5.1)."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch

from krmm.v3.damage import restore_weights, snapshot_weights, state_sha256
from krmm.v3.parameter_repair import repair as v3_repair
from krmm.v3.schemas import KnowledgeUnit


def sequential_write_and_public_verify(
    omni,
    unit: KnowledgeUnit,
    candidate: str,
    public_retain_prompts: Sequence[str],
    initial_retain_logits: Sequence[torch.Tensor],
    cfg: Dict[str, object],
    adapter_save_dir: Path,
    min_public_retain_agreement: float = 0.90,
) -> Dict[str, object]:
    """Apply ONE standard-KL LoRA repair step to the CURRENT cumulative model state M.

    - Saves pre-step state hash & snapshot (compact target layers only).
    - Runs fixed standard_kl LoRA (32 steps), saves adapter to adapter_save_dir.
    - Checks public retain agreement (strictly unlabeled self-check, NO hidden audit!).
    - If public self-check passes: merges adapter into M so subsequent steps operate on updated M!
    - If public self-check fails: unloads & restores pre-step state (rollback=True).
    """
    t0 = time.perf_counter()
    pre_state = snapshot_weights(omni, cfg)
    pre_hash = state_sha256(pre_state)

    adapter_save_dir.mkdir(parents=True, exist_ok=True)
    rep = v3_repair(
        omni,
        unit,
        candidate,
        public_retain_prompts,
        initial_retain_logits,
        cfg,
        save_dir=str(adapter_save_dir),
    )

    # Public unlabeled self-check on public retain prompts
    with torch.inference_mode():
        curr_logits = omni.retain_logits(list(public_retain_prompts))
    ret_agree = float(
        np.mean([int(a.argmax()) == int(b.argmax()) for a, b in zip(initial_retain_logits, curr_logits)])
    )
    rollback = bool(ret_agree < min_public_retain_agreement)

    if rollback:
        if hasattr(omni.model, "unload"):
            omni.model = omni.model.unload()
        restore_weights(omni, pre_state)
        post_hash = pre_hash
    else:
        # Merge into cumulative model weights M so next step operates on updated M!
        if hasattr(omni.model, "merge_and_unload"):
            omni.model = omni.model.merge_and_unload()
        for p in omni.model.parameters():
            p.requires_grad_(False)
        omni.model.eval()
        post_state = snapshot_weights(omni, cfg)
        post_hash = state_sha256(post_state)

    return {
        "pre_hash": pre_hash,
        "post_hash": post_hash,
        "adapter_path": str(adapter_save_dir.resolve()),
        "adapter_hash": rep.adapter_sha256,
        "train_steps": int(cfg.get("repair_steps", 32)),
        "public_selfcheck": {
            "public_retain_agreement": ret_agree,
            "threshold": min_public_retain_agreement,
            "passed": not rollback,
        },
        "rollback": rollback,
        "elapsed": time.perf_counter() - t0,
        "failure_reason": "public_retain_selfcheck_failed" if rollback else None,
    }
