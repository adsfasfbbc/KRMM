"""Evaluator-side artificial damage injection, balanced wrong-target assignment, and state snapshots (M1)."""
from __future__ import annotations

import collections
import hashlib
import random
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from .knowledge_registry import question_prompt
from .schemas import EvaluatorUnit, KnowledgeUnit


def snapshot_weights(omni, cfg: Dict[str, object]) -> Dict[str, torch.Tensor]:
    """Capture CPU clones of trainable target layer weights for deterministic rollback/restore."""
    num_layers = omni.model.config.text_config.num_hidden_layers
    last_layers = int(cfg.get("last_layers", 4))
    target_modules = tuple(cfg.get("target_modules", ("q_proj", "v_proj", "down_proj")))
    suffixes = tuple(f"{m}.weight" for m in target_modules)
    layer_tags = tuple(f"layers.{j}." for j in range(num_layers - last_layers, num_layers))
    return {
        k: p.detach().cpu().clone()
        for k, p in omni.model.named_parameters()
        if any(lt in k for lt in layer_tags) and any(s in k for s in suffixes)
    }


def restore_weights(omni, state: Dict[str, torch.Tensor]) -> None:
    params = dict(omni.model.named_parameters())
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v)
    for p in omni.model.parameters():
        p.requires_grad_(False)
    omni.model.eval()


def state_sha256(state: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode("utf-8"))
        h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def balanced_wrong_assignments(
    ev_units: Sequence[EvaluatorUnit],
    seed: int,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    """Assign independent, random, balanced wrong targets with no deterministic inverse shortcut."""
    rng = random.Random(seed)
    counts: collections.Counter[str] = collections.Counter()
    assign: Dict[str, str] = {}
    ordered = list(ev_units)
    rng.shuffle(ordered)
    for ev in ordered:
        allowed = [c for c in "ABCD" if c != ev.gold_answer]
        low = min(counts[c] for c in allowed)
        choice = rng.choice([c for c in allowed if counts[c] == low])
        counts[choice] += 1
        assign[ev.unit.unit_id] = choice

    table = {
        w: collections.Counter(ev.gold_answer for ev in ev_units if assign[ev.unit.unit_id] == w)
        for w in "ABCD"
    }
    in_sample_inv = sum(max(v.values(), default=0) for v in table.values()) / max(1, len(ev_units))
    cyclic_inv = float(
        np.mean(["ABCD"[("ABCD".index(assign[ev.unit.unit_id]) - 1) % 4] == ev.gold_answer for ev in ev_units])
    )
    active_wrong = [w for w, v in table.items() if sum(v.values()) > 0]
    no_det_inv = bool(len(active_wrong) >= 2 and all(len(table[w]) >= 2 for w in active_wrong))
    audit = {
        "counts": dict(counts),
        "confusion_wrong_to_truth": {k: dict(v) for k, v in table.items()},
        "best_in_sample_inverse_accuracy": in_sample_inv,
        "cyclic_inverse_accuracy": cyclic_inv,
        "no_deterministic_inverse": no_det_inv,
    }
    return assign, audit


def attach_lora(omni, cfg: Dict[str, object]) -> List[torch.nn.Parameter]:
    num_layers = omni.model.config.text_config.num_hidden_layers
    last_layers = int(cfg.get("last_layers", 4))
    target_modules = list(cfg.get("target_modules", ["q_proj", "v_proj", "down_proj"]))
    lc = LoraConfig(
        r=int(cfg.get("rank", 8)),
        lora_alpha=int(cfg.get("alpha", 16)),
        target_modules=target_modules,
        layers_to_transform=list(range(num_layers - last_layers, num_layers)),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    omni.model = get_peft_model(omni.model, lc)
    for conf in omni.model.peft_config.values():
        conf.base_model_name_or_path = str(cfg["model_path"])
    return [p for p in omni.model.parameters() if p.requires_grad]


def inject_damage(
    omni,
    unit: KnowledgeUnit,
    gold_answer: str,
    wrong_target: str,
    retains: Sequence[str],
    teacher_logits: Sequence[torch.Tensor],
    cfg: Dict[str, object],
    damage_type: str = "D1_wrong_association",
    adapter_save_dir: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Inject localized D1 (wrong association) or D2 (response suppression) via LoRA and merge."""
    if damage_type not in {"D1_wrong_association", "D2_suppression"}:
        raise ValueError(f"Unsupported damage_type: {damage_type}")
    params = attach_lora(omni, cfg)
    lr = float(cfg.get("damage_lr", cfg.get("lr", 5e-4)))
    steps = int(cfg.get("damage_steps", 24))
    retain_weight = float(cfg.get("damage_retain_weight", cfg.get("retain_weight", 50.0)))
    l2_weight = float(cfg.get("damage_l2_weight", cfg.get("l2_weight", 1e-4)))
    retain_batch = int(cfg.get("retain_batch_size", len(retains)))
    use_perm = bool(cfg.get("damage_use_permuted_target", True))

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    choice_ids = torch.tensor(
        [omni.tokenizer.encode(c, add_special_tokens=False)[0] for c in "ABCD"],
        device="cuda",
    )
    logs: List[Dict[str, float]] = []
    perm_schedule = [("A", "B", "C", "D"), ("D", "C", "B", "A")] if use_perm else [("A", "B", "C", "D")]
    variants = (0, 1)

    for step in range(steps):
        omni.model.train()
        opt.zero_grad(set_to_none=True)
        var = variants[step % len(variants)]
        perm = perm_schedule[(step // len(variants)) % len(perm_schedule)]
        prompt, _, c2d = question_prompt(unit, variant=var, permutation=perm)

        if damage_type == "D1_wrong_association":
            disp_target = c2d[wrong_target]
            target_loss = omni.model(**omni.encode(prompt, disp_target)).loss
        else:
            disp_gold = c2d[gold_answer]
            ce = omni.model(**omni.encode(prompt, disp_gold)).loss
            # Unlikelihood pushes P(gold) -> 0 without assigning a specific false target
            target_loss = -torch.log1p(-torch.exp(-ce).clamp(max=1.0 - 1e-6))

        # Sample retention batch deterministically by rotating over full retention pool
        n_ret = len(retains)
        bsize = min(retain_batch, n_ret)
        idxs = [(step * bsize + k) % n_ret for k in range(bsize)]
        r_prompts = [retains[j] for j in idxs]
        logits = omni.model(**omni.encode_many(r_prompts)).logits[:, -1].float()
        t = torch.stack([teacher_logits[j] for j in idxs]).to(logits.device)
        kl = torch.nn.functional.kl_div(logits.log_softmax(-1), t.softmax(-1), reduction="batchmean")
        l2 = sum(p.float().square().sum() for p in params)

        loss = target_loss + retain_weight * kl + l2_weight * l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        logs.append(
            {
                "step": step,
                "target_loss": float(target_loss.detach()),
                "retain_kl": float(kl.detach()),
                "l2": float(l2.detach()),
            }
        )

    if adapter_save_dir is not None:
        omni.model.save_pretrained(adapter_save_dir, save_embedding_layers=False)
    omni.model = omni.model.merge_and_unload()
    for p in omni.model.parameters():
        p.requires_grad_(False)
    omni.model.eval()
    return logs
