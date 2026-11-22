"""Constrained parameter-level LoRA repairer (M5). Strictly isolated from evaluator truth."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from .knowledge_registry import question_prompt
from .schemas import KnowledgeUnit, RepairAttempt


def _adapter_sha256(adapter_dir: Path) -> str:
    h = hashlib.sha256()
    for fname in sorted(p.name for p in adapter_dir.iterdir() if p.is_file()):
        h.update(fname.encode("utf-8"))
        h.update((adapter_dir / fname).read_bytes())
    return h.hexdigest()


def _attach_repair_lora(omni, cfg: Dict[str, object]) -> List[torch.nn.Parameter]:
    num_layers = omni.model.config.text_config.num_hidden_layers
    last_layers = int(cfg.get("repair_last_layers", cfg.get("last_layers", 4)))
    target_modules = list(cfg.get("repair_target_modules", cfg.get("target_modules", ["q_proj", "v_proj", "down_proj"])))
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


def _probe_behavior(omni, unit: KnowledgeUnit, variants: Sequence[int] = (2, 3)) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for v in variants:
        p, d2c, _ = question_prompt(unit, variant=v)
        probs, mass = omni.probabilities(p)
        disp = "ABCD"[int(np.argmax(probs))]
        out[f"v{v}"] = {"displayed_pred": disp, "canonical_pred": d2c[disp], "choice_mass": float(mass)}
    return out


def repair(
    omni,
    unit: KnowledgeUnit,
    candidate: str,
    retention_pool: Sequence[str],
    teacher_logits: Sequence[torch.Tensor],
    repair_config: Dict[str, object],
    save_dir: str,
) -> RepairAttempt:
    """Apply budgeted LoRA writeback for candidate using only retention_pool and damaged teacher logits."""
    if candidate not in {"A", "B", "C", "D"}:
        raise ValueError(f"Invalid repair candidate: {candidate}")

    pre_behavior = _probe_behavior(omni, unit)
    params = _attach_repair_lora(omni, repair_config)
    steps = int(repair_config.get("repair_steps", 32))
    lr = float(repair_config.get("repair_lr", repair_config.get("lr", 5e-4)))
    retain_weight = float(repair_config.get("repair_retain_weight", repair_config.get("retain_weight", 50.0)))
    l2_weight = float(repair_config.get("repair_l2_weight", repair_config.get("l2_weight", 1e-4)))
    retain_batch = int(repair_config.get("retain_batch_size", len(retention_pool)))
    use_perm = bool(repair_config.get("repair_use_permuted_target", True))
    write_strategy = str(repair_config.get("write_strategy", "standard_kl"))
    unconstrained = bool(repair_config.get("unconstrained", False))

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    variants = tuple(repair_config.get("repair_variants", (2, 3)))
    perm_schedule = [("A", "B", "C", "D"), ("D", "C", "B", "A")] if use_perm else [("A", "B", "C", "D")]
    logs: List[Dict[str, float]] = []

    for step in range(steps):
        omni.model.train()
        opt.zero_grad(set_to_none=True)
        var = variants[step % len(variants)]
        perm = perm_schedule[(step // len(variants)) % len(perm_schedule)]
        prompt, _, c2d = question_prompt(unit, variant=var, permutation=perm)
        disp_target = c2d[candidate]
        target_loss = omni.model(**omni.encode(prompt, disp_target)).loss

        kl_val = 0.0
        l2 = sum(p.float().square().sum() for p in params)
        if not unconstrained and len(retention_pool) > 0:
            n_ret = len(retention_pool)
            bsize = min(retain_batch, n_ret)
            idxs = [(step * bsize + k) % n_ret for k in range(bsize)]
            r_prompts = [retention_pool[j] for j in idxs]
            logits = omni.model(**omni.encode_many(r_prompts)).logits[:, -1].float()
            t = torch.stack([teacher_logits[j] for j in idxs]).to(logits.device)
            kl = torch.nn.functional.kl_div(logits.log_softmax(-1), t.softmax(-1), reduction="batchmean")
            kl_val = float(kl.detach())

            if write_strategy == "orthogonal_retain_proj":
                # Project target gradient orthogonal to retention gradient before combining
                target_loss.backward(retain_graph=True)
                g_t = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                opt.zero_grad(set_to_none=True)
                (retain_weight * kl + l2_weight * l2).backward()
                g_r = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                flat_t = torch.cat([g.float().flatten() for g in g_t])
                flat_r = torch.cat([g.float().flatten() for g in g_r])
                denom = float(flat_r.dot(flat_r)) + 1e-12
                dot_tr = float(flat_t.dot(flat_r))
                proj_scale = dot_tr / denom if dot_tr < 0.0 else 0.0
                for p, gt, gr in zip(params, g_t, g_r):
                    p.grad = (gt - proj_scale * gr) + gr
            else:
                loss = target_loss + retain_weight * kl + l2_weight * l2
                loss.backward()
        else:
            loss = target_loss + l2_weight * l2
            loss.backward()

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        logs.append(
            {
                "step": step,
                "target_loss": float(target_loss.detach()),
                "retain_kl": kl_val,
                "l2": float(l2.detach()),
            }
        )

    omni.model.eval()
    adapter_dir = Path(save_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    omni.model.save_pretrained(str(adapter_dir), save_embedding_layers=False)
    sha = _adapter_sha256(adapter_dir)

    lora_frob = float(torch.sqrt(sum(p.detach().float().square().sum() for p in params)).cpu())
    max_abs = float(max((p.detach().float().abs().max().item() for p in params), default=0.0))

    post_behavior = _probe_behavior(omni, unit)
    # Check in-training retention agreement
    if len(retention_pool) > 0:
        with torch.inference_mode():
            curr_logits = omni.retain_logits(list(retention_pool))
        ret_agree = float(np.mean([int(a.argmax()) == int(b.argmax()) for a, b in zip(teacher_logits, curr_logits)]))
    else:
        ret_agree = 1.0

    min_ret_agree = float(repair_config.get("min_retain_agreement", 0.90))
    rolled_back = ret_agree < min_ret_agree
    return RepairAttempt(
        unit_id=unit.unit_id,
        pre_target_behavior=pre_behavior,
        post_target_behavior=post_behavior,
        training_log=logs,
        weight_update_info={"lora_frobenius_norm": lora_frob, "max_abs_param": max_abs, "steps": steps},
        adapter_path=str(adapter_dir.resolve()),
        adapter_sha256=sha,
        retention_check={"train_retain_agreement": ret_agree, "passed": not rolled_back},
        persistent=False,
        rolled_back=rolled_back,
        failure_reason="train_retain_violation" if rolled_back else None,
        recovery_mode="parametric_lora",
    )
