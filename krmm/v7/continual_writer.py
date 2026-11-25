"""Continual Safe Knowledge Repair Writer Matrix: W0, W1, W2, W3, W4, W5 (Section 5.2 & 5.3)."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import torch
from peft import LoraConfig, get_peft_model

from krmm.v3.knowledge_registry import question_prompt
from krmm.v3.schemas import KnowledgeUnit
from .state_ledger import compute_effective_delta_w_frobenius

WRITER_VARIANTS = ("W0_standard_kl", "W1_generic_matched", "W2_relation_retain", "W3_history_replay", "W4_relation_plus_history", "W5_joint_multitarget")


def _adapter_sha256(adapter_dir: Path) -> str:
    h = hashlib.sha256()
    for fname in sorted(p.name for p in adapter_dir.iterdir() if p.is_file()):
        h.update(fname.encode("utf-8"))
        h.update((adapter_dir / fname).read_bytes())
    return h.hexdigest()


def _attach_lora(backend, cfg: Dict[str, object]) -> List[torch.nn.Parameter]:
    num_layers = backend.model.config.text_config.num_hidden_layers
    last_layers = int(cfg.get("last_layers", 4))
    layers_to_transform = list(range(num_layers - last_layers, num_layers))
    conf = LoraConfig(
        r=int(cfg.get("rank", 8)),
        lora_alpha=int(cfg.get("alpha", 16)),
        target_modules=list(cfg.get("target_modules", ["q_proj", "v_proj", "down_proj"])),
        layers_to_transform=layers_to_transform,
        lora_dropout=0.0,
        bias="none",
    )
    conf.base_model_name_or_path = str(cfg["model_path"])
    backend.model = get_peft_model(backend.model, conf)
    params = [p for p in backend.model.parameters() if p.requires_grad]
    for p in params:
        p.requires_grad_(True)
    return params


def fit_continual_step(
    backend,
    unit: KnowledgeUnit,
    candidate: str,
    generic_retains: Sequence[str],
    generic_teacher_logits: Sequence[torch.Tensor],
    cfg: Dict[str, object],
    save_dir: Path,
    writer_mode: str = "W0_standard_kl",
    extra_retains: Sequence[str] = (),
    extra_teacher_logits: Sequence[torch.Tensor] = (),
    history_pairs: Sequence[Tuple[str, str]] = (),
) -> Dict[str, object]:
    """Fit one budgeted LoRA step under W0/W1/W2/W3/W4 and save compact adapter to save_dir.

    Leaves backend.model as PeftModel (caller decides commit/merge vs rollback/unload!).
    Tracks exact effective ||DeltaW||_F = ||scaling * B @ A||_F.
    """
    t0 = time.perf_counter()
    params = _attach_lora(backend, cfg)
    steps = int(cfg.get("repair_steps", 32))
    lr = float(cfg.get("repair_lr", 5e-4))
    total_retain_w = float(cfg.get("repair_retain_weight", 50.0))
    l2_w = float(cfg.get("repair_l2_weight", 1e-4))
    hist_w = float(cfg.get("history_replay_weight", 0.5))
    batch_size = int(cfg.get("retain_batch_size", 24))

    # Fixed total retain weight budget (Section 5.3): split between generic and extra (W1 generic-matched or W2/W4 relation)
    use_extra = writer_mode in ("W1_generic_matched", "W2_relation_retain", "W4_relation_plus_history") and len(extra_retains) > 0
    use_hist = writer_mode in ("W3_history_replay", "W4_relation_plus_history") and len(history_pairs) > 0

    if use_extra:
        lambda_g = 0.5 * total_retain_w
        lambda_r = 0.5 * total_retain_w
    else:
        lambda_g = total_retain_w
        lambda_r = 0.0

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    # 4 balanced option permutations on target training prompts to minimize option-letter bias
    perm_schedule = [
        ("A", "B", "C", "D"),
        ("D", "C", "B", "A"),
        ("B", "D", "A", "C"),
        ("C", "A", "D", "B"),
    ]
    variants = (0, 1)
    n_gen = len(generic_retains)
    n_ext = len(extra_retains)
    n_hist = len(history_pairs)

    for step in range(steps):
        backend.model.train()
        opt.zero_grad(set_to_none=True)
        var = variants[step % len(variants)]
        perm = perm_schedule[(step // len(variants)) % len(perm_schedule)]
        prompt, _, c2d = question_prompt(unit, variant=var, permutation=perm)
        disp_cand = c2d[candidate]
        loss_target = backend.model(**backend.encode(prompt, disp_cand)).loss

        # Generic retain KL
        bsize = min(batch_size, n_gen)
        idxs = [(step * bsize + k) % n_gen for k in range(bsize)]
        r_prompts = [generic_retains[j] for j in idxs]
        logits = backend.model(**backend.encode_many(r_prompts)).logits[:, -1].float()
        t_ref = torch.stack([generic_teacher_logits[j] for j in idxs]).to(logits.device)
        kl_gen = torch.nn.functional.kl_div(logits.log_softmax(-1), t_ref.softmax(-1), reduction="batchmean")

        loss = loss_target + lambda_g * kl_gen

        # Relation-specific (W2/W4) or count-matched generic (W1) retain KL
        if use_extra:
            e_prompts = list(extra_retains)
            e_logits = backend.model(**backend.encode_many(e_prompts)).logits[:, -1].float()
            e_ref = torch.stack(list(extra_teacher_logits)).to(e_logits.device)
            kl_ext = torch.nn.functional.kl_div(e_logits.log_softmax(-1), e_ref.softmax(-1), reduction="batchmean")
            loss = loss + lambda_r * kl_ext

        # Provisional history replay (W3/W4)
        if use_hist:
            hp_prompt, hp_ans = history_pairs[step % n_hist]
            loss_hist = backend.model(**backend.encode(hp_prompt, hp_ans)).loss
            loss = loss + hist_w * loss_hist

        l2 = sum(p.float().square().sum() for p in params)
        loss = loss + l2_w * l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    backend.model.eval()
    dw_stats = compute_effective_delta_w_frobenius(backend.model)
    save_dir.mkdir(parents=True, exist_ok=True)
    backend.model.save_pretrained(str(save_dir), save_embedding_layers=False)
    sha = _adapter_sha256(save_dir)

    return {
        "writer_mode": writer_mode,
        "adapter_path": str(save_dir.resolve()),
        "adapter_hash": sha,
        "train_steps": steps,
        "n_extra_retains": n_ext if use_extra else 0,
        "n_history_pairs": n_hist if use_hist else 0,
        **dw_stats,
        "elapsed": time.perf_counter() - t0,
    }


def fit_joint_multitarget(
    backend,
    units_and_candidates: Sequence[Tuple[KnowledgeUnit, str]],
    generic_retains: Sequence[str],
    generic_teacher_logits: Sequence[torch.Tensor],
    cfg: Dict[str, object],
    save_dir: Path,
    extra_retains: Sequence[str] = (),
    extra_teacher_logits: Sequence[torch.Tensor] = (),
) -> Dict[str, object]:
    """D4 / W5: Joint multi-target LoRA update matching total target-step budget (len * repair_steps)."""
    t0 = time.perf_counter()
    params = _attach_lora(backend, cfg)
    n_targets = max(1, len(units_and_candidates))
    steps_per_target = int(cfg.get("repair_steps", 32))
    total_steps = steps_per_target * n_targets
    lr = float(cfg.get("repair_lr", 5e-4))
    total_retain_w = float(cfg.get("repair_retain_weight", 50.0))
    l2_w = float(cfg.get("repair_l2_weight", 1e-4))
    batch_size = int(cfg.get("retain_batch_size", 24))

    use_extra = len(extra_retains) > 0
    lambda_g = 0.5 * total_retain_w if use_extra else total_retain_w
    lambda_r = 0.5 * total_retain_w if use_extra else 0.0

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    perm_schedule = [("A", "B", "C", "D"), ("D", "C", "B", "A"), ("B", "D", "A", "C"), ("C", "A", "D", "B")]
    variants = (0, 1)
    n_gen = len(generic_retains)
    n_ext = len(extra_retains)

    for step in range(total_steps):
        backend.model.train()
        opt.zero_grad(set_to_none=True)
        u_idx = step % n_targets
        unit, cand = units_and_candidates[u_idx]
        cycle = step // n_targets
        var = variants[cycle % len(variants)]
        perm = perm_schedule[(cycle // len(variants)) % len(perm_schedule)]
        prompt, _, c2d = question_prompt(unit, variant=var, permutation=perm)
        loss_target = backend.model(**backend.encode(prompt, c2d[cand])).loss

        bsize = min(batch_size, n_gen)
        idxs = [(step * bsize + k) % n_gen for k in range(bsize)]
        r_prompts = [generic_retains[j] for j in idxs]
        logits = backend.model(**backend.encode_many(r_prompts)).logits[:, -1].float()
        t_ref = torch.stack([generic_teacher_logits[j] for j in idxs]).to(logits.device)
        kl_gen = torch.nn.functional.kl_div(logits.log_softmax(-1), t_ref.softmax(-1), reduction="batchmean")
        loss = loss_target + lambda_g * kl_gen

        if use_extra:
            eb = min(12, n_ext)
            eidxs = [(step * eb + k) % n_ext for k in range(eb)]
            e_prompts = [extra_retains[j] for j in eidxs]
            e_logits = backend.model(**backend.encode_many(e_prompts)).logits[:, -1].float()
            e_ref = torch.stack([extra_teacher_logits[j] for j in eidxs]).to(e_logits.device)
            kl_ext = torch.nn.functional.kl_div(e_logits.log_softmax(-1), e_ref.softmax(-1), reduction="batchmean")
            loss = loss + lambda_r * kl_ext

        l2 = sum(p.float().square().sum() for p in params)
        loss = loss + l2_w * l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    backend.model.eval()
    dw_stats = compute_effective_delta_w_frobenius(backend.model)
    save_dir.mkdir(parents=True, exist_ok=True)
    backend.model.save_pretrained(str(save_dir), save_embedding_layers=False)
    sha = _adapter_sha256(save_dir)

    return {
        "writer_mode": "W5_joint_multitarget",
        "adapter_path": str(save_dir.resolve()),
        "adapter_hash": sha,
        "train_steps": total_steps,
        "n_targets": n_targets,
        **dw_stats,
        "elapsed": time.perf_counter() - t0,
    }
