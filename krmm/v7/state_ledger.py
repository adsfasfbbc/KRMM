"""Immutable M_0 base snapshot, exact cumulative state ledger, rollback verification & no-op merge check."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from krmm.v3.damage import restore_weights, snapshot_weights, state_sha256


@dataclass
class LedgerEntry:
    step: int
    unit_id: str
    entity_id: str
    domain: str
    candidate: str
    provisional: bool = True
    conflicted: bool = False
    adapter_path: Optional[str] = None
    adapter_hash: Optional[str] = None
    pre_hash: str = ""
    post_hash: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class StateLedger:
    """Manages exact M_0 anchor, committed adapter sequence, and verified rollback without subtraction drift."""

    def __init__(self, initial_state: Dict[str, torch.Tensor], lora_cfg: Dict[str, object]) -> None:
        self.initial_state = {k: v.detach().cpu().clone() for k, v in initial_state.items()}
        self.initial_hash = state_sha256(self.initial_state)
        self.lora_cfg = dict(lora_cfg)
        self.entries: List[LedgerEntry] = []

    def current_hash(self, backend) -> str:
        return state_sha256(snapshot_weights(backend, self.lora_cfg))

    def snapshot(self, backend) -> Tuple[Dict[str, torch.Tensor], str]:
        st = snapshot_weights(backend, self.lora_cfg)
        return st, state_sha256(st)

    def rollback_to_snapshot(self, backend, pre_state: Dict[str, torch.Tensor], expected_hash: str) -> bool:
        if hasattr(backend.model, "unload"):
            backend.model = backend.model.unload()
        restore_weights(backend, pre_state)
        actual_hash = self.current_hash(backend)
        return actual_hash == expected_hash

    def reconstruct_from_initial(self, backend) -> str:
        """Deterministically rebuild current state from immutable M_0 + committed adapter paths."""
        if hasattr(backend.model, "unload"):
            backend.model = backend.model.unload()
        restore_weights(backend, self.initial_state)
        for e in self.entries:
            if e.adapter_path:
                backend.model = PeftModel.from_pretrained(backend.model, e.adapter_path, is_trainable=False)
                backend.model = backend.model.merge_and_unload()
        for p in backend.model.parameters():
            p.requires_grad_(False)
        backend.model.eval()
        return self.current_hash(backend)

    def committed_entries(self, active_only: bool = True) -> List[LedgerEntry]:
        if active_only:
            return [e for e in self.entries if not e.conflicted]
        return list(self.entries)

    def to_list(self) -> List[Dict[str, object]]:
        return [e.to_dict() for e in self.entries]


def compute_effective_delta_w_frobenius(peft_model) -> Dict[str, float]:
    """Compute exact effective ||DeltaW||_F = ||scaling * B @ A||_F across LoRA layers (NOT factor L2!)."""
    sq_sum = 0.0
    max_layer_norm = 0.0
    n_layers = 0
    for _, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and "default" in module.lora_A:
            A = module.lora_A["default"].weight.detach().float()
            B = module.lora_B["default"].weight.detach().float()
            scaling = float(module.scaling["default"]) if isinstance(module.scaling, dict) else float(module.scaling)
            dw = scaling * (B @ A)
            lnorm = float(torch.linalg.norm(dw, ord="fro").cpu())
            sq_sum += lnorm * lnorm
            max_layer_norm = max(max_layer_norm, lnorm)
            n_layers += 1
    return {
        "effective_delta_w_frobenius": float(sq_sum ** 0.5),
        "max_layer_delta_w_frobenius": max_layer_norm,
        "n_lora_modules": n_layers,
    }


def run_noop_lora_merge(backend, cfg: Dict[str, object], n_merges: int = 1) -> Dict[str, object]:
    """D1 diagnostic: attach zero-DeltaW LoRA (B=0) and merge_and_unload n_merges times to test dtype/merge drift."""
    pre_st = snapshot_weights(backend, cfg)
    pre_hash = state_sha256(pre_st)
    num_layers = backend.model.config.text_config.num_hidden_layers
    last_layers = int(cfg.get("last_layers", 4))
    layers_to_transform = list(range(num_layers - last_layers, num_layers))
    max_dw_norm = 0.0

    for _ in range(n_merges):
        lconf = LoraConfig(
            r=int(cfg.get("rank", 8)),
            lora_alpha=int(cfg.get("alpha", 16)),
            target_modules=list(cfg.get("target_modules", ["q_proj", "v_proj", "down_proj"])),
            layers_to_transform=layers_to_transform,
            lora_dropout=0.0,
            bias="none",
        )
        backend.model = get_peft_model(backend.model, lconf)
        # Verify B is strictly zero -> DeltaW == 0 mathematically
        dw_stats = compute_effective_delta_w_frobenius(backend.model)
        max_dw_norm = max(max_dw_norm, dw_stats["effective_delta_w_frobenius"])
        backend.model = backend.model.merge_and_unload()
        for p in backend.model.parameters():
            p.requires_grad_(False)
        backend.model.eval()

    post_st = snapshot_weights(backend, cfg)
    post_hash = state_sha256(post_st)
    max_abs_weight_diff = max(
        float((post_st[k].float() - pre_st[k].float()).abs().max().item()) for k in pre_st
    )
    return {
        "n_merges": n_merges,
        "pre_hash": pre_hash,
        "post_hash": post_hash,
        "hash_exact_match": pre_hash == post_hash,
        "max_effective_delta_w": max_dw_norm,
        "max_abs_weight_diff": max_abs_weight_diff,
    }
