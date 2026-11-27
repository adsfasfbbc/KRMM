"""FP32 protection bases and PEFT LoRA factors for DEV-V8."""
from contextlib import ExitStack

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize

from krmm.v3.knowledge_registry import question_prompt
from krmm.v7.continual_writer import _attach_lora

PERMUTATIONS = ("ABCD", "DCBA", "BDAC", "CADB")


class NullProjection(nn.Module):
    def __init__(self, basis):
        super().__init__()
        self.register_buffer("basis", basis)

    def forward(self, weight):
        fp32 = weight.float()
        return (fp32 - (fp32 @ self.basis) @ self.basis.T).to(weight.dtype)


def orthogonal_basis(samples, limit):
    # SVD drops numerical null directions instead of protecting arbitrary QR columns.
    left, singular, _ = torch.linalg.svd(samples.T.float(), full_matrices=False)
    nonzero = int((singular > singular[0] * 1e-5).sum())
    if nonzero == 0:
        raise ValueError("Protection activations contain no numerical directions")
    return left[:, :min(limit, nonzero)].contiguous()


@torch.inference_mode()
def collect_basis(backend, weights, prompts, limit):
    activations = {}
    modules = dict(backend.model.named_modules())

    def capture(key):
        def hook(module, inputs):
            value = inputs[0]
            flat = value[mask] if value.ndim == 3 else value[mask.flatten()]
            positions = torch.linspace(0, flat.shape[0] - 1, min(limit, flat.shape[0]), device=flat.device).long()
            activations[key] = flat[positions].float()
        return hook

    batch = backend.encode_many(prompts)
    mask = batch["attention_mask"].bool()
    backend.model.eval()
    with ExitStack() as stack:
        for key in weights:
            handle = modules[key.removesuffix(".weight")].register_forward_pre_hook(capture(key))
            stack.callback(handle.remove)
        backend.model(**batch)
    return {key: orthogonal_basis(rows, limit) for key, rows in activations.items()}


def random_bases(real, seed):
    bases = {}
    for index, (key, basis) in enumerate(real.items()):
        random = torch.Generator(device=basis.device).manual_seed(seed + index)
        values = torch.randn(basis.shape, generator=random, device=basis.device, dtype=torch.float32)
        bases[key] = torch.linalg.qr(values, mode="reduced").Q
    return bases


def active_layers(model, weights):
    for name, layer in model.get_base_model().named_modules():
        if hasattr(layer, "lora_A") and "default" in layer.lora_A:
            key = name + ".weight"
            if key not in weights or layer.fan_in_fan_out or layer.use_dora["default"]:
                raise RuntimeError(f"Unexpected LoRA layer: {name}")
            yield key, layer


def delta(layer):
    a = layer.lora_A["default"].weight.detach().float()
    b = layer.lora_B["default"].weight.detach().float()
    return (b @ a) * layer.scaling["default"]


def norm(tensor):
    return float(torch.linalg.vector_norm(tensor.float()))


def relative_residual(matrix, basis):
    return norm(matrix @ basis) / max(norm(matrix), 1e-12)


def fit_update(backend, unit, gold, cfg, weights, retains, teacher, extra, extra_teacher, bases, matched):
    params = _attach_lora(backend, cfg)
    layers = dict(active_layers(backend.model, weights))
    if set(layers) != set(weights):
        raise RuntimeError("Weight snapshot and LoRA layers differ")

    if bases:
        for key, layer in layers.items():
            # Clone inference-mode SVD output before autograd needs to save it.
            parametrize.register_parametrization(layer.lora_A["default"], "weight", NullProjection(bases[key].clone()))
        params = [parameter for parameter in backend.model.parameters() if parameter.requires_grad]

    steps = cfg["repair_steps"]
    n_retains = min(cfg["retain_batch_size"], len(retains))
    strength = cfg["repair_retain_weight"]
    optimizer = torch.optim.AdamW(params, lr=cfg["repair_lr"], weight_decay=0.0)
    train_tokens = backend.compute_tokens
    backend.model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prompt, _, canonical_to_display = question_prompt(unit, step % 2, tuple(PERMUTATIONS[(step // 2) % 4]))
        target_loss = backend.model(**backend.encode(prompt, canonical_to_display[gold])).loss
        indices = [(step * n_retains + offset) % len(retains) for offset in range(n_retains)]
        logits = backend.model(**backend.encode_many([retains[index] for index in indices])).logits[:, -1].float()
        reference = torch.stack([teacher[index] for index in indices]).to(logits.device)
        loss = target_loss + (strength / 2 if extra else strength) * F.kl_div(
            logits.log_softmax(-1), reference.softmax(-1), reduction="batchmean",
        )
        if extra:
            logits = backend.model(**backend.encode_many(extra)).logits[:, -1].float()
            reference = torch.stack(extra_teacher).to(logits.device)
            loss = loss + strength / 2 * F.kl_div(logits.log_softmax(-1), reference.softmax(-1), reduction="batchmean")
        loss = loss + cfg["repair_l2_weight"] * sum(p.float().square().sum() for p in params)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
    backend.model.eval()

    with torch.no_grad():
        if bases:
            for layer in layers.values():
                parametrize.remove_parametrizations(layer.lora_A["default"], "weight", leave_parametrized=True)
        if matched:
            for key, layer in layers.items():
                current = norm(delta(layer))
                if current <= 1e-12:
                    raise RuntimeError(f"No usable equal-norm update in {key}")
                layer.lora_B["default"].weight.mul_(matched[key] / current)

        measurements = {}
        for key, layer in layers.items():
            matrix = delta(layer)
            value = norm(matrix)
            if not torch.isfinite(matrix).all():
                raise FloatingPointError(f"Nonfinite effective update in {key}")
            measurements[key] = {"effective_norm": value}
            if bases:
                measurements[key].update(rank=bases[key].shape[1], analytic_residual=relative_residual(matrix, bases[key]))
    return measurements, backend.compute_tokens - train_tokens


def audit_merged(before, after, measurements, protection, applied_bases, target_norms):
    for key, measurement in measurements.items():
        actual = after[key].to(protection[key].device).float() - before[key].to(protection[key].device).float()
        measurement["merged_norm"] = norm(actual)
        measurement["protection_residual"] = relative_residual(actual, protection[key])
        if applied_bases:
            measurement["merged_residual"] = relative_residual(actual, applied_bases[key])
        if target_norms:
            measurement["matched_relative_error"] = abs(measurement["merged_norm"] - target_norms[key]) / max(target_norms[key], 1e-12)
