"""Precision of the stored and executed weights in the twelve edited linear layers."""
import torch
from torch import nn


def promote_edited_layers(model, anchor):
    modules = dict(model.named_modules())
    for key, weight in anchor.items():
        if not key.endswith(".weight") or weight.dtype != torch.bfloat16:
            raise ValueError(f"Expected original BF16 linear weight: {key}")
        layer = modules[key.removesuffix(".weight")]
        if not isinstance(layer, nn.Linear) or layer.weight.dtype != torch.bfloat16:
            raise TypeError(f"Unexpected edited module for FP32 promotion: {key}")
        with torch.no_grad():
            layer.weight.data = layer.weight.data.float()
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.float()
        layer.register_forward_pre_hook(_float_input)
        layer.register_forward_hook(_bf16_output)
    return {key: weight.float() for key, weight in anchor.items()}


def _float_input(_layer, args):
    value, *rest = args
    if value.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"Unexpected input dtype for promoted layer: {value.dtype}")
    return (value.float(), *rest)


def _bf16_output(_layer, _args, result):
    return result.to(torch.bfloat16)
