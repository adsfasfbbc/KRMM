import torch
from torch import nn
from torch.nn import functional as F

from krmm.v8.merge_precision import promote_edited_layers


def test_promoted_forward_uses_fp32_stored_weights_and_keeps_bf16_interface():
    layer = nn.Linear(4, 3, bias=True).to(torch.bfloat16)
    model = nn.Sequential(layer)
    anchor = {"0.weight": layer.weight.detach().clone()}
    promoted = promote_edited_layers(model, anchor)
    assert layer.weight.dtype == torch.float32
    assert torch.equal(layer.weight, promoted["0.weight"])
    data = torch.tensor([[0.5, -1.0, 0.25, 2.0]], dtype=torch.bfloat16, requires_grad=True)
    with torch.no_grad():
        layer.weight.add_(0.0001)
    actual = model(data)
    expected = F.linear(data.float(), layer.weight, layer.bias).bfloat16()
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)
    actual.float().sum().backward()
    assert data.grad is not None


def test_gate_selects_fp32_only_when_its_actual_merge_reloads():
    from scripts.v8.precision_pilot_worker import gate
    probe = [{"probabilities": [0.1, 0.2, 0.3, 0.4], "label": "D"}]
    states = {
        "unmerged": {"layers": {"w": {"rho": 1e-6, "adapter_forward_rho": 1e-6}}},
        "BF16": {"layers": {"w": {"rho": 0.05}}, "state_sha256": "bf", "probes": probe},
        "FP32": {"layers": {"w": {"rho": 1e-6}}, "state_sha256": "fp", "probes": probe},
    }
    assert gate(states, states, 0.01, 0.0001)["eligible_merge_dtype"] == "FP32"
    fresh = {**states, "FP32": {**states["FP32"], "state_sha256": "different"}}
    assert gate(states, fresh, 0.01, 0.0001)["status"] == "FAIL"
