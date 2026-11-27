import os

import torch
from torch import nn
from torch.nn.utils import parametrize

from krmm.v8.constrained_writer import NullProjection, audit_merged, orthogonal_basis, relative_residual
from krmm.v8.dev_eval import score_open
from krmm.v8.isolation import require_denied, restrict
from scripts.v8.dev_observer import decide


def test_projection_survives_optimizer_and_materialization():
    generator = torch.Generator().manual_seed(5)
    basis = orthogonal_basis(torch.randn(6, 11, generator=generator), 4)
    factor = nn.Linear(11, 3, bias=False)
    parametrize.register_parametrization(factor, "weight", NullProjection(basis))
    optimizer = torch.optim.AdamW(factor.parameters(), lr=0.02)
    for _ in range(3):
        optimizer.zero_grad()
        factor(torch.randn(2, 11, generator=generator)).square().sum().backward()
        optimizer.step()
        assert relative_residual(factor.weight.detach(), basis) < 1e-5
    projected = factor.weight.detach().clone()
    parametrize.remove_parametrizations(factor, "weight", leave_parametrized=True)
    torch.testing.assert_close(factor.weight, projected)


def test_actual_bf16_audit_and_matched_error():
    weights = {"layer": torch.eye(4).to(torch.bfloat16)}
    proposed = weights["layer"].clone()
    proposed[0, 2] = 0.25
    basis = {"layer": torch.eye(4)[:, :1]}
    measurements = {"layer": {}}
    audit_merged(weights, {"layer": proposed}, measurements, basis, basis, {"layer": 0.25})
    assert measurements["layer"]["matched_relative_error"] == 0
    assert measurements["layer"]["merged_residual"] == 0


def test_open_rule_keeps_unknown_and_conflicts_out():
    assert score_open("With a bow.", "A bow", "A hammer") == "correct"
    assert score_open("A bow or a hammer.", "A bow", "A hammer") == "unclassified"
    assert score_open("A stick.", "A bow", "A hammer") == "unclassified"
    assert score_open("Not a bow.", "A bow", "A hammer") == "unclassified"


def test_evaluator_access_denied_in_child(tmp_path):
    public = tmp_path / "public"
    private = tmp_path / "private"
    public.write_text("public")
    private.write_text("private")
    process = os.fork()
    if process == 0:
        try:
            restrict([public], [], [])
            require_denied(private)
            assert public.read_text() == "public"
        except BaseException:
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(process, 0)
    assert os.waitstatus_to_exitcode(status) == 0


def test_no_test_after_wp_stops_on_constraint():
    regular = {"committed": 8, "same_entity_harm": 2, "healthy_harm": 0, "H_ever": 3, "active_restored": 8,
               "R_final": 1, "fresh_pass": True, "strict_yield": 0.0, "semantics": {"coverage": 1}}
    blocked = {**regular, "committed": 0, "active_restored": 0, "same_entity_harm": 0}
    cfg = {"oracle_targets": 8, "min_active": 6, "max_recovery_drop": 0.1, "min_semantic_coverage": 0.5}
    result = decide({"W0": regular, "W1": regular, "WN": regular, "WR": regular, "WP": blocked}, cfg)
    assert result["decision"] == "DO_NOT_ENTER_TEST"
