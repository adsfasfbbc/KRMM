"""Verify ground-truth config vs run_meta consistency for P0 audit assertions."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_exp_write_strategy_and_d2_regularization_match_configs():
    exp06_cfg = json.loads((REPO_ROOT / "configs/v3/p2_dev12_orth_e6.json").read_text(encoding="utf-8"))
    exp07_cfg = json.loads((REPO_ROOT / "configs/v3/p2_dev36_full.json").read_text(encoding="utf-8"))
    exp08_cfg = json.loads((REPO_ROOT / "configs/v3/p2_cal36_full.json").read_text(encoding="utf-8"))
    exp09_cfg = json.loads((REPO_ROOT / "configs/v3/p2_test48_frozen.json").read_text(encoding="utf-8"))
    d2_cfg = json.loads((REPO_ROOT / "configs/v3/p0_d2_steps64.json").read_text(encoding="utf-8"))

    assert exp06_cfg["write_strategy"] == "orthogonal_retain_proj"
    assert exp07_cfg["write_strategy"] == "standard_kl"
    assert exp08_cfg["write_strategy"] == "standard_kl"
    assert exp09_cfg["write_strategy"] == "standard_kl"
    assert exp09_cfg["repair_steps"] == 32

    assert d2_cfg["damage_type"] == "D2_suppression"
    assert float(d2_cfg["damage_retain_weight"]) == 50.0
    assert float(d2_cfg.get("damage_l2_weight", 1e-4)) > 0.0

    # Cross-check run_meta.json on disk matches frozen configs
    for rel_dir, expected_ws in [
        ("results/v3-new/exp06-e5-e6-dev12", "orthogonal_retain_proj"),
        ("results/v3-new/exp07-p2-dev36", "standard_kl"),
        ("results/v3-new/exp08-p2-cal36", "standard_kl"),
        ("results/v3-new/exp09-p2-test48", "standard_kl"),
    ]:
        rm = json.loads((REPO_ROOT / rel_dir / "run_meta.json").read_text(encoding="utf-8"))
        assert rm["config"]["write_strategy"] == expected_ws
