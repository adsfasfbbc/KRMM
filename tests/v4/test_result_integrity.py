"""Verify GPU execution guard and immutability of v0.3 protected artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path
import pytest

from krmm.v4.repair_adapter import RepairAdapter

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_live_gpu_repair_blocked_without_mock_mode():
    adapter = RepairAdapter(strategy="standard_kl", mock_mode=False)
    with pytest.raises(RuntimeError):
        adapter.apply_repair("u1", "A")


def test_v3_protected_files_remain_untouched():
    expected = {
        "docs/v3/EXPERIMENTS.md": "c34d89ecafff6889f6e676eb34f639854e68d5321ccb8c31849f1011bb0a4b63",
        "docs/v3/GATE_REPORT.md": "f7d6f9f51e2cc9a19888cad81e02bc4cb077ff848cb8635dbcfc131e00401961",
        "results/v3-new/e0_e7_full_final.json": "0a4b6c596747e0a540e2573a7fe209d7d9b9100d1a66883d1636fb97b19fbab3",
    }
    for rel, h in expected.items():
        actual = hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()
        assert actual == h, f"Protected v3 artifact modified: {rel}"
