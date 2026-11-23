"""Verify CPU-only audit script and modules never import krmm.model.Omni or require CUDA."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_offline_audit_does_not_import_omni_or_cuda():
    src = (REPO_ROOT / "scripts/v3/offline_audit.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith("krmm.model"), f"Forbidden import: {mod}"
            assert not mod.startswith("torch.cuda"), f"Forbidden import: {mod}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("krmm.model"), f"Forbidden import: {alias.name}"


def test_pipeline_p1_p3_import_is_cpu_safe_without_omni():
    assert "krmm.model" not in sys.modules
    import krmm.v3.pipeline_p1_p3  # noqa: F401
    assert "krmm.model" not in sys.modules
