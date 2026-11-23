"""Analyze v0.4 CPU mechanism and mock pipeline outputs (CPU-only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p_reaudit = REPO_ROOT / "results/v4-new/v3_evidence_reaudit.json"
    p_mech = REPO_ROOT / "results/v4-new/cpu_mechanism_results.json"
    p_mat = REPO_ROOT / "results/v4-new/repair_matrix_mock.json"
    summary = {
        "v3_evidence_reaudit_present": p_reaudit.exists(),
        "cpu_mechanism_present": p_mech.exists(),
        "repair_matrix_mock_present": p_mat.exists(),
    }
    if p_reaudit.exists():
        d = json.loads(p_reaudit.read_text(encoding="utf-8"))
        summary["v4_evidence_classes_v3_archive"] = d["by_v4_evidence_class"]
        summary["persistence_142_clarification"] = d["persistence_142_breakdown"]["clarification"]
    if p_mech.exists():
        summary["cpu_mechanism_scope"] = json.loads(p_mech.read_text(encoding="utf-8"))["scope_warning"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
