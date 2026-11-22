"""Automated markdown report and gate report generator for KRMM v0.3 (Section 16)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def render_p0_summary_markdown(run_dir: Path) -> str:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    g0 = metrics["gate_g0"]
    lines = [
        f"## Run: `{run_dir.name}`",
        "",
        f"- **Timestamp (UTC):** {meta['utc_start']} -> {metrics['utc_end']}",
        f"- **Git Commit:** `{meta['git_commit']}`",
        f"- **v3 Source SHA256:** `{meta['v3_source_sha256']}`",
        f"- **Config SHA256:** `{meta['config_sha256']}`",
        f"- **Split / Domain / Limit:** `{meta['split']}` / `{meta['domain']}` / `{meta['limit']}`",
        f"- **Elapsed Seconds:** {metrics['elapsed_seconds']:.1f}s",
        "",
        "### P0 / G0 Acceptance Checklist (A, B, C, D)",
        "",
        f"- **A. Clean Mastery >= 0.75:** {metrics['stably_mastered']}/{metrics['total_candidates']} ({metrics['clean_mastery_rate']:.1%}) -> `{'PASS' if g0['A_clean_mastery_ok'] else 'FAIL'}` (Heldout also mastered: {metrics['heldout_clean_mastered']}/{metrics['total_candidates']})",
        f"- **B. Damage Effective & Independent Locality Pass:** {metrics['strict_eligible']}/{metrics['stably_mastered']} mastered ({metrics['strict_eligibility_rate']:.1%}, min_train_retain={metrics['min_damage_train_retention']:.3f}) -> `{'PASS' if g0['B_damage_local_and_effective_ok'] else 'FAIL'}`",
        f"- **C. Oracle Strict Writeback >= 0.75:** {metrics['oracle_strict_restorations']}/{metrics['strict_eligible']} strict-eligible ({metrics['oracle_strict_rate_among_strict_eligible']:.1%}) [Write feasible among counterfactual: {metrics['oracle_write_feasible_count']}/{metrics['evaluated_counterfactual']}] -> `{'PASS' if g0['C_oracle_strict_writeback_ok'] else 'FAIL'}`",
        f"- **D. Exclusion / Failure Breakdown:** `{json.dumps(metrics['exclusion_breakdown'])}`",
        f"- **Overall G0 Status:** **{'PASS' if g0['pass'] else 'FAIL'}** (Operational-only check: `{g0['operational_only_ok']}`)",
        "",
    ]
    return "\n".join(lines)
