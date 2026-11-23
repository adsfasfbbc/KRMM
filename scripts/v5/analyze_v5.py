#!/usr/bin/env python3
"""Aggregate and audit all KRMM v0.5 experimental JSON outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize KRMM v0.5 results")
    parser.add_argument("--results-dir", default="results/v5-new")
    parser.add_argument("--out", default="results/v5-new/v5_summary.json")
    args = parser.parse_args()

    rdir = Path(args.results_dir)
    g1 = _load(rdir / "g1_discovery_results.json")
    g2_mech = _load(rdir / "g2_mechanism_results.json")
    g2_rep = _load(rdir / "g2_repair_results.json")
    g3a = _load(rdir / "g3a_d2_bounded_results.json")
    g3b = _load(rdir / "g3b_second_model_results.json")

    summary: Dict[str, Any] = {
        "version": "krmm-v0.5",
        "files_present": {
            "g1_discovery": g1 is not None,
            "g2_mechanism": g2_mech is not None,
            "g2_repair": g2_rep is not None,
            "g3a_d2_bounded": g3a is not None,
            "g3b_second_model": g3b is not None,
        },
    }

    if g1:
        summary["g1_discovery"] = {
            "pool_counts": g1.get("pool_counts"),
            "behavioral_truth_counts": g1.get("behavioral_truth_counts"),
            "collateral_damage_count": g1.get("collateral_damage_count"),
            "budget_curves": g1.get("budget_curves"),
        }
    if g2_mech:
        summary["g2_mechanism"] = {
            "n_targets": g2_mech.get("n_targets"),
            "baseline_comparison": g2_mech.get("baseline_comparison"),
            "support_class_breakdown": g2_mech.get("support_class_breakdown"),
            "intervention_summary": g2_mech.get("intervention_summary"),
        }
    if g2_rep:
        summary["g2_repair"] = {
            "n_targets": g2_rep.get("n_targets"),
            "strategy_summary": g2_rep.get("strategy_summary"),
            "paired_bootstrap": g2_rep.get("paired_bootstrap"),
            "locality_summary": g2_rep.get("locality_summary"),
            "reliability_summary": g2_rep.get("reliability_summary"),
        }
    if g3a:
        summary["g3a_d2_bounded"] = {
            "mode": g3a.get("mode"),
            "model_path": g3a.get("model_path"),
            "summary": g3a.get("summary"),
        }
    if g3b:
        summary["g3b_second_model"] = {
            "mode": g3b.get("mode"),
            "model_path": g3b.get("model_path"),
            "summary": g3b.get("summary"),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
