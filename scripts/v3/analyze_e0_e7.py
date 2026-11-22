"""Comprehensive E0-E7 matrix & M6.3 statistical analyzer for KRMM v0.3."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from krmm.v3.metrics import paired_cluster_bootstrap_ci, safe_auprc, safe_auroc, safe_brier, summarize_p0_results
from krmm.v3.pipeline_p1_p3 import evaluate_selector_comparisons
from krmm.v3.repairability import DecomposedPredictor


def load_unit_records(run_dir: Path) -> List[dict]:
    recs = []
    skip = {"metrics.json", "run_meta.json", "damage-randomization-audit.json"}
    for p in sorted(run_dir.glob("*.json")):
        if p.name in skip:
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if "unit" in r and "mastery" in r:
            recs.append(r)
    return recs


def summarize_e0_e4(records: List[dict]) -> Dict[str, object]:
    """Compute E0-E4 matrix over all strict-eligible units (plus full-pool denominators)."""
    p0 = summarize_p0_results(records)
    elig = [r for r in records if r.get("strict_eligible")]
    n = len(elig)
    if n == 0:
        return {"p0": p0, "n_strict_eligible": 0}

    clusters = [r["unit"]["source_group"] for r in elig]
    gold = [r["gold_evaluator_only"] for r in elig]

    # E0: damaged heldout accuracy
    e0_acc = float(np.mean([r["damaged_heldout"]["accuracy"] for r in elig]))
    # E1: direct & generative candidate accuracy (no writeback) and post-writeback
    dir_cand_ok = [int(r["reconstruction_aux"]["direct_candidate"] == g) for r, g in zip(elig, gold)]
    gen_cand_ok = [int(r["reconstruction_aux"]["generative_candidate"] == g) for r, g in zip(elig, gold)]
    # E2: structured candidate accuracy, coverage, composition breakdown
    str_cov = [int(r["reconstruction"]["candidate_answer"] is not None) for r in elig]
    str_cand_ok = [int(r["reconstruction"]["candidate_answer"] == g) for r, g in zip(elig, gold)]
    true_comp = [r for r in elig if r["reconstruction_aux"].get("true_composition")]
    single_hop = [
        r for r in elig
        if r["reconstruction"]["candidate_answer"] is not None and not r["reconstruction_aux"].get("true_composition")
    ]

    # E3 / E4: strict restoration after writeback
    branches = {}
    for b in ["direct", "generative", "structured", "oracle"]:
        vals = [int(r.get("repairs", {}).get(b, {}).get("strict_restoration", False)) for r in elig]
        loc_pass = [int(r.get("repairs", {}).get(b, {}).get("independent_locality_pass", False)) for r in elig if b in r.get("repairs", {})]
        wrong_pseudo = sum(
            1 for r, g in zip(elig, gold)
            if b in r.get("repairs", {}) and (
                (b == "direct" and r["reconstruction_aux"]["direct_candidate"] != g)
                or (b == "generative" and r["reconstruction_aux"]["generative_candidate"] != g)
                or (b == "structured" and r["reconstruction"]["candidate_answer"] != g)
            )
        )
        branches[b] = {
            "strict_restoration_count": sum(vals),
            "strict_restoration_rate": sum(vals) / n,
            "attempted": len(loc_pass),
            "locality_pass_rate_attempted": sum(loc_pass) / max(1, len(loc_pass)),
            "wrong_pseudolabels_written": wrong_pseudo,
            "per_unit_vector": vals,
        }

    # Paired cluster bootstrap comparisons: structured vs generative & structured vs direct
    ci_str_vs_gen_cand = paired_cluster_bootstrap_ci(str_cand_ok, gen_cand_ok, clusters)
    ci_str_vs_dir_cand = paired_cluster_bootstrap_ci(str_cand_ok, dir_cand_ok, clusters)
    ci_str_vs_gen_strict = paired_cluster_bootstrap_ci(
        branches["structured"]["per_unit_vector"],
        branches["generative"]["per_unit_vector"],
        clusters,
    )
    ci_str_vs_dir_strict = paired_cluster_bootstrap_ci(
        branches["structured"]["per_unit_vector"],
        branches["direct"]["per_unit_vector"],
        clusters,
    )

    return {
        "p0": p0,
        "n_strict_eligible": n,
        "e0_damaged_heldout_mean_acc": e0_acc,
        "e1_direct_candidate_acc": float(np.mean(dir_cand_ok)),
        "e1_generative_candidate_acc": float(np.mean(gen_cand_ok)),
        "e2_structured_coverage": float(np.mean(str_cov)),
        "e2_structured_candidate_acc_all_eligible": float(np.mean(str_cand_ok)),
        "e2_structured_precision_when_non_null": sum(str_cand_ok) / max(1, sum(str_cov)),
        "e2_true_composition_count": len(true_comp),
        "e2_true_composition_correct": sum(1 for r in true_comp if r["reconstruction"]["candidate_answer"] == r["gold_evaluator_only"]),
        "e2_single_hop_count": len(single_hop),
        "e2_single_hop_correct": sum(1 for r in single_hop if r["reconstruction"]["candidate_answer"] == r["gold_evaluator_only"]),
        "branches_strict": {k: {kk: vv for kk, vv in v.items() if kk != "per_unit_vector"} for k, v in branches.items()},
        "paired_ci_structured_vs_generative_candidate": ci_str_vs_gen_cand,
        "paired_ci_structured_vs_direct_candidate": ci_str_vs_dir_cand,
        "paired_ci_structured_vs_generative_strict": ci_str_vs_gen_strict,
        "paired_ci_structured_vs_direct_strict": ci_str_vs_dir_strict,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-dir", required=True)
    ap.add_argument("--cal-dir", default=None)
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev_recs = load_unit_records(Path(args.dev_dir))
    report = {"dev": summarize_e0_e4(dev_recs)}

    if args.cal_dir:
        cal_recs = load_unit_records(Path(args.cal_dir))
        report["calibration"] = summarize_e0_e4(cal_recs)

        # Fit DecomposedPredictor on dev (or cal) and evaluate on held-out split (E7)
        for fit_name, fit_recs, eval_name, eval_recs in [
            ("dev", dev_recs, "calibration", cal_recs),
            ("calibration", cal_recs, "dev", dev_recs),
        ]:
            # Fit on generative+structured outcomes? For H2, test on BOTH structured and generative if structured has both classes
            for branch in ("structured", "generative"):
                valid_tr = [r for r in fit_recs if r.get("strict_eligible") and branch in r.get("repairs", {})]
                pred = DecomposedPredictor()
                if valid_tr:
                    X = np.asarray([r["diagnostic_features"] for r in valid_tr], dtype=np.float64)
                    if branch == "structured":
                        yc = [int(r["reconstruction"]["candidate_answer"] == r["gold_evaluator_only"]) for r in valid_tr]
                    else:
                        yc = [int(r["reconstruction_aux"]["generative_candidate"] == r["gold_evaluator_only"]) for r in valid_tr]
                    yw = [int(r["repairs"][branch]["strict_restoration"]) for r in valid_tr]
                    pred.fit(X, yc, yw)
                report[f"e7_fit_{fit_name}_eval_{eval_name}_{branch}"] = {
                    "calibration_status": pred.calibration_status,
                    "train_n": len(valid_tr),
                    "comparison": evaluate_selector_comparisons(eval_recs, budget_fraction=0.5, predictor=pred, branch=branch),
                }

    if args.test_dir:
        test_recs = load_unit_records(Path(args.test_dir))
        report["test"] = summarize_e0_e4(test_recs)
        if args.cal_dir:
            cal_recs = load_unit_records(Path(args.cal_dir))
            train_pool = dev_recs + cal_recs
            for branch in ("structured", "generative"):
                valid_tr = [r for r in train_pool if r.get("strict_eligible") and branch in r.get("repairs", {})]
                pred = DecomposedPredictor()
                if valid_tr:
                    X = np.asarray([r["diagnostic_features"] for r in valid_tr], dtype=np.float64)
                    if branch == "structured":
                        yc = [int(r["reconstruction"]["candidate_answer"] == r["gold_evaluator_only"]) for r in valid_tr]
                    else:
                        yc = [int(r["reconstruction_aux"]["generative_candidate"] == r["gold_evaluator_only"]) for r in valid_tr]
                    yw = [int(r["repairs"][branch]["strict_restoration"]) for r in valid_tr]
                    pred.fit(X, yc, yw)
                report[f"e7_confirmatory_test_{branch}"] = {
                    "calibration_status": pred.calibration_status,
                    "train_n": len(valid_tr),
                    "comparison": evaluate_selector_comparisons(test_recs, budget_fraction=0.5, predictor=pred, branch=branch),
                }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
