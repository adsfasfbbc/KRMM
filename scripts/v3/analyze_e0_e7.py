"""Comprehensive E0-E7 matrix & M6.3 statistical analyzer for KRMM v0.3 (CPU-only safe)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from krmm.v3.metrics import paired_cluster_bootstrap_ci, summarize_p0_results
from krmm.v3.pipeline_p1_p3 import evaluate_selector_comparisons
from krmm.v3.repairability import (
    CandidateReliabilityPredictor,
    DecomposedPredictor,
    DirectSuccessPredictor,
)


def load_unit_records(run_dir: Path) -> List[dict]:
    recs = []
    skip = {"metrics.json", "run_meta.json", "damage-randomization-audit.json", "e0_e7_analysis.json"}
    for p in sorted(run_dir.glob("*.json")):
        if p.name in skip:
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(r, dict) and "unit" in r and "mastery" in r:
            recs.append(r)
    return recs


def summarize_e0_e4(
    records: List[dict],
    exploratory_or_confirmatory: str = "exploratory",
) -> Dict[str, object]:
    """Compute E0-E4 matrix with unified denominators over catalog, mastered, eligible, and attempted."""
    p0 = summarize_p0_results(records)
    elig = [r for r in records if r.get("strict_eligible")]
    n = len(elig)
    denominators = {
        "n_total_catalog": len(records),
        "n_stably_mastered": int(p0["stably_mastered"]),
        "n_damage_effective": int(p0["damage_effective"]),
        "n_strict_eligible": n,
    }
    if n == 0:
        return {"p0": p0, "denominators": denominators, "n_strict_eligible": 0}

    clusters = [r["unit"]["source_group"] for r in elig]
    gold = [r["gold_evaluator_only"] for r in elig]

    e0_acc = float(np.mean([r["damaged_heldout"]["accuracy"] for r in elig]))
    dir_cand_ok = [int(r["reconstruction_aux"]["direct_candidate"] == g) for r, g in zip(elig, gold)]
    gen_cand_ok = [int(r["reconstruction_aux"]["generative_candidate"] == g) for r, g in zip(elig, gold)]
    str_cov = [int(r["reconstruction"]["candidate_answer"] is not None) for r in elig]
    str_cand_ok = [int(r["reconstruction"]["candidate_answer"] == g) for r, g in zip(elig, gold)]
    true_comp = [r for r in elig if r["reconstruction_aux"].get("true_composition")]
    single_hop = [
        r for r in elig
        if r["reconstruction"]["candidate_answer"] is not None and not r["reconstruction_aux"].get("true_composition")
    ]

    branches = {}
    for b in ["direct", "generative", "structured", "oracle"]:
        vals = [int(r.get("repairs", {}).get(b, {}).get("strict_restoration", False)) for r in elig]
        loc_pass = [
            int(r.get("repairs", {}).get(b, {}).get("independent_locality_pass", False))
            for r in elig if b in r.get("repairs", {})
        ]
        wrong_pseudo = sum(
            1 for r, g in zip(elig, gold)
            if b in r.get("repairs", {}) and (
                (b == "direct" and r["reconstruction_aux"]["direct_candidate"] != g)
                or (b == "generative" and r["reconstruction_aux"]["generative_candidate"] != g)
                or (b == "structured" and r["reconstruction"]["candidate_answer"] != g)
            )
        )
        n_att = len(loc_pass)
        branches[b] = {
            "n_strict_eligible": n,
            "n_candidate_produced": sum(str_cov) if b == "structured" else n,
            "n_repair_attempted": n_att,
            "n_strict_restored": sum(vals),
            "strict_restoration_count": sum(vals),
            "strict_restoration_rate": sum(vals) / n,
            "strict_restoration_rate_among_attempted": sum(vals) / max(1, n_att),
            "attempted": n_att,
            "locality_pass_count": sum(loc_pass),
            "locality_pass_rate_attempted": sum(loc_pass) / max(1, n_att),
            "wrong_pseudolabels_written": wrong_pseudo,
            "per_unit_vector": vals,
        }

    ci_str_vs_gen_cand = paired_cluster_bootstrap_ci(
        str_cand_ok, gen_cand_ok, clusters, exploratory_or_confirmatory=exploratory_or_confirmatory
    )
    ci_str_vs_dir_cand = paired_cluster_bootstrap_ci(
        str_cand_ok, dir_cand_ok, clusters, exploratory_or_confirmatory=exploratory_or_confirmatory
    )
    ci_str_vs_gen_strict = paired_cluster_bootstrap_ci(
        branches["structured"]["per_unit_vector"],
        branches["generative"]["per_unit_vector"],
        clusters,
        exploratory_or_confirmatory=exploratory_or_confirmatory,
    )
    ci_str_vs_dir_strict = paired_cluster_bootstrap_ci(
        branches["structured"]["per_unit_vector"],
        branches["direct"]["per_unit_vector"],
        clusters,
        exploratory_or_confirmatory=exploratory_or_confirmatory,
    )

    return {
        "p0": p0,
        "denominators": {
            **denominators,
            "n_structured_candidate_produced": sum(str_cov),
            "n_structured_repair_attempted": branches["structured"]["n_repair_attempted"],
            "n_structured_strict_restored": branches["structured"]["n_strict_restored"],
        },
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


def _fit_all_predictors(fit_recs: List[dict], branch: str):
    valid_tr = [r for r in fit_recs if r.get("strict_eligible") and branch in r.get("repairs", {})]
    pred_decomp = DecomposedPredictor()
    pred_direct = DirectSuccessPredictor()
    pred_rel = CandidateReliabilityPredictor()
    if valid_tr:
        X = np.asarray([r["diagnostic_features"] for r in valid_tr], dtype=np.float64)
        if branch == "structured":
            yc = [int(r["reconstruction"]["candidate_answer"] == r["gold_evaluator_only"]) for r in valid_tr]
        else:
            yc = [int(r["reconstruction_aux"]["generative_candidate"] == r["gold_evaluator_only"]) for r in valid_tr]
        yw = [int(r["repairs"][branch]["strict_restoration"]) for r in valid_tr]
        pred_decomp.fit(X, yc, yw)
        pred_direct.fit(X, yw)
        pred_rel.fit(X, yc)
    return valid_tr, pred_decomp, pred_direct, pred_rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-dir", required=True)
    ap.add_argument("--cal-dir", default=None)
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev_recs = load_unit_records(Path(args.dev_dir))
    report = {"dev": summarize_e0_e4(dev_recs, exploratory_or_confirmatory="exploratory")}

    if args.cal_dir:
        cal_recs = load_unit_records(Path(args.cal_dir))
        report["calibration"] = summarize_e0_e4(cal_recs, exploratory_or_confirmatory="exploratory")
        for fit_name, fit_recs, eval_name, eval_recs in [
            ("dev", dev_recs, "calibration", cal_recs),
            ("calibration", cal_recs, "dev", dev_recs),
        ]:
            for branch in ("structured", "generative"):
                valid_tr, pred_d, pred_dir, pred_rel = _fit_all_predictors(fit_recs, branch)
                report[f"e7_fit_{fit_name}_eval_{eval_name}_{branch}"] = {
                    "decomposed_status": pred_d.calibration_status,
                    "write_model_stats": pred_d.write_predictor.fit_stats,
                    "direct_model_status": pred_dir.calibration_status,
                    "reliability_model_status": pred_rel.calibration_status,
                    "train_n": len(valid_tr),
                    "comparison": evaluate_selector_comparisons(
                        eval_recs,
                        budget_fraction=0.5,
                        predictor=pred_d,
                        direct_predictor=pred_dir,
                        reliability_predictor=pred_rel,
                        branch=branch,
                        exploratory_or_confirmatory="exploratory",
                    ),
                }

    if args.test_dir:
        test_recs = load_unit_records(Path(args.test_dir))
        report["test"] = summarize_e0_e4(test_recs, exploratory_or_confirmatory="post_hoc_audit")
        if args.cal_dir:
            cal_recs = load_unit_records(Path(args.cal_dir))
            train_pool = dev_recs + cal_recs
            for branch in ("structured", "generative"):
                valid_tr, pred_d, pred_dir, pred_rel = _fit_all_predictors(train_pool, branch)
                report[f"e7_posthoc_audit_test_{branch}"] = {
                    "decomposed_status": pred_d.calibration_status,
                    "write_model_stats": pred_d.write_predictor.fit_stats,
                    "direct_model_status": pred_dir.calibration_status,
                    "reliability_model_status": pred_rel.calibration_status,
                    "train_n": len(valid_tr),
                    "comparison": evaluate_selector_comparisons(
                        test_recs,
                        budget_fraction=0.5,
                        predictor=pred_d,
                        direct_predictor=pred_dir,
                        reliability_predictor=pred_rel,
                        branch=branch,
                        exploratory_or_confirmatory="post_hoc_audit",
                    ),
                }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
