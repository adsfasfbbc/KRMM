"""CPU-only offline scientific audit entrypoint for KRMM v0.3.

Strictly forbids importing krmm.model.Omni, loading weights, or invoking CUDA.
Reads archived JSON artifacts only and writes independent audit outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from krmm.v3.audit_cpu.legacy import LegacyBuggyDecomposedPredictor
from krmm.v3.metrics import paired_cluster_bootstrap_ci, safe_auprc, safe_auroc, safe_brier, summarize_p0_results
from krmm.v3.reconstruction import reconstruct_structured
from krmm.v3.repairability import (
    CandidateReliabilityPredictor,
    ConditionalWriteFeasibilityPredictor,
    DecomposedPredictor,
    DirectSuccessPredictor,
    baseline_score,
)
from krmm.v3.schemas import AtomicFact, CandidateReport


PROTECTED_FILES = [
    "docs/v3/EXPERIMENTS.md",
    "docs/v3/GATE_REPORT.md",
    "docs/v3/EVIDENCE_PACK.md",
    "docs/v3/PROTOCOL.md",
    "docs/v3/DATA_MANIFEST.jsonl",
    "results/v3-new/e0_e7_full_final.json",
    "configs/v3/p0_d2_steps64.json",
    "configs/v3/p2_dev12_orth_e6.json",
    "configs/v3/p2_dev36_full.json",
    "configs/v3/p2_cal36_full.json",
    "configs/v3/p2_test48_frozen.json",
]


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_unit_records(run_dir: Path) -> List[dict]:
    skip = {"metrics.json", "run_meta.json", "damage-randomization-audit.json", "e0_e7_analysis.json"}
    recs: List[dict] = []
    for p in sorted(run_dir.glob("*.json")):
        if p.name in skip:
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "unit" in obj and "mastery" in obj:
            recs.append(obj)
    return recs


def _collect_integrity() -> Dict[str, object]:
    if "krmm.model" in sys.modules:
        raise RuntimeError("Forbidden import detected: krmm.model must never be imported in offline_audit.py")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    git_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=str(REPO_ROOT), text=True
    ).strip()
    hashes = {rel: _sha256_file(REPO_ROOT / rel) for rel in PROTECTED_FILES}
    return {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha,
        "git_branch": git_branch,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        "krmm_model_imported": "krmm.model" in sys.modules,
        "python_version": sys.version.split()[0],
        "protected_file_sha256": hashes,
    }


def _audit_denominators_and_evidence_graph(name: str, recs: List[dict]) -> Dict[str, object]:
    p0 = summarize_p0_results(recs)
    elig = [r for r in recs if r.get("strict_eligible")]
    n_elig = len(elig)
    gold = [r["gold_evaluator_only"] for r in elig]
    clusters = [r["unit"]["source_group"] for r in elig]

    # Re-evaluate archived atomic_facts with corrected minimal-support EvidenceGraph
    eg_counts: Dict[str, int] = {
        "single_hop": 0,
        "minimal_multi_premise": 0,
        "redundant_support": 0,
        "conflicting_evidence": 0,
        "insufficient_evidence": 0,
    }
    eg_correct_by_class: Dict[str, int] = {k: 0 for k in eg_counts}
    candidate_diffs = 0
    orig_true_comp = 0
    new_true_comp = 0
    per_unit_eg = []
    for r, g in zip(elig, gold):
        domain = r["unit"]["domain"]
        raw_facts = r["reconstruction"].get("atomic_facts", [])
        facts = [AtomicFact(**f) for f in raw_facts]
        new_res = reconstruct_structured(domain, facts)
        orig_cand = r["reconstruction"].get("candidate_answer")
        new_cand = new_res["candidate"]
        if orig_cand != new_cand:
            candidate_diffs += 1
        if r.get("reconstruction_aux", {}).get("true_composition"):
            orig_true_comp += 1
        if new_res["true_composition"]:
            new_true_comp += 1
        ec = str(new_res["explanation_class"])
        eg_counts[ec] = eg_counts.get(ec, 0) + 1
        if new_cand == g:
            eg_correct_by_class[ec] = eg_correct_by_class.get(ec, 0) + 1
        per_unit_eg.append(
            {
                "unit_id": r["unit"]["unit_id"],
                "entity_id": r["unit"]["entity_id"],
                "domain": domain,
                "orig_candidate": orig_cand,
                "new_candidate": new_cand,
                "orig_true_composition": bool(r.get("reconstruction_aux", {}).get("true_composition")),
                "new_explanation_class": ec,
                "min_support_size": new_res["min_support_size"],
                "n_minimal_sets": len(new_res["minimal_support_sets"]),
                "candidate_correct": bool(new_cand == g),
            }
        )

    # Branch counts with explicit denominators
    branches = {}
    for b in ("direct", "generative", "structured", "oracle"):
        att = [r for r in elig if b in r.get("repairs", {})]
        restored = [r for r in att if r["repairs"][b].get("strict_restoration")]
        branches[b] = {
            "n_strict_eligible": n_elig,
            "n_repair_attempted": len(att),
            "n_strict_restored": len(restored),
            "rate_over_eligible": len(restored) / max(1, n_elig),
            "rate_over_attempted": len(restored) / max(1, len(att)),
        }

    # Bootstrap seed stability check (structured vs generative strict restoration)
    str_vec = [int(r.get("repairs", {}).get("structured", {}).get("strict_restoration", False)) for r in elig]
    gen_vec = [int(r.get("repairs", {}).get("generative", {}).get("strict_restoration", False)) for r in elig]
    seed_cis = {}
    for s in (107, 202, 303, 404, 505):
        ci = paired_cluster_bootstrap_ci(
            str_vec,
            gen_vec,
            clusters,
            seed=s,
            exploratory_or_confirmatory="post_hoc_audit" if name == "test" else "exploratory",
        )
        seed_cis[str(s)] = {"point_estimate": ci["point_estimate"], "ci95": ci["ci95"], "clusters": ci["clusters"]}

    return {
        "split": name,
        "denominators": {
            "n_total_catalog": len(recs),
            "n_stably_mastered": int(p0["stably_mastered"]),
            "n_damage_effective": int(p0["damage_effective"]),
            "n_strict_eligible": n_elig,
            "n_unique_source_groups_eligible": len(set(clusters)),
            "has_duplicate_source_groups_eligible": len(set(clusters)) < n_elig,
            "n_structured_candidate_produced": sum(
                1 for r in elig if r["reconstruction"].get("candidate_answer") is not None
            ),
            "n_structured_repair_attempted": branches["structured"]["n_repair_attempted"],
            "n_structured_strict_restored": branches["structured"]["n_strict_restored"],
        },
        "branches": branches,
        "evidence_graph_audit": {
            "candidate_differences_orig_vs_corrected": candidate_diffs,
            "orig_true_composition_count": orig_true_comp,
            "corrected_true_composition_count": new_true_comp,
            "explanation_class_counts": eg_counts,
            "explanation_class_correct_counts": eg_correct_by_class,
            "per_unit": per_unit_eg,
        },
        "bootstrap_seed_stability_structured_vs_generative": seed_cis,
    }


def _extract_xy(recs: List[dict], branch: str):
    elig = [r for r in recs if r.get("strict_eligible")]
    att = [r for r in elig if branch in r.get("repairs", {})]
    X = np.asarray([r["diagnostic_features"] for r in att], dtype=np.float64) if att else np.empty((0, 11))
    yc = []
    for r in att:
        g = r["gold_evaluator_only"]
        if branch == "structured":
            c = r["reconstruction"].get("candidate_answer")
        elif branch == "generative":
            c = r["reconstruction_aux"].get("generative_candidate")
        else:
            c = r["reconstruction_aux"].get("direct_candidate")
        yc.append(int(c == g))
    ys = [int(r["repairs"][branch]["strict_restoration"]) for r in att]
    yo = [int(r["repairs"]["oracle"]["strict_restoration"]) for r in att]
    return elig, att, X, yc, ys, yo


def _score_methods(
    eval_elig: List[dict],
    eval_att: List[dict],
    y_strict: List[int],
    y_cand_ok: List[int],
    score_map: Dict[str, Tuple[List[float], Dict[str, object]]],
    budget_fraction: float = 0.5,
) -> Dict[str, object]:
    n_elig = len(eval_elig)
    n_att = len(eval_att)
    cap = max(1, int(round(n_att * budget_fraction))) if n_att > 0 else 0
    out: Dict[str, object] = {}
    for m, (sc, meta) in score_map.items():
        if not sc or len(sc) != n_att:
            out[m] = {"status": meta.get("status", "unavailable"), **meta}
            continue
        ranked = sorted(range(n_att), key=lambda i: sc[i], reverse=True)[:cap]
        succ_sel = sum(y_strict[i] for i in ranked)
        wrong_sel = sum(1 for i in ranked if y_cand_ok[i] == 0)
        out[m] = {
            **meta,
            "auroc": safe_auroc(y_strict, sc),
            "auprc": safe_auprc(y_strict, sc),
            "brier": safe_brier(y_strict, [max(0.0, min(1.0, v)) for v in sc]),
            "selected_count": cap,
            "n_attempted": n_att,
            "n_strict_eligible": n_elig,
            "candidate_coverage": float(n_att / max(1, n_elig)),
            "selected_coverage_among_attempted": float(cap / max(1, n_att)),
            "selected_coverage_among_all_eligible": float(cap / max(1, n_elig)),
            "strict_success_count_selected": int(succ_sel),
            "strict_success_among_selected": float(succ_sel / max(1, cap)),
            "strict_success_all_eligible": float(succ_sel / max(1, n_elig)),
            "wrong_write_count_selected": int(wrong_sel),
            "wrong_write_rate_among_selected": float(wrong_sel / max(1, cap)),
            "abstention_rate_all_eligible": float((n_elig - n_att) / max(1, n_elig)),
        }
    return out


def _compare_repairability_models(
    train_recs: List[dict],
    eval_recs: List[dict],
    branch: str,
    cal_recs: Optional[List[dict]] = None,
    label: str = "exploratory",
) -> Dict[str, object]:
    _, tr_att, X_tr, yc_tr, ys_tr, yo_tr = _extract_xy(train_recs, branch)
    ev_elig, ev_att, X_ev, yc_ev, ys_ev, yo_ev = _extract_xy(eval_recs, branch)

    # Sparsity diagnostics on conditional write feasibility
    cond_pos = sum(1 for c, s in zip(yc_tr, ys_tr) if c == 1 and s == 1)
    cond_neg = sum(1 for c, s in zip(yc_tr, ys_tr) if c == 1 and s == 0)
    oracle_pos = sum(yo_tr)
    oracle_neg = len(yo_tr) - oracle_pos

    legacy = LegacyBuggyDecomposedPredictor()
    legacy.fit(X_tr, yc_tr, ys_tr)

    direct_m = DirectSuccessPredictor()
    direct_m.fit(X_tr, ys_tr)

    rel_m = CandidateReliabilityPredictor()
    rel_m.fit(X_tr, yc_tr)

    decomp_cond = DecomposedPredictor(write_label_source="correct_candidates")
    decomp_cond.fit(X_tr, yc_tr, ys_tr)

    decomp_oracle = DecomposedPredictor(write_label_source="oracle_writeback")
    decomp_oracle.fit(X_tr, yc_tr, ys_tr, X_oracle=X_tr, y_oracle_write_ok=yo_tr)

    # Optional independent calibration on cal_recs (e.g. fit dev -> calibrate cal -> eval test)
    direct_cal_m: Optional[DirectSuccessPredictor] = None
    if cal_recs is not None:
        _, _, X_c, yc_c, ys_c, _ = _extract_xy(cal_recs, branch)
        direct_cal_m = DirectSuccessPredictor()
        direct_cal_m.fit(X_tr, ys_tr)
        direct_cal_m.calibrate(X_c, ys_c)

    score_map: Dict[str, Tuple[List[float], Dict[str, object]]] = {}

    if legacy.calibration_status != "insufficient_data":
        score_map["legacy_buggy_ere"] = (
            [float(legacy.predict(x)["repair_success"] or 0.0) for x in X_ev],
            {"status": legacy.calibration_status, "is_calibrated_probability": False, "note": "double_counted_bug"},
        )
    if direct_m.calibration_status in {"fitted_uncalibrated", "calibrated"}:
        score_map["direct_success_model"] = (
            [float(direct_m.predict_proba_one(x) or 0.0) for x in X_ev],
            {"status": direct_m.calibration_status, "is_calibrated_probability": direct_m.calibration_status == "calibrated"},
        )
    else:
        score_map["direct_success_model"] = ([], {"status": direct_m.calibration_status})

    if rel_m.calibration_status in {"fitted_uncalibrated", "calibrated"}:
        score_map["candidate_reliability_model"] = (
            [float(rel_m.predict_proba_one(x) or 0.0) for x in X_ev],
            {"status": rel_m.calibration_status, "is_calibrated_probability": rel_m.calibration_status == "calibrated"},
        )
    else:
        score_map["candidate_reliability_model"] = ([], {"status": rel_m.calibration_status})

    if decomp_cond.calibration_status in {"fitted_uncalibrated", "calibrated"}:
        score_map["fixed_decomposed_ere_conditional"] = (
            [float(decomp_cond.predict(x)["repair_success"] or 0.0) for x in X_ev],
            {"status": decomp_cond.calibration_status, "is_calibrated_probability": decomp_cond.calibration_status == "calibrated"},
        )
    else:
        score_map["fixed_decomposed_ere_conditional"] = (
            [],
            {
                "status": decomp_cond.calibration_status,
                "write_fit_stats": decomp_cond.write_predictor.fit_stats,
            },
        )

    if decomp_oracle.calibration_status in {"fitted_uncalibrated", "calibrated"}:
        score_map["fixed_decomposed_ere_oracle_write"] = (
            [float(decomp_oracle.predict(x)["repair_success"] or 0.0) for x in X_ev],
            {"status": decomp_oracle.calibration_status, "is_calibrated_probability": decomp_oracle.calibration_status == "calibrated"},
        )
    else:
        score_map["fixed_decomposed_ere_oracle_write"] = (
            [],
            {
                "status": decomp_oracle.calibration_status,
                "write_fit_stats": decomp_oracle.write_predictor.fit_stats,
            },
        )

    if direct_cal_m is not None and direct_cal_m.calibration_status in {"fitted_uncalibrated", "calibrated"}:
        score_map["direct_success_independently_calibrated"] = (
            [float(direct_cal_m.predict_proba_one(x) or 0.0) for x in X_ev],
            {
                "status": direct_cal_m.calibration_status,
                "is_calibrated_probability": direct_cal_m.calibration_status == "calibrated",
                "fit_stats": direct_cal_m.fit_stats,
            },
        )

    # Baselines (raw ranking + train-only logistic probability mapped versions)
    for bname in ("candidate_change", "direct_confidence", "support_completeness", "entropy", "self_consistency", "random"):
        ev_sc = [baseline_score(bname, r["diagnostic_features"], CandidateReport(**r["reconstruction"])) for r in ev_att]
        score_map[bname] = (ev_sc, {"status": "ranking_baseline", "is_calibrated_probability": False})
        if bname in ("candidate_change", "direct_confidence") and len(set(ys_tr)) >= 2 and len(tr_att) >= 6:
            tr_sc = np.asarray(
                [[baseline_score(bname, r["diagnostic_features"], CandidateReport(**r["reconstruction"]))] for r in tr_att],
                dtype=np.float64,
            )
            lr_b = LogisticRegression(C=1.0, random_state=107, max_iter=1000).fit(tr_sc, ys_tr)
            ev_p = lr_b.predict_proba(np.asarray([[v] for v in ev_sc], dtype=np.float64))[
                :, list(lr_b.classes_).index(1)
            ].tolist()
            score_map[f"{bname}_train_prob_mapped"] = (
                [float(v) for v in ev_p],
                {"status": "fitted_uncalibrated", "is_calibrated_probability": False},
            )

    return {
        "evaluation_mode": label,
        "branch": branch,
        "train_n_attempted": len(tr_att),
        "train_positives_strict": int(sum(ys_tr)),
        "train_negatives_strict": int(len(ys_tr) - sum(ys_tr)),
        "train_conditional_write_pos": int(cond_pos),
        "train_conditional_write_neg": int(cond_neg),
        "train_oracle_write_pos": int(oracle_pos),
        "train_oracle_write_neg": int(oracle_neg),
        "eval_n_strict_eligible": len(ev_elig),
        "eval_n_attempted": len(ev_att),
        "eval_positives_strict": int(sum(ys_ev)),
        "eval_candidate_correct_count": int(sum(yc_ev)),
        "methods": _score_methods(ev_elig, ev_att, ys_ev, yc_ev, score_map),
    }


def _audit_m2_artifacts() -> Dict[str, object]:
    m2_dirs = [
        "results/v3-new/exp05c-m2-discovery-dev36",
        "results/v3-new/exp10-m2-discovery-cal36",
        "results/v3-new/exp11-m2-discovery-test48",
    ]
    out = {}
    for rel in m2_dirs:
        p = REPO_ROOT / rel
        if not p.exists():
            out[rel] = {"exists": False, "status": "MISSING"}
            continue
        files = sorted(f.name for f in p.iterdir())
        has_per_unit = any("per_unit" in f or "scores" in f for f in files)
        summary = json.loads((p / "m2_discovery_metrics.json").read_text(encoding="utf-8")) if (p / "m2_discovery_metrics.json").exists() else None
        out[rel] = {
            "exists": True,
            "files": files,
            "has_per_unit_raw_scores": has_per_unit,
            "full_pool_and_equal_budget_status": (
                "RECOMPUTABLE" if has_per_unit else "NOT_RECOMPUTABLE_FROM_AVAILABLE_ARTIFACTS"
            ),
            "historical_summary": summary,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="KRMM v0.3 CPU-Only Offline Scientific Audit")
    ap.add_argument("--dev", required=True)
    ap.add_argument("--cal", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    integrity = _collect_integrity()
    dev_recs = _load_unit_records(Path(args.dev))
    cal_recs = _load_unit_records(Path(args.cal))
    test_recs = _load_unit_records(Path(args.test))

    recomputed = {
        "dev": _audit_denominators_and_evidence_graph("dev", dev_recs),
        "calibration": _audit_denominators_and_evidence_graph("calibration", cal_recs),
        "test_post_hoc_audit": _audit_denominators_and_evidence_graph("test", test_recs),
        "m2_historical_artifacts": _audit_m2_artifacts(),
    }

    comparisons = {
        "dev_to_cal_structured": _compare_repairability_models(dev_recs, cal_recs, "structured", label="exploratory"),
        "dev_to_cal_generative": _compare_repairability_models(dev_recs, cal_recs, "generative", label="exploratory"),
        "cal_to_dev_structured": _compare_repairability_models(cal_recs, dev_recs, "structured", label="exploratory"),
        "cal_to_dev_generative": _compare_repairability_models(cal_recs, dev_recs, "generative", label="exploratory"),
        "dev_cal_to_test_structured_posthoc": _compare_repairability_models(
            dev_recs + cal_recs, test_recs, "structured", label="post_hoc_audit"
        ),
        "dev_cal_to_test_generative_posthoc": _compare_repairability_models(
            dev_recs + cal_recs, test_recs, "generative", label="post_hoc_audit"
        ),
        "dev_fit_cal_calibrated_to_test_generative_posthoc": _compare_repairability_models(
            dev_recs, test_recs, "generative", cal_recs=cal_recs, label="post_hoc_audit"
        ),
        "dev_fit_cal_calibrated_to_test_structured_posthoc": _compare_repairability_models(
            dev_recs, test_recs, "structured", cal_recs=cal_recs, label="post_hoc_audit"
        ),
    }

    summary = {
        "audit_mode": "CPU_ONLY_OFFLINE",
        "historical_artifacts_unchanged": True,
        "krmm_model_imported": False,
        "key_findings": {
            "write_strategy_actual_full_splits": "standard_kl (exp07/exp08/exp09); orthogonal_retain_proj only in exp06 (dev12)",
            "d2_actual_regularization": "KL retain_weight=50.0 + L2=1e-4 WERE active; fragility came from unbounded unlikelihood target loss",
            "m4_conditional_write_sparsity_dev_cal_structured": {
                "cond_pos": comparisons["dev_cal_to_test_structured_posthoc"]["train_conditional_write_pos"],
                "cond_neg": comparisons["dev_cal_to_test_structured_posthoc"]["train_conditional_write_neg"],
                "oracle_pos": comparisons["dev_cal_to_test_structured_posthoc"]["train_oracle_write_pos"],
                "oracle_neg": comparisons["dev_cal_to_test_structured_posthoc"]["train_oracle_write_neg"],
                "status": comparisons["dev_cal_to_test_structured_posthoc"]["methods"]["fixed_decomposed_ere_conditional"]["status"],
            },
            "m4_conditional_write_sparsity_dev_cal_generative": {
                "cond_pos": comparisons["dev_cal_to_test_generative_posthoc"]["train_conditional_write_pos"],
                "cond_neg": comparisons["dev_cal_to_test_generative_posthoc"]["train_conditional_write_neg"],
                "oracle_pos": comparisons["dev_cal_to_test_generative_posthoc"]["train_oracle_write_pos"],
                "oracle_neg": comparisons["dev_cal_to_test_generative_posthoc"]["train_oracle_write_neg"],
                "status": comparisons["dev_cal_to_test_generative_posthoc"]["methods"]["fixed_decomposed_ere_conditional"]["status"],
            },
            "m2_historical_recomputability": "NOT_RECOMPUTABLE_FROM_AVAILABLE_ARTIFACTS",
            "evidence_graph_candidate_diffs": {
                "dev": recomputed["dev"]["evidence_graph_audit"]["candidate_differences_orig_vs_corrected"],
                "cal": recomputed["calibration"]["evidence_graph_audit"]["candidate_differences_orig_vs_corrected"],
                "test": recomputed["test_post_hoc_audit"]["evidence_graph_audit"]["candidate_differences_orig_vs_corrected"],
            },
        },
        "pending_gpu_validation": [
            "orthogonal_retain_proj full dev/cal/test evaluation",
            "bounded/margin D2_suppression G0 validation",
            "M2 stratified randomization + post-injection behavioral truth + equal-budget curve + disjoint unknown controls",
            "M4 independent calibration & write-risk validation on higher-failure-rate write regime",
            "Model-level causal dependence intervention on EvidenceGraph premises",
        ],
    }

    (out_dir / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    (out_dir / "recomputed_metrics.json").write_text(json.dumps(recomputed, indent=2), encoding="utf-8")
    (out_dir / "baseline_comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
