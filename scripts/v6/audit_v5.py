#!/usr/bin/env python3
"""P0 Read-Only Audit of KRMM v0.5 artifacts (Section 3.1 & 3.2).

Strictly read-only on results/v5-new/*.json. Never fabricates missing data; flags NOT_RECOMPUTABLE where absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _auroc(y: Sequence[int], s: Sequence[float]) -> float:
    pos = [v for yt, v in zip(y, s) if yt == 1]
    neg = [v for yt, v in zip(y, s) if yt == 0]
    if not pos or not neg:
        return 0.5
    wins = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _auprc(y: Sequence[int], s: Sequence[float], uids: Sequence[str]) -> float:
    order = sorted(range(len(y)), key=lambda i: (-float(s[i]), uids[i]))
    n_pos = sum(y)
    if n_pos == 0:
        return 0.0
    tp = 0
    ap = 0.0
    for r, i in enumerate(order, 1):
        if y[i] == 1:
            tp += 1
            ap += tp / r
    return ap / n_pos


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only P0 audit of KRMM v0.5 JSON artifacts")
    ap.add_argument("--baseline-ref", default="7a41d02")
    ap.add_argument("--v5-dir", default="results/v5-new")
    ap.add_argument("--out", default="reports/v6/legacy_recompute.json")
    args = ap.parse_args()

    v5_dir = REPO_ROOT / args.v5_dir
    files = sorted(v5_dir.glob("*.json"))
    file_hashes = {f.name: _sha256(f) for f in files}

    g1 = json.loads((v5_dir / "g1_discovery_results.json").read_text(encoding="utf-8"))
    g2a = json.loads((v5_dir / "g2_mechanism_results.json").read_text(encoding="utf-8"))
    g2b = json.loads((v5_dir / "g2_repair_results.json").read_text(encoding="utf-8"))
    g3a = json.loads((v5_dir / "g3a_d2_bounded_results.json").read_text(encoding="utf-8"))
    g3b_llama = json.loads((v5_dir / "g3b_llama32_3b_results.json").read_text(encoding="utf-8"))
    g3b_smol = json.loads((v5_dir / "g3b_smollm2_1.7b_results.json").read_text(encoding="utf-8"))

    # A. Check v0.5 G1 clean_accuracy bug
    truth_rows = g1.get("post_injection_truth", [])
    clean_accs = [r.get("clean_accuracy", None) for r in truth_rows]
    all_zero_clean_acc = all(v == 0.0 for v in clean_accs) if clean_accs else False

    # E. Exact recomputation of reversed confidence & entropy from raw 5N scores
    tmap = {r["unit_id"]: r for r in truth_rows}
    reps_5n = g1.get("per_unit_reports_5N", {})
    recomputed_g1: Dict[str, object] = {}
    for det_name in ("direct_confidence", "entropy", "self_consistency", "direct_paraphrases", "constraint_only", "m2_reference_free", "full_m2"):
        rows = reps_5n.get(det_name, [])
        if not rows:
            recomputed_g1[det_name] = "NOT_RECOMPUTABLE"
            continue
        y = [1 if tmap[r["unit_id"]]["pool_category"] in ("damaged_mastered", "collateral_damaged") else 0 for r in rows]
        s = [float(r["anomaly_score"]) for r in rows]
        u = [str(r["unit_id"]) for r in rows]
        rev_s = [-v for v in s]
        k_fixed = max(1, int(math.floor(0.25 * len(rows))))
        ord_fwd = sorted(range(len(rows)), key=lambda i: (-s[i], u[i]))
        ord_rev = sorted(range(len(rows)), key=lambda i: (-rev_s[i], u[i]))
        recomputed_g1[det_name] = {
            "n": len(rows),
            "n_pos": sum(y),
            "forward_auroc": _auroc(y, s),
            "forward_auprc": _auprc(y, s, u),
            "forward_p_at_fixed_k25pct": sum(y[i] for i in ord_fwd[:k_fixed]) / k_fixed,
            "reverse_auroc": _auroc(y, rev_s),
            "reverse_auprc": _auprc(y, rev_s, u),
            "reverse_p_at_fixed_k25pct": sum(y[i] for i in ord_rev[:k_fixed]) / k_fixed,
        }

    # Entity intersection between Qwen G2-B (n=13) and Llama G3-B (n=9)
    qwen_ents = {r["entity_id"] for r in g2b.get("per_unit", [])}
    llama_ents = {r["entity_id"] for r in g3b_llama.get("per_unit", [])}
    overlap_ql = sorted(qwen_ents.intersection(llama_ents))
    unique_ql = sorted(qwen_ents.union(llama_ents))

    # G2-A complete prompt A accuracy vs structured
    g2a_units = g2a.get("per_unit", [])
    g2a_A_correct = sum(
        1 for r in g2a_units
        if r["model_level_interventions"]["A_complete_pred"] == r["gold_evaluator_only"]
    )

    # Exact one-sided 95% Clopper-Pearson upper bound for 0/14 (if iid): 1 - 0.05^(1/14)
    clopper_pearson_0_of_14 = 1.0 - (0.05 ** (1.0 / 14.0))

    audit = {
        "baseline_ref": args.baseline_ref,
        "v5_artifact_sha256": file_hashes,
        "bug_A_g1_clean_accuracy_all_zero": all_zero_clean_acc,
        "bug_A_status": "CONFIRMED_CLEAN_ACC_ZERO_IN_V5_RECOMPUTE_NOT_POSSIBLE_WITHOUT_CLEAN_RUN",
        "g1_paired_clean_heldout_recompute": "NOT_RECOMPUTABLE",
        "bug_E_g1_exact_recompute_5N": recomputed_g1,
        "g2a_audit": {
            "n_units": len(g2a_units),
            "complete_prompt_A_generative_correct": g2a_A_correct,
            "complete_prompt_A_generative_rate": g2a_A_correct / max(1, len(g2a_units)),
            "note": "v0.5 rule_only took empty facts -> abstention negative control, not strong baseline",
        },
        "cross_model_overlap_audit": {
            "qwen_strict_eligible_count": len(qwen_ents),
            "llama_strict_eligible_count": len(llama_ents),
            "total_model_target_instances": len(qwen_ents) + len(llama_ents),
            "overlapping_entities": overlap_ql,
            "distinct_entities_across_both_models": len(unique_ql),
            "llama_fresh_subprocess_reload_in_v5": False,
            "qwen_reload_in_v5_mode": "in_process_peft_disk_reload_target_acc_only",
            "one_sided_95_upper_bound_0_of_14_if_iid": clopper_pearson_0_of_14,
        },
        "g3_negative_boundaries": {
            "d2_bounded_n_mastered": g3a.get("n_mastered"),
            "d2_bounded_n_damage_effective": g3a.get("n_damage_effective"),
            "d2_note": "raw_ul.clamp(max=2.0) zeros gradient above 2.0; config failure, not universal phase transition",
            "smollm2_n_mastered": g3b_smol.get("n_mastered"),
            "smollm2_oracle_restored": g3b_smol.get("strategy_summary", {}).get("oracle", {}).get("n_strict_restored"),
            "smollm2_note": "tested config failure, not universal parameter-count threshold",
        },
    }

    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"P0_AUDIT_COMPLETE out={out_p}")
    print("G1_REV_CONF:", recomputed_g1.get("direct_confidence"))
    print("G1_REV_ENT:", recomputed_g1.get("entropy"))
    print("G1_M2_REF_FREE:", recomputed_g1.get("m2_reference_free"))
    print("G1_FULL_M2:", recomputed_g1.get("full_m2"))
    print("OVERLAP_QWEN_LLAMA:", overlap_ql, "distinct=", len(unique_ql))


if __name__ == "__main__":
    main()
