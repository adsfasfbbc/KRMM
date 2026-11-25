#!/usr/bin/env python3
"""KRMM v0.7 P0 Re-Audit of v0.6 (bdb628a): exact integrity check & Oracle per-fact failure forensics."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="P0 Read-Only Re-Audit of KRMM v0.6")
    ap.add_argument("--base", default="bdb628a")
    ap.add_argument("--out", default="reports/v7")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    full_sha = subprocess.check_output(["git", "rev-parse", args.base], cwd=REPO_ROOT, text=True).strip()
    parent_sha = subprocess.check_output(["git", "rev-parse", f"{args.base}^"], cwd=REPO_ROOT, text=True).strip()

    # Load v0.6 artifacts read-only
    q_g1 = json.loads((REPO_ROOT / "results/v6-new/qwen_pilot_g1_r2/metrics.json").read_text(encoding="utf-8"))
    q_e2e = json.loads((REPO_ROOT / "results/v6-new/qwen_pilot_e2e_r2/metrics.json").read_text(encoding="utf-8"))
    q_g2a = json.loads((REPO_ROOT / "results/v6-new/qwen_pilot_g2a_r3/metrics.json").read_text(encoding="utf-8"))
    l_g1 = json.loads((REPO_ROOT / "results/v6-new/llama32_pilot_g1/metrics.json").read_text(encoding="utf-8"))
    l_e2e = json.loads((REPO_ROOT / "results/v6-new/llama32_pilot_e2e/metrics.json").read_text(encoding="utf-8"))

    # Inspect EXACT same-entity regressions for Qwen & Llama Oracle and M_evidence_gated
    forensics: Dict[str, object] = {}
    for label, ep_dir in [
        ("qwen_Oracle_upper_bound", REPO_ROOT / "results/v6-new/qwen_pilot_e2e_r2/Oracle_upper_bound"),
        ("qwen_M_evidence_gated", REPO_ROOT / "results/v6-new/qwen_pilot_e2e_r2/M_evidence_gated"),
        ("qwen_M_full", REPO_ROOT / "results/v6-new/qwen_pilot_e2e_r2/M_full"),
        ("qwen_B1_generative", REPO_ROOT / "results/v6-new/qwen_pilot_e2e_r2/B1_generative"),
        ("llama_Oracle_upper_bound", REPO_ROOT / "results/v6-new/llama32_pilot_e2e/Oracle_upper_bound"),
        ("llama_M_evidence_gated", REPO_ROOT / "results/v6-new/llama32_pilot_e2e/M_evidence_gated"),
    ]:
        fresh = json.loads((ep_dir / "fresh_audit_result.json").read_text(encoding="utf-8"))
        actions = [
            json.loads(line)
            for line in (ep_dir / "method_actions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        la = fresh["locality_audit"]
        forensics[label] = {
            "n_actions": len(actions),
            "written_entities": [a["unit_id"] for a in actions if a["action"] == "ATTEMPT_REPAIR" and not a["rollback"]],
            "strict_locality_pass": la["strict_locality_pass"],
            "overall_token_agreement": la["overall_token_agreement"],
            "by_group": la["by_group"],
            "same_entity_regressions": la.get("same_entity_regressions", []),
        }

    # Map unit_id -> entity_id & domain from pilot_manifest.json
    pm = json.loads((REPO_ROOT / "data/v6/pilot_manifest.json").read_text(encoding="utf-8"))
    umap = {u["unit_id"]: u for u in pm["public_units"]}
    for k, v in forensics.items():
        v["written_entity_names"] = [
            f"{umap[uid]['entity_id']} ({umap[uid]['domain']})" if uid in umap else uid
            for uid in v["written_entities"]
        ]

    # Check if any regressed same-entity facts belong to WRITTEN entities vs UNWRITTEN entities!
    for k, v in forensics.items():
        written_names = {umap[uid]["entity_id"] for uid in v["written_entities"] if uid in umap}
        on_written = [r for r in v["same_entity_regressions"] if r["entity_id"] in written_names]
        on_unwritten = [r for r in v["same_entity_regressions"] if r["entity_id"] not in written_names]
        v["regressions_on_written_entities"] = on_written
        v["regressions_on_unwritten_entities"] = on_unwritten

    integrity = {
        "base_ref": args.base,
        "full_commit_sha": full_sha,
        "parent_commit_sha": parent_sha,
        "v5_frozen_parent_verified": parent_sha == "7a41d02a0b7cfa226d8ef7bc47e13d5402d44c1f",
        "pilot_manifest_sha256": _sha256(REPO_ROOT / "data/v6/pilot_manifest.json"),
        "qwen_g1_D": q_g1["denominator_D"],
        "llama_g1_D": l_g1["denominator_D"],
        "qwen_g2a_cand_gen": q_g2a["candidate_generation_comparison"],
        "forensics": forensics,
    }
    (out_dir / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    print("=== V0.7 P0 FORENSICS SUMMARY ===")
    print("FULL_SHA:", full_sha, "PARENT_OK:", integrity["v5_frozen_parent_verified"])
    for k, v in forensics.items():
        se = v["by_group"].get("same_entity_other_relation", {})
        print(
            f"\n[{k}] written={len(v['written_entities'])} se_c2w={se.get('correct_to_wrong')} "
            f"se_w2c={se.get('wrong_to_correct')} ag={se.get('token_agreement', 0):.3f}"
        )
        print("  written_names:", v["written_entity_names"])
        print("  on_WRITTEN_entities:", len(v["regressions_on_written_entities"]), [(r["entity_id"], r["question"].splitlines()[0]) for r in v["regressions_on_written_entities"]])
        print("  on_UNWRITTEN_entities:", len(v["regressions_on_unwritten_entities"]), [(r["entity_id"], r["question"].splitlines()[0]) for r in v["regressions_on_unwritten_entities"]])


if __name__ == "__main__":
    main()
