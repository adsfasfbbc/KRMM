"""Build entity-disjoint blind mixed knowledge pool manifest for v0.4 (CPU-only, no model weights)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from krmm.v3.knowledge_registry import all_evaluator_units
from krmm.v4.damage_truth_audit import stratified_damage_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="calibration", choices=["dev", "calibration", "test"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results/v4-new/blind_pool_manifest.json")
    args = ap.parse_args()

    evs = all_evaluator_units(split=args.split)
    unks = all_evaluator_units(split="unknown_control")
    dmg, hlt, manifest = stratified_damage_split(evs, damage_fraction=0.5, seed=args.seed, stratify_by_gold=True)
    payload = {
        "split": args.split,
        "seed": args.seed,
        "n_known_pool": len(evs),
        "n_unknown_pool": len(unks),
        "warning_unknown_controls": "Next GPU protocol requires fresh disjoint unknown_control units per split.",
        "stratified_manifest": manifest,
        "unknown_unit_ids": [u.unit.unit_id for u in unks],
    }
    payload["payload_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_p), "manifest_sha256": manifest["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
