#!/usr/bin/env python3
"""Build & freeze the brand-new KRMM v0.6 pilot pool with strict Public vs Evaluator-Only separation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from krmm.v6.truth_audit import shuffle_public_pool
from krmm.v6.v6_catalog import (
    TWIN_WORLD_PAIRS,
    build_v6_pilot_catalog,
    verify_v6_disjoint_from_history,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build frozen v0.6 public & evaluator-only manifests")
    ap.add_argument("--config", default="configs/v6/pilot.json")
    ap.add_argument("--out", default="data/v6/pilot_manifest.json")
    args = ap.parse_args()

    cfg = json.loads((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 2026))

    disjoint = verify_v6_disjoint_from_history()
    assert disjoint["is_100_percent_disjoint"], f"Overlap with history: {disjoint['overlap_with_history']}"

    pub_units, eval_recs, _ = build_v6_pilot_catalog()
    shuffled_pub, pool_sha = shuffle_public_pool(pub_units, seed=seed)

    pub_manifest = {
        "version": "krmm-v0.6-pilot",
        "seed": seed,
        "pool_sha256": pool_sha,
        "n_units": len(shuffled_pub),
        "disjoint_audit": disjoint,
        "public_units": [u.to_public_dict() for u in shuffled_pub],
    }

    out_pub = REPO_ROOT / args.out
    out_pub.parent.mkdir(parents=True, exist_ok=True)
    pub_text = json.dumps(pub_manifest, indent=2, ensure_ascii=False)
    out_pub.write_text(pub_text, encoding="utf-8")
    pub_file_sha = hashlib.sha256(pub_text.encode("utf-8")).hexdigest()
    out_pub.with_suffix(".sha256").write_text(f"{pub_file_sha}  {out_pub.name}\n", encoding="utf-8")

    # Evaluator-only isolated directory
    eval_dir = out_pub.parent / "evaluator_only"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_manifest = {
        "version": "krmm-v0.6-pilot-evaluator-only",
        "public_manifest_sha256": pub_file_sha,
        "pool_sha256": pool_sha,
        "role_counts": dict(Counter(r.pool_role for r in eval_recs)),
        "domain_counts": dict(Counter(r.domain for r in eval_recs)),
        "evaluator_records": [r.to_dict() for r in eval_recs],
        "twin_world_negative_controls": TWIN_WORLD_PAIRS,
    }
    eval_path = eval_dir / "pilot_hidden.json"
    eval_path.write_text(json.dumps(eval_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"V6_POOL_BUILT n={len(shuffled_pub)} overlap=0 pool_sha={pool_sha[:12]} roles={eval_manifest['role_counts']}")


if __name__ == "__main__":
    main()
