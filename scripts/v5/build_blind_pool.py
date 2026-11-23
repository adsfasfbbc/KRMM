#!/usr/bin/env python3
"""Export and audit the brand-new KRMM v0.5 blind evaluation pool."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from krmm.v3.knowledge_registry import all_evaluator_units
from krmm.v5.v5_catalog import (
    all_v5_entities,
    v5_ambiguous_evaluator_units,
    v5_known_evaluator_units,
    v5_unknown_evaluator_units,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build & audit KRMM v0.5 blind pool")
    parser.add_argument("--out", default="results/v5-new/v5_blind_pool_manifest.json")
    args = parser.parse_args()

    v3_entities = {u.unit.entity_id.lower() for u in all_evaluator_units()}
    known = v5_known_evaluator_units()
    unknown = v5_unknown_evaluator_units()
    ambiguous = v5_ambiguous_evaluator_units()
    v5_entities = {e.lower() for e in all_v5_entities()}

    overlap = sorted(v3_entities.intersection(v5_entities))
    assert not overlap, f"v0.5 pool overlaps v0.3 entities: {overlap}"

    manifest = {
        "version": "krmm-v0.5",
        "counts": {
            "known": len(known),
            "unknown_control": len(unknown),
            "ambiguous": len(ambiguous),
            "total": len(known) + len(unknown) + len(ambiguous),
        },
        "v3_entity_count": len(v3_entities),
        "v3_overlap_count": len(overlap),
        "known_by_domain": dict(Counter(u.unit.domain for u in known)),
        "unknown_by_domain": dict(Counter(u.unit.domain for u in unknown)),
        "ambiguous_by_domain": dict(Counter(u.unit.domain for u in ambiguous)),
        "entities": {
            "known": [
                {"unit_id": u.unit.unit_id, "entity": u.unit.entity_id, "domain": u.unit.domain}
                for u in known
            ],
            "unknown_control": [
                {"unit_id": u.unit.unit_id, "entity": u.unit.entity_id, "domain": u.unit.domain}
                for u in unknown
            ],
            "ambiguous": [
                {"unit_id": u.unit.unit_id, "entity": u.unit.entity_id, "domain": u.unit.domain}
                for u in ambiguous
            ],
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote v0.5 blind pool manifest to {out_path}: {manifest['counts']}, overlap={len(overlap)}")


if __name__ == "__main__":
    main()
