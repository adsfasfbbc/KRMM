"""P0 deep re-audit of v0.3 evidence structure (27/27, 1/4) and 142/142 persistence accounting (CPU-only)."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from krmm.v3.knowledge_graph import EvidenceGraph
from krmm.v3.schemas import AtomicFact


def _load_records(run_dir: Path) -> List[dict]:
    skip = {"metrics.json", "run_meta.json", "damage-randomization-audit.json", "e0_e7_analysis.json"}
    recs = []
    for p in sorted(run_dir.glob("*.json")):
        if p.name in skip:
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "unit" in obj and "mastery" in obj:
            recs.append(obj)
    return recs


def classify_v4_evidence_structure(domain: str, facts: List[AtomicFact]) -> Dict[str, object]:
    """Classify atomic facts into the 7 v0.4 evidence structure categories without accessing gold labels."""
    valid_facts = [f for f in facts if f.value != "unparsed"]
    g = EvidenceGraph(domain)
    res = g.find_minimal_support_sets(valid_facts)
    full_eval = res["full_eval"]
    cand = res["candidate"]
    msets = res["minimal_support_sets"]
    min_size = res["min_support_size"]

    if full_eval["conflict"]:
        v4_class = "conflicting_support"
        overlap = 0.0
        shared_premises: List[str] = []
    elif cand is None or not msets:
        v4_class = "insufficient_support"
        overlap = 0.0
        shared_premises = []
    elif len(msets) == 1:
        v4_class = "single_decisive" if min_size == 1 else "single_chain"
        overlap = 0.0
        shared_premises = []
    else:
        # >= 2 minimal support sets exist; check provenance independence & premise overlap
        all_prov_groups: Set[str] = set()
        for ms in msets:
            all_prov_groups.update(ms["provenance_groups"])
        pair_overlaps = []
        shared_idx: Set[int] = set()
        for a, b in itertools.combinations(msets, 2):
            sa, sb = set(a["indices"]), set(b["indices"])
            inter = sa.intersection(sb)
            union = sa.union(sb)
            pair_overlaps.append(len(inter) / max(1, len(union)))
            shared_idx.update(inter)
        overlap = float(sum(pair_overlaps) / max(1, len(pair_overlaps)))
        shared_premises = [f"{valid_facts[i].attribute}={valid_facts[i].value}" for i in sorted(shared_idx)]

        # True independent_source_support requires >=2 distinct external/independent provenance groups
        if len(all_prov_groups) >= 2:
            v4_class = "independent_source_support"
        else:
            # All facts come from the SAME damaged model ("damaged_model_internal")!
            # Distinguish whether distinct semantic attributes corroborate or same attribute repeats
            attr_sets = [frozenset(ms["attributes"]) for ms in msets]
            distinct_semantic = len(set(attr_sets)) >= 2 and overlap < 1.0
            v4_class = "redundant_distinct_paths" if distinct_semantic else "redundant_same_source"

    prov_groups = sorted(full_eval["unique_provenance_groups"])
    return {
        "candidate": cand,
        "v3_explanation_class": res["explanation_class"],
        "v4_evidence_class": v4_class,
        "n_minimal_sets": len(msets),
        "min_support_size": min_size,
        "minimal_support_sets": msets,
        "dependency_overlap": overlap,
        "shared_premises": shared_premises,
        "provenance_groups": prov_groups,
        "same_model_only": bool(len(prov_groups) <= 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="results/v3-new/exp07-p2-dev36")
    ap.add_argument("--cal", default="results/v3-new/exp08-p2-cal36")
    ap.add_argument("--test", default="results/v3-new/exp09-p2-test48")
    ap.add_argument("--out", default="results/v4-new/v3_evidence_reaudit.json")
    args = ap.parse_args()

    splits = [
        ("dev", _load_records(REPO_ROOT / args.dev)),
        ("calibration", _load_records(REPO_ROOT / args.cal)),
        ("test", _load_records(REPO_ROOT / args.test)),
    ]

    all_rows = []
    by_v3_class: Dict[str, Dict[str, int]] = {}
    by_v4_class: Dict[str, Dict[str, int]] = {}
    entities_seen: List[str] = []
    persistence_breakdown: Dict[str, Dict[str, int]] = {}

    for sp_name, recs in splits:
        elig = [r for r in recs if r.get("strict_eligible")]
        sp_pers = {"oracle": 0, "structured": 0, "generative": 0, "direct": 0, "total_branch_reloads": 0, "unique_strict_eligible_entities": len(elig)}
        for r in elig:
            u = r["unit"]
            entities_seen.append(u["entity_id"])
            gold = r["gold_evaluator_only"]
            raw_facts = r["reconstruction"].get("atomic_facts", [])
            facts = [AtomicFact(**f) for f in raw_facts]
            cls = classify_v4_evidence_structure(u["domain"], facts)
            st_rep = (r.get("repairs") or {}).get("structured") or {}
            cand_ok = bool(cls["candidate"] == gold)
            abstained = cls["candidate"] is None
            strict_repaired = bool(st_rep.get("strict_restoration", False))

            # Persistence count audit across all 4 counterfactual branches
            for br in ("oracle", "structured", "generative", "direct"):
                br_rep = (r.get("repairs") or {}).get(br) or {}
                if br_rep.get("strict_restoration"):
                    fv = br_rep.get("fresh_process_verified")
                    if fv is True or (isinstance(fv, dict) and fv.get("passed")):
                        sp_pers[br] += 1
                        sp_pers["total_branch_reloads"] += 1

            row = {
                "split": sp_name,
                "unit_id": u["unit_id"],
                "entity_id": u["entity_id"],
                "domain": u["domain"],
                "source_group": u["source_group"],
                "candidate": cls["candidate"],
                "gold_evaluator_only": gold,
                "candidate_correct": cand_ok,
                "abstained": abstained,
                "structured_strict_repaired": strict_repaired,
                "v3_explanation_class": cls["v3_explanation_class"],
                "v4_evidence_class": cls["v4_evidence_class"],
                "n_minimal_sets": cls["n_minimal_sets"],
                "min_support_size": cls["min_support_size"],
                "dependency_overlap": cls["dependency_overlap"],
                "shared_premises": cls["shared_premises"],
                "provenance_groups": cls["provenance_groups"],
                "same_model_only": cls["same_model_only"],
                "paths_summary": [
                    f"({','.join(ms['attributes'])})" for ms in cls["minimal_support_sets"]
                ],
            }
            all_rows.append(row)

            for bucket, key in ((by_v3_class, cls["v3_explanation_class"]), (by_v4_class, cls["v4_evidence_class"])):
                b = bucket.setdefault(key, {"total": 0, "candidate_correct": 0, "abstained": 0, "strict_repaired": 0, "same_model_only": 0})
                b["total"] += 1
                b["candidate_correct"] += int(cand_ok)
                b["abstained"] += int(abstained)
                b["strict_repaired"] += int(strict_repaired)
                b["same_model_only"] += int(cls["same_model_only"])

        persistence_breakdown[sp_name] = sp_pers

    total_reloads = sum(v["total_branch_reloads"] for v in persistence_breakdown.values())
    out = {
        "n_strict_eligible_total": len(all_rows),
        "n_unique_entities": len(set(entities_seen)),
        "has_duplicate_entities_across_splits": len(set(entities_seen)) != len(entities_seen),
        "by_v3_explanation_class": by_v3_class,
        "by_v4_evidence_class": by_v4_class,
        "persistence_142_breakdown": {
            "per_split": persistence_breakdown,
            "total_branch_unit_reloads": total_reloads,
            "unique_strict_eligible_targets_total": len(all_rows),
            "unique_targets_restored_by_structured": sum(v["structured"] for v in persistence_breakdown.values()),
            "unique_targets_restored_by_oracle": sum(v["oracle"] for v in persistence_breakdown.values()),
            "clarification": (
                "142/142 counts (unit, counterfactual_branch) strict restorations verified on fresh reload "
                "across 4 branches (oracle=62, structured=46, generative=33, direct=1) over 66 unique targets; "
                "it is NOT 142 independent target entities."
            ),
        },
        "per_unit_rows": all_rows,
    }
    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_unit_rows"}, indent=2))
    print("\n=== SINGLE_CHAIN & REDUNDANT_DISTINCT_PATHS SAMPLE AUDIT ===")
    for r in all_rows:
        if r["v4_evidence_class"] in ("single_chain", "redundant_distinct_paths", "redundant_same_source"):
            print(
                f"{r['split']:11s} {r['unit_id']} {r['entity_id']:28s} v4={r['v4_evidence_class']:24s} "
                f"n_ms={r['n_minimal_sets']} ovlp={r['dependency_overlap']:.2f} prov={r['provenance_groups']} "
                f"ok={r['candidate_correct']} rep={r['structured_strict_repaired']} paths={r['paths_summary']}"
            )


if __name__ == "__main__":
    main()
