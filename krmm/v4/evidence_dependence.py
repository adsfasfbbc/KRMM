"""Evidence path dependence, provenance-group independence, and correlated-support analysis."""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Set

from krmm.v3.schemas import AtomicFact


def analyze_evidence_dependence(
    minimal_support_sets: Sequence[Dict[str, object]],
    valid_facts: Sequence[AtomicFact],
    conflict_status: bool,
    identifiable: bool,
) -> Dict[str, object]:
    """Compute pairwise premise overlap, provenance independence, and the 7-way v0.4 evidence category."""
    if conflict_status:
        return {
            "evidence_category": "conflicting_support",
            "dependency_overlap": 0.0,
            "shared_premises": [],
            "provenance_groups": sorted({f.provenance_group for f in valid_facts}),
            "distinct_semantic_paths": 0,
            "has_independent_sources": False,
        }

    if not identifiable or not minimal_support_sets:
        return {
            "evidence_category": "insufficient_support",
            "dependency_overlap": 0.0,
            "shared_premises": [],
            "provenance_groups": sorted({f.provenance_group for f in valid_facts}),
            "distinct_semantic_paths": 0,
            "has_independent_sources": False,
        }

    active_idx: Set[int] = {idx for ms in minimal_support_sets for idx in ms["indices"]}
    prov_groups = sorted({valid_facts[i].provenance_group for i in active_idx})
    has_independent_sources = len(prov_groups) >= 2

    if len(minimal_support_sets) == 1:
        size = int(minimal_support_sets[0]["size"])
        cat = "single_decisive" if size == 1 else "single_chain"
        return {
            "evidence_category": cat,
            "dependency_overlap": 0.0,
            "shared_premises": [],
            "provenance_groups": prov_groups,
            "distinct_semantic_paths": 1,
            "has_independent_sources": has_independent_sources,
        }

    # Multiple minimal support sets: compute Jaccard overlap of premise indices and semantic attributes
    pair_overlaps: List[float] = []
    shared_idx: Set[int] = set()
    for a, b in itertools.combinations(minimal_support_sets, 2):
        sa, sb = set(a["indices"]), set(b["indices"])
        inter = sa.intersection(sb)
        union = sa.union(sb)
        pair_overlaps.append(len(inter) / max(1, len(union)))
        shared_idx.update(inter)
    overlap = float(sum(pair_overlaps) / max(1, len(pair_overlaps)))
    shared_premises = [f"{valid_facts[i].attribute}={valid_facts[i].value}" for i in sorted(shared_idx)]

    attr_signatures = {frozenset(ms["attributes"]) for ms in minimal_support_sets}
    distinct_semantic = len(attr_signatures)

    if has_independent_sources:
        cat = "independent_source_support"
    elif distinct_semantic >= 2 and overlap < 1.0:
        # Distinct semantic attributes corroborated, but still from the same model/source!
        cat = "redundant_distinct_paths"
    else:
        # Repeated attribute / paraphrased same-source duplication -> spurious redundancy
        cat = "redundant_same_source"

    return {
        "evidence_category": cat,
        "dependency_overlap": overlap,
        "shared_premises": shared_premises,
        "provenance_groups": prov_groups,
        "distinct_semantic_paths": distinct_semantic,
        "has_independent_sources": has_independent_sources,
    }
