"""Minimal sufficient support-set enumeration and 7-category v0.4 evidence classification."""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Set, Tuple

from krmm.v3.knowledge_graph import ALL_OPTIONS, SINGLETON_CONSTRAINTS, EvidenceGraph
from krmm.v3.schemas import AtomicFact


def enumerate_minimal_support_sets(
    domain: str,
    facts: Sequence[AtomicFact],
) -> Dict[str, object]:
    """Enumerate all minimal sufficient subsets of active facts without accessing evaluator truth."""
    valid_facts = [f for f in facts if f.value != "unparsed"]
    graph = EvidenceGraph(domain)
    full_eval = graph.evaluate(valid_facts)

    if full_eval["conflict"]:
        return {
            "candidate": None,
            "compatible_candidates": [],
            "conflict_status": True,
            "identifiable": False,
            "minimal_support_sets": [],
            "min_support_size": None,
            "full_eval": full_eval,
            "valid_facts": valid_facts,
        }

    compatible: List[str] = list(full_eval["compatible"])
    if len(compatible) != 1:
        return {
            "candidate": None,
            "compatible_candidates": compatible,
            "conflict_status": False,
            "identifiable": False,
            "minimal_support_sets": [],
            "min_support_size": None,
            "full_eval": full_eval,
            "valid_facts": valid_facts,
        }

    candidate = compatible[0]
    active_indices: List[int] = list(full_eval["active_indices"])
    minimal_tuples: List[Tuple[int, ...]] = []
    minimal_details: List[Dict[str, object]] = []

    max_k = min(len(active_indices), 10)
    for k in range(1, max_k + 1):
        for comb in itertools.combinations(active_indices, k):
            comb_set = set(comb)
            if any(set(prev).issubset(comb_set) for prev in minimal_tuples):
                continue
            sub_facts = [valid_facts[i] for i in comb]
            sub_eval = graph.evaluate(sub_facts)
            if not sub_eval["conflict"] and sub_eval["compatible"] == [candidate]:
                minimal_tuples.append(comb)
                minimal_details.append(
                    {
                        "indices": list(comb),
                        "size": len(comb),
                        "attributes": [valid_facts[i].attribute for i in comb],
                        "values": [valid_facts[i].value for i in comb],
                        "rules": [p["rule"] for p in sub_eval["support_paths"]],
                        "sources": sorted({valid_facts[i].source for i in comb}),
                        "provenance_groups": sorted({valid_facts[i].provenance_group for i in comb}),
                    }
                )

    min_size: Optional[int] = min((len(t) for t in minimal_tuples), default=None)
    return {
        "candidate": candidate,
        "compatible_candidates": compatible,
        "conflict_status": False,
        "identifiable": bool(minimal_tuples),
        "minimal_support_sets": minimal_details,
        "min_support_size": min_size,
        "full_eval": full_eval,
        "valid_facts": valid_facts,
    }
