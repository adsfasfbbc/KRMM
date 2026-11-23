"""Correlated evidence & provenance separation: redundant_same_model vs redundant_distinct_attributes vs independent_source."""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Set

from krmm.v3.schemas import AtomicFact


def assess_evidence_correlation(
    minimal_support_sets: Sequence[Dict[str, object]],
    valid_facts: Sequence[AtomicFact],
    conflict_status: bool,
    identifiable: bool,
    flagged_correlated_wrong: bool = False,
) -> Dict[str, object]:
    """Strictly separate same-model paraphrase repetition, same-model distinct attributes, and independent sources."""
    warnings: List[str] = []
    if conflict_status:
        return {
            "support_class": "conflicting_support",
            "dependency_overlap": 0.0,
            "shared_premises": [],
            "provenance_groups": sorted({f.provenance_group for f in valid_facts}),
            "same_model_only": len({f.provenance_group for f in valid_facts}) <= 1,
            "warnings": ["Conflicting atomic facts yield empty compatible candidate set."],
        }
    if not identifiable or not minimal_support_sets:
        return {
            "support_class": "insufficient_support",
            "dependency_overlap": 0.0,
            "shared_premises": [],
            "provenance_groups": sorted({f.provenance_group for f in valid_facts}),
            "same_model_only": len({f.provenance_group for f in valid_facts}) <= 1,
            "warnings": ["Remaining facts cannot uniquely identify a single candidate."],
        }

    active_idx: Set[int] = {idx for ms in minimal_support_sets for idx in ms["indices"]}
    prov_groups = sorted({valid_facts[i].provenance_group for i in active_idx})
    same_model_only = len(prov_groups) <= 1

    if flagged_correlated_wrong:
        warnings.append("Correlated wrong consensus detected/flagged in controlled stress test.")
        return {
            "support_class": "correlated_wrong_support",
            "dependency_overlap": 1.0,
            "shared_premises": [],
            "provenance_groups": prov_groups,
            "same_model_only": same_model_only,
            "warnings": warnings,
        }

    if len(minimal_support_sets) == 1:
        size = int(minimal_support_sets[0]["size"])
        s_class = "single_decisive" if size == 1 else "single_chain"
        if s_class == "single_chain":
            warnings.append("Single uncorroborated multi-premise chain; fragile to any premise error.")
        return {
            "support_class": s_class,
            "dependency_overlap": 0.0,
            "shared_premises": [],
            "provenance_groups": prov_groups,
            "same_model_only": same_model_only,
            "warnings": warnings,
        }

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

    attr_sigs = {frozenset(ms["attributes"]) for ms in minimal_support_sets}
    if not same_model_only:
        s_class = "independent_source_support"
    elif len(attr_sigs) >= 2 and overlap < 1.0:
        s_class = "redundant_distinct_attributes"
        warnings.append("Distinct semantic attributes from same model; NOT statistically independent sources.")
    else:
        s_class = "redundant_same_model"
        warnings.append("Repeated same-attribute paraphrases from same model; discounted as spurious redundancy.")

    return {
        "support_class": s_class,
        "dependency_overlap": overlap,
        "shared_premises": shared_premises,
        "provenance_groups": prov_groups,
        "same_model_only": same_model_only,
        "warnings": warnings,
    }
