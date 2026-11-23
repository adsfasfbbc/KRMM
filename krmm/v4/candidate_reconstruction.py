"""Candidate reconstruction with Internal-Only vs Evidence-Augmented separation and shortcut baselines (P2)."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

from krmm.v3.schemas import AtomicFact, KnowledgeUnit
from .evidence_graph import EvidenceGraphAnalyzer
from .schemas import EvidenceAssessment, assert_method_boundary


def reconstruct_evidence_gated(
    domain: str,
    facts: Sequence[AtomicFact],
    study_condition: str = "internal_only",
    external_facts: Optional[Sequence[AtomicFact]] = None,
    min_repair_reliability: float = 0.65,
) -> EvidenceAssessment:
    """Run evidence-gated reconstruction cleanly separating internal-only from evidence-augmented inputs."""
    combined = list(facts)
    if study_condition == "evidence_augmented" and external_facts:
        combined.extend(external_facts)
    elif study_condition == "internal_only" and external_facts:
        raise ValueError("Internal-Only condition forbids passing external_facts")
    analyzer = EvidenceGraphAnalyzer(domain=domain, study_condition=study_condition)
    return analyzer.assess(combined, min_repair_reliability=min_repair_reliability)


def run_shortcut_baselines(
    unit: KnowledgeUnit,
    facts: Sequence[AtomicFact],
    all_domain_facts_pool: Sequence[Sequence[AtomicFact]],
    seed: int = 107,
    direct_probs: Optional[Sequence[float]] = None,
    meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Evaluate shortcut controls: rule_only, entity_only, shuffled_facts, and full_evidence_gated."""
    assert_method_boundary(meta)
    rng = random.Random(f"{seed}::{unit.unit_id}")
    analyzer = EvidenceGraphAnalyzer(domain=unit.domain, study_condition="internal_only")

    # 1. Rule-only (no model-extracted atomic facts -> must be unidentifiable!)
    rule_only = analyzer.assess([])

    # 2. Entity-only (direct model probability over entity + options without atomic facts)
    if direct_probs is not None:
        idx = max(range(len(direct_probs)), key=lambda i: direct_probs[i])
        entity_only_cand = "ABCD"[idx]
    else:
        entity_only_cand = None

    # 3. EvidenceGraph with shuffled facts from another unit in same domain
    donor = rng.choice(list(all_domain_facts_pool)) if all_domain_facts_pool else []
    shuffled_facts = [
        AtomicFact(
            entity=unit.entity_id,
            attribute=f.attribute,
            value=f.value,
            source="shuffled_control",
            modality=f.modality,
            confidence=f.confidence,
            provenance_group="shuffled_control",
        )
        for f in donor
    ]
    shuffled_res = analyzer.assess(shuffled_facts)

    # 4. Full evidence-gated
    full_res = analyzer.assess(facts)

    return {
        "rule_only": rule_only.to_dict(),
        "entity_only_candidate": entity_only_cand,
        "shuffled_facts": shuffled_res.to_dict(),
        "full_evidence_gated": full_res.to_dict(),
    }
