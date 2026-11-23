"""Residual knowledge reconstruction and sufficiency diagnostics (M3)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from .evidence import extract_atomic_facts
from .knowledge_graph import EvidenceGraph
from .knowledge_registry import question_prompt
from .schemas import AtomicFact, CandidateReport, KnowledgeUnit


def _choose(probs: List[float]) -> str:
    return "ABCD"[int(np.argmax(probs))]


def reconstruct_structured(domain: str, facts: List[AtomicFact]) -> Dict[str, object]:
    """Evaluate structured graph over atomic facts using minimal support-set analysis."""
    valid_facts = [f for f in facts if f.value != "unparsed"]
    graph = EvidenceGraph(domain)
    min_res = graph.find_minimal_support_sets(valid_facts)
    eval_res = min_res["full_eval"]
    candidate = min_res["candidate"]
    explanation_class = str(min_res["explanation_class"])
    min_support_size = min_res["min_support_size"]

    # True multi-premise composition requires that NO singleton subset uniquely determines candidate
    true_composition = bool(
        candidate is not None
        and min_support_size is not None
        and int(min_support_size) >= 2
    )
    source_type = "insufficient_or_conflicting"
    if candidate is not None:
        source_type = "structured_composition" if true_composition else "structured_single_hop"

    return {
        "candidate": candidate,
        "source_type": source_type,
        "explanation_class": explanation_class,
        "minimal_support_sets": min_res["minimal_support_sets"],
        "min_support_size": min_support_size,
        "true_composition": true_composition,
        "graph": eval_res,
        "rule_trace": eval_res["rule_trace"],
        "valid_facts": valid_facts,
    }


def reconstruct(
    omni,
    unit: KnowledgeUnit,
    allowed_evidence: Optional[Dict[str, object]] = None,
    budget: int = 24,
    max_new_tokens: int = 32,
) -> Tuple[CandidateReport, Dict[str, object]]:
    """Run M3 reconstruction: direct residual, structured composition, and generative reconstruction."""
    calls = 0

    direct_probs: List[List[float]] = []
    for v in (2, 3):
        p, _, _ = question_prompt(unit, variant=v)
        probs, _ = omni.probabilities(p)
        direct_probs.append(probs)
        calls += 1
    direct_candidate = _choose(direct_probs[0])

    facts, probe_calls = extract_atomic_facts(
        omni, unit, max_new_tokens=max_new_tokens, extra_evidence=allowed_evidence
    )
    calls += probe_calls

    struct_res = reconstruct_structured(unit.domain, facts)
    structured_candidate = struct_res["candidate"]
    valid_facts: List[AtomicFact] = struct_res["valid_facts"]

    obs_lines = [f"- {f.attribute}: {f.raw_output or f.value}" for f in facts]
    opt_lines = [f"{k}: {v}" for k, v in unit.options.items()]
    gen_prompt = (
        "Infer the correct category using ONLY these observations:\n"
        + "\n".join(obs_lines)
        + "\n"
        + "\n".join(opt_lines)
        + "\nReply with one letter only."
    )
    gen_probs, _ = omni.probabilities(gen_prompt)
    calls += 1
    generative_candidate = _choose(gen_probs)

    # Rule-level sufficiency diagnostics: key-premise ablation & sham-premise stability
    key_drops: List[Dict[str, object]] = []
    for i in range(len(valid_facts)):
        reduced = valid_facts[:i] + valid_facts[i + 1 :]
        r_res = reconstruct_structured(unit.domain, reduced)
        changed = r_res["candidate"] != structured_candidate
        key_drops.append(
            {
                "removed_attribute": valid_facts[i].attribute,
                "candidate_after": r_res["candidate"],
                "changed": changed,
            }
        )
    key_sensitive = any(k["changed"] for k in key_drops)

    sham_fact = AtomicFact(
        entity=unit.entity_id,
        attribute="unrelated_container_color",
        value="brown",
        source="sham_control",
        modality="text",
        confidence=1.0,
        provenance_group="sham",
        raw_output="The storage container is brown.",
    )
    sham_res = reconstruct_structured(unit.domain, valid_facts + [sham_fact])
    sham_stable = sham_res["candidate"] == structured_candidate

    if calls > budget:
        raise RuntimeError(f"Reconstruction exceeded probe budget {budget}: used {calls}")

    graph_info = struct_res["graph"]
    abstain_reason = None
    if graph_info["conflict"]:
        abstain_reason = "conflicting_evidence"
    elif structured_candidate is None:
        abstain_reason = "insufficient_or_ambiguous_evidence"

    ambiguity = max(0.0, (len(graph_info["compatible"]) - 1) / 3.0)
    report = CandidateReport(
        unit_id=unit.unit_id,
        candidate_answer=structured_candidate,
        candidate_source_type=str(struct_res["source_type"]),
        atomic_facts=[f.to_dict() for f in facts],
        support_paths=list(graph_info["support_paths"]),
        ambiguity=float(ambiguity),
        conflict=bool(graph_info["conflict"]),
        key_intervention_result={"sensitive": key_sensitive, "ablations": key_drops},
        sham_intervention_result={"stable": sham_stable, "candidate_with_sham": sham_res["candidate"]},
        abstain_reason=abstain_reason,
        query_cost=calls,
    )
    aux = {
        "direct_probs": direct_probs,
        "direct_candidate": direct_candidate,
        "structured_candidate": structured_candidate,
        "generative_probs": gen_probs,
        "generative_candidate": generative_candidate,
        "true_composition": struct_res["true_composition"],
        "explanation_class": struct_res["explanation_class"],
        "minimal_support_sets": struct_res["minimal_support_sets"],
        "min_support_size": struct_res["min_support_size"],
        "unique_provenance_groups": graph_info["unique_provenance_groups"],
        "same_model_correlated_only": graph_info["same_model_correlated_only"],
        "candidate_change_vs_direct": int(
            structured_candidate is not None and structured_candidate != direct_candidate
        ),
    }
    return report, aux
