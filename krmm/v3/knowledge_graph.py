"""Domain-general knowledge constraint graph for KRMM v0.3 (M2.3)."""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Set, Tuple
from .schemas import AtomicFact

ALL_OPTIONS: Set[str] = {"A", "B", "C", "D"}

# (domain, attribute, value) -> (allowed_classes, rule_name)
SINGLETON_CONSTRAINTS: Dict[Tuple[str, str, str], Tuple[Set[str], str]] = {
    # Vertebrate class (A=mammal, B=bird, C=reptile, D=fish)
    ("vertebrate_class", "thermoregulation", "warm_blooded"): ({"A", "B"}, "endothermy_mammal_or_bird"),
    ("vertebrate_class", "thermoregulation", "cold_blooded"): ({"C", "D"}, "ectothermy_reptile_or_fish"),
    ("vertebrate_class", "respiration", "lungs"): ({"A", "B", "C"}, "pulmonary_breathing"),
    ("vertebrate_class", "respiration", "gills"): ({"D"}, "branchial_gills"),
    ("vertebrate_class", "reproduction", "live_birth"): ({"A", "C", "D"}, "viviparous_vertebrate"),
    ("vertebrate_class", "reproduction", "hard_shelled_eggs"): ({"B", "C"}, "amniote_shelled_eggs"),
    ("vertebrate_class", "reproduction", "aquatic_eggs"): ({"D"}, "aquatic_spawning_eggs"),
    ("vertebrate_class", "covering", "fur_or_hair"): ({"A"}, "mammalian_hair"),
    ("vertebrate_class", "covering", "smooth_skin"): ({"A"}, "mammalian_smooth_skin"),
    ("vertebrate_class", "covering", "feathers"): ({"B"}, "avian_feathers"),
    ("vertebrate_class", "covering", "scales"): ({"C", "D"}, "scaled_integument"),
    ("vertebrate_class", "appendage", "fins"): ({"A", "D"}, "aquatic_fins_or_flippers"),

    # Everyday material (A=metal, B=wood/plant, C=stone/ceramic/glass, D=polymer/rubber)
    ("everyday_material", "origin", "inorganic"): ({"A", "C"}, "inorganic_mineral_or_metal"),
    ("everyday_material", "origin", "organic_or_polymer"): ({"B", "D"}, "carbon_organic_or_polymer"),
    ("everyday_material", "origin", "biological_plant"): ({"B"}, "plant_cellulose_origin"),
    ("everyday_material", "origin", "synthetic_petro"): ({"D"}, "synthetic_polymer_origin"),
    ("everyday_material", "conductivity", "conductor"): ({"A"}, "metallic_electrical_conductor"),
    ("everyday_material", "conductivity", "insulator"): ({"B", "C", "D"}, "non_metallic_insulator"),
    ("everyday_material", "heat_behavior", "high_melting"): ({"A", "C", "D"}, "heat_resistant_inorganic_or_elastomer"),
    ("everyday_material", "heat_behavior", "low_heat_degrade"): ({"B", "D"}, "chars_or_softens_low_heat"),

    # Sound source mechanism (A=chordophone, B=aerophone, C=membranophone, D=idiophone)
    ("sound_source_mechanism", "oscillator", "stretched_string"): ({"A"}, "hs_stretched_string"),
    ("sound_source_mechanism", "oscillator", "air_column"): ({"B"}, "hs_air_column"),
    ("sound_source_mechanism", "oscillator", "stretched_membrane"): ({"C"}, "hs_stretched_membrane"),
    ("sound_source_mechanism", "oscillator", "solid_body"): ({"D"}, "hs_solid_body"),
    ("sound_source_mechanism", "oscillator", "flexible_reed"): ({"B", "D"}, "hs_reed_ambiguous"),
    ("sound_source_mechanism", "excitation", "direct_air_flow"): ({"B", "C", "D"}, "aerodynamic_or_air_excitation"),
    ("sound_source_mechanism", "excitation", "lip_buzz_air"): ({"B"}, "brass_lip_air"),
    ("sound_source_mechanism", "excitation", "struck"): ({"A", "C", "D"}, "percussive_or_hammered"),
    ("sound_source_mechanism", "excitation", "plucked_or_bowed"): ({"A", "D"}, "plucked_or_bowed"),
}


class EvidenceGraph:
    """Evaluates support edges, logical constraints, minimal support sets, conflicts, and provenance groups."""

    def __init__(self, domain: str) -> None:
        self.domain = domain

    def evaluate(self, facts: List[AtomicFact]) -> Dict[str, object]:
        compatible: Set[str] = set(ALL_OPTIONS)
        support_paths: List[Dict[str, object]] = []
        rule_trace: List[Dict[str, object]] = []
        conflicts: List[Dict[str, object]] = []
        invalid_atoms: List[Dict[str, object]] = []
        valid_facts: List[Tuple[int, AtomicFact]] = []

        for idx, fact in enumerate(facts):
            key = (self.domain, fact.attribute, fact.value)
            if key not in SINGLETON_CONSTRAINTS:
                invalid_atoms.append({"index": idx, "attribute": fact.attribute, "value": fact.value})
                continue
            allowed, rule_name = SINGLETON_CONSTRAINTS[key]
            valid_facts.append((idx, fact))
            compatible.intersection_update(allowed)
            path = {
                "indices": [idx],
                "rule": rule_name,
                "attribute": fact.attribute,
                "value": fact.value,
                "allowed": sorted(allowed),
                "source": fact.source,
                "provenance_group": fact.provenance_group,
            }
            support_paths.append(path)
            rule_trace.append({**path, "compatible_after": sorted(compatible)})

        # Joint compositional rule: flexible_reed + direct_air_flow -> B (aerophone)
        if self.domain == "sound_source_mechanism":
            reeds = [(i, f) for i, f in valid_facts if f.attribute == "oscillator" and f.value == "flexible_reed"]
            flows = [(i, f) for i, f in valid_facts if f.attribute == "excitation" and f.value == "direct_air_flow"]
            for i, r_fact in reeds:
                for j, f_fact in flows:
                    if r_fact.entity == f_fact.entity:
                        compatible.intersection_update({"B"})
                        joint_path = {
                            "indices": [i, j],
                            "rule": "hs_air_driven_reed_joint",
                            "allowed": ["B"],
                            "provenance_groups": [r_fact.provenance_group, f_fact.provenance_group],
                        }
                        support_paths.append(joint_path)
                        rule_trace.append({**joint_path, "compatible_after": sorted(compatible)})

        conflict = bool(support_paths) and len(compatible) == 0
        if conflict:
            conflicts.append(
                {
                    "reason": "empty_compatible_intersection",
                    "paths": support_paths,
                }
            )

        active_indices = {idx for p in support_paths for idx in p["indices"]}
        active_groups = {facts[i].provenance_group for i in active_indices}
        same_model_only = len(active_groups) <= 1 and len(active_indices) > 1

        return {
            "compatible": sorted(compatible),
            "support_paths": support_paths,
            "rule_trace": rule_trace,
            "conflicts": conflicts,
            "conflict": conflict,
            "invalid_atoms": invalid_atoms,
            "active_indices": sorted(active_indices),
            "unique_provenance_groups": sorted(active_groups),
            "same_model_correlated_only": same_model_only,
            "ambiguous": len(compatible) != 1,
        }

    def find_minimal_support_sets(self, facts: List[AtomicFact]) -> Dict[str, object]:
        """Enumerate minimal sufficient subsets of active facts determining a unique candidate."""
        full_eval = self.evaluate(facts)
        if full_eval["conflict"]:
            return {
                "candidate": None,
                "explanation_class": "conflicting_evidence",
                "minimal_support_sets": [],
                "min_support_size": None,
                "full_eval": full_eval,
            }

        compatible = full_eval["compatible"]
        if len(compatible) != 1:
            return {
                "candidate": None,
                "explanation_class": "insufficient_evidence",
                "minimal_support_sets": [],
                "min_support_size": None,
                "full_eval": full_eval,
            }

        candidate = compatible[0]
        active_indices: List[int] = list(full_eval["active_indices"])
        minimal_tuples: List[Tuple[int, ...]] = []
        minimal_details: List[Dict[str, object]] = []

        # Enumerate subsets in increasing cardinality; skip supersets of already found minimal sets
        max_k = min(len(active_indices), 10)
        for k in range(1, max_k + 1):
            for comb in itertools.combinations(active_indices, k):
                comb_set = set(comb)
                if any(set(prev).issubset(comb_set) for prev in minimal_tuples):
                    continue
                sub_facts = [facts[i] for i in comb]
                sub_eval = self.evaluate(sub_facts)
                if not sub_eval["conflict"] and sub_eval["compatible"] == [candidate]:
                    minimal_tuples.append(comb)
                    minimal_details.append(
                        {
                            "indices": list(comb),
                            "size": len(comb),
                            "attributes": [facts[i].attribute for i in comb],
                            "values": [facts[i].value for i in comb],
                            "rules": [p["rule"] for p in sub_eval["support_paths"]],
                            "provenance_groups": sorted({facts[i].provenance_group for i in comb}),
                        }
                    )

        if not minimal_tuples:
            return {
                "candidate": candidate,
                "explanation_class": "insufficient_evidence",
                "minimal_support_sets": [],
                "min_support_size": None,
                "full_eval": full_eval,
            }

        min_size: Optional[int] = min(len(t) for t in minimal_tuples)
        if len(minimal_tuples) >= 2:
            explanation_class = "redundant_support"
        elif min_size == 1:
            explanation_class = "single_hop"
        else:
            explanation_class = "minimal_multi_premise"

        return {
            "candidate": candidate,
            "explanation_class": explanation_class,
            "minimal_support_sets": minimal_details,
            "min_support_size": min_size,
            "all_minimal_require_multi_premise": bool(min_size is not None and min_size >= 2),
            "full_eval": full_eval,
        }
