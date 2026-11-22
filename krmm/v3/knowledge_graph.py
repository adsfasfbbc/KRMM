"""Domain-general knowledge constraint graph for KRMM v0.3 (M2.3)."""
from __future__ import annotations

from typing import Dict, List, Set, Tuple
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
    """Evaluates support edges, logical constraints, conflicts, and provenance groups."""

    def __init__(self, domain: str) -> None:
        self.domain = domain

    def evaluate(self, facts: List[AtomicFact]) -> Dict[str, object]:
        compatible: Set[str] = set(ALL_OPTIONS)
        support_paths: List[Dict[str, object]] = []
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
            support_paths.append(
                {
                    "indices": [idx],
                    "rule": rule_name,
                    "attribute": fact.attribute,
                    "value": fact.value,
                    "allowed": sorted(allowed),
                    "source": fact.source,
                    "provenance_group": fact.provenance_group,
                }
            )

        # Joint compositional rule: flexible_reed + direct_air_flow -> B (aerophone)
        if self.domain == "sound_source_mechanism":
            reeds = [(i, f) for i, f in valid_facts if f.attribute == "oscillator" and f.value == "flexible_reed"]
            flows = [(i, f) for i, f in valid_facts if f.attribute == "excitation" and f.value == "direct_air_flow"]
            for i, r_fact in reeds:
                for j, f_fact in flows:
                    if r_fact.entity == f_fact.entity:
                        compatible.intersection_update({"B"})
                        support_paths.append(
                            {
                                "indices": [i, j],
                                "rule": "hs_air_driven_reed_joint",
                                "allowed": ["B"],
                                "provenance_groups": [r_fact.provenance_group, f_fact.provenance_group],
                            }
                        )

        # Check conflict (empty intersection when constraints fired)
        conflict = bool(support_paths) and len(compatible) == 0
        if conflict:
            conflicts.append(
                {
                    "reason": "empty_compatible_intersection",
                    "paths": support_paths,
                }
            )

        # Check provenance independence: count unique provenance groups among active constraints
        active_indices = {idx for p in support_paths for idx in p["indices"]}
        active_groups = {facts[i].provenance_group for i in active_indices}
        same_model_only = len(active_groups) <= 1 and len(active_indices) > 1

        return {
            "compatible": sorted(compatible),
            "support_paths": support_paths,
            "conflicts": conflicts,
            "conflict": conflict,
            "invalid_atoms": invalid_atoms,
            "unique_provenance_groups": sorted(active_groups),
            "same_model_correlated_only": same_model_only,
            "ambiguous": len(compatible) != 1,
        }
