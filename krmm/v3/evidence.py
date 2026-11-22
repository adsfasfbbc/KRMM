"""Atomic fact extraction via alternative probing paths (M3.1)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from .schemas import AtomicFact, KnowledgeUnit

# (domain, attribute, prompt_template)
ATTRIBUTE_PROBES: Dict[str, List[Tuple[str, str]]] = {
    "vertebrate_class": [
        ("thermoregulation", "Does a {entity} generate its own constant warm body temperature internally (endothermic), or rely on external environmental heat (ectothermic)? Answer with one word: ectothermic or endothermic."),
        ("respiration", "Does an adult {entity} breathe air using lungs or breathe underwater using gills? Reply briefly with 'lungs' or 'gills'."),
        ("reproduction", "How does a {entity} typically reproduce: live birth, hard-shelled eggs, or aquatic spawning eggs? Reply briefly."),
        ("covering", "What covers the outer skin of a {entity}: fur/hair, feathers, scales, or smooth bare skin? Reply briefly."),
    ],
    "everyday_material": [
        ("origin", "What primary substance is a standard {entity} made of (such as metal, wood/plant fiber, glass/ceramic/stone, or plastic/rubber)? Reply in a few words."),
        ("conductivity", "Is a standard {entity} an electrical conductor or an electrical insulator? Reply briefly."),
        ("heat_behavior", "Under moderate kitchen/household heat (around 250C), does a {entity} remain rigid without melting/charring, or does it char/melt/soften? Reply briefly."),
    ],
    "sound_source_mechanism": [
        ("oscillator", "What physical part of an acoustic {entity} vibrates to produce sound (stretched string, air column, flexible reed, stretched membrane, or solid body)? Reply briefly."),
        ("excitation", "How is vibration initiated in a {entity} (direct air flow, lip buzz, plucked, bowed, or struck)? Reply briefly."),
    ],
}


def _parse_attribute(domain: str, attribute: str, raw: str) -> Tuple[Optional[str], str]:
    t = raw.lower()
    if domain == "vertebrate_class":
        if attribute == "thermoregulation":
            if "ectotherm" in t or "cold-blooded" in t or "cold blooded" in t or "external" in t:
                return "cold_blooded", "kw_ectothermic"
            if "endotherm" in t or "warm-blooded" in t or "warm blooded" in t or "constant" in t:
                return "warm_blooded", "kw_endothermic"
        elif attribute == "respiration":
            if "gill" in t:
                return "gills", "kw_gills"
            if "lung" in t:
                return "lungs", "kw_lungs"
        elif attribute == "reproduction":
            if "live" in t or "vivipar" in t or "give birth" in t:
                return "live_birth", "kw_live_birth"
            if "aquatic" in t or "spawn" in t or "in water" in t:
                return "aquatic_eggs", "kw_aquatic_eggs"
            if "egg" in t:
                return "hard_shelled_eggs", "kw_eggs"
        elif attribute == "covering":
            if "feather" in t:
                return "feathers", "kw_feathers"
            if "fur" in t or "hair" in t:
                return "fur_or_hair", "kw_fur_hair"
            if "smooth" in t or "bare skin" in t or "skin" in t:
                return "smooth_skin", "kw_smooth_skin"
            if "scale" in t:
                return "scales", "kw_scales"
    elif domain == "everyday_material":
        if attribute == "origin":
            if any(k in t for k in ("wood", "bamboo", "plant", "cotton", "paper", "cardboard", "cellulose", "linen", "fiber")):
                return "biological_plant", "kw_plant"
            if any(k in t for k in ("plastic", "polymer", "pvc", "nylon", "silicone", "rubber", "polyethylene", "polyester", "synthetic", "elastomer")):
                return "synthetic_petro", "kw_synthetic"
            if any(k in t for k in ("metal", "steel", "iron", "copper", "aluminum", "brass", "silver", "gold", "titanium", "alloy", "glass", "porcelain", "ceramic", "clay", "stone", "granite", "inorganic", "mineral")):
                return "inorganic", "kw_inorganic"
        elif attribute == "conductivity":
            if "insulator" in t or "non-conductor" in t or "does not conduct" in t:
                return "insulator", "kw_insulator"
            if "conductor" in t or "conducts" in t:
                return "conductor", "kw_conductor"
        elif attribute == "heat_behavior":
            if "not melt" in t or "remain rigid" in t or "without melting" in t or "high melting" in t or "withstand" in t:
                return "high_melting", "kw_high_melting"
            if "char" in t or "burn" in t or "melt" in t or "soften" in t or "degrade" in t:
                return "low_heat_degrade", "kw_low_heat"
    elif domain == "sound_source_mechanism":
        if attribute == "oscillator":
            if "string" in t:
                return "stretched_string", "kw_string"
            if "membrane" in t or "drumhead" in t or "skin" in t:
                return "stretched_membrane", "kw_membrane"
            if "reed" in t:
                return "flexible_reed", "kw_reed"
            if "air column" in t or "column of air" in t:
                return "air_column", "kw_air_column"
            if "solid" in t or "bar" in t or "metal" in t or "wood" in t or "body" in t:
                return "solid_body", "kw_solid_body"
        elif attribute == "excitation":
            if "lip" in t:
                return "lip_buzz_air", "kw_lip"
            if "air" in t or "blow" in t or "breath" in t:
                return "direct_air_flow", "kw_air"
            if "struck" in t or "strike" in t or "hit" in t or "mallet" in t or "stick" in t:
                return "struck", "kw_struck"
            if "pluck" in t or "bow" in t:
                return "plucked_or_bowed", "kw_pluck_bow"
    return None, "unparsed"


def extract_atomic_facts(
    omni,
    unit: KnowledgeUnit,
    max_new_tokens: int = 32,
    extra_evidence: Optional[Dict[str, object]] = None,
) -> Tuple[List[AtomicFact], int]:
    """Probe the damaged model on non-target attributes and parse into AtomicFacts."""
    probes = ATTRIBUTE_PROBES.get(unit.domain, [])
    facts: List[AtomicFact] = []
    calls = 0
    for attr, tmpl in probes:
        prompt = tmpl.format(entity=unit.entity_id)
        raw = omni.generate(prompt, max_new_tokens=max_new_tokens)
        calls += 1
        val, rule = _parse_attribute(unit.domain, attr, raw)
        facts.append(
            AtomicFact(
                entity=unit.entity_id,
                attribute=attr,
                value=val if val is not None else "unparsed",
                source="damaged_model_probe",
                modality="text",
                confidence=0.8 if val is not None else 0.0,
                provenance_group="damaged_model_internal",
                raw_output=raw,
                matched_rule=rule,
            )
        )
    if extra_evidence and "facts" in extra_evidence:
        for ef in extra_evidence["facts"]:
            facts.append(ef)
    return facts, calls
