"""Strictly public schema-derived relation retain & monitor prompt generator (ZERO overlap with hidden audit!).

Separates three prompt namespaces explicitly (tested by test_train_monitor_audit_prompt_separation):
1. TRAIN_RETAIN_RELATION: used inside W2/W4 KL loss to prevent same-entity & sibling-domain letter-bias bleed.
2. PUBLIC_MONITOR_RELATION: used inside transaction verify_public() to detect regressions before commit.
3. HIDDEN_AUDIT: evaluator-only, NEVER accessible here.
"""
from __future__ import annotations

from typing import Dict, List, Sequence
from krmm.v6.schemas import PublicUnit

# Public domain-general non-target property templates (strictly distinct from hidden audit questions!)
# Includes balanced A/B/C/D answer-position diversity so KL retain directly cancels letter-logit bias!
_TRAIN_RELATION_TEMPLATES: Dict[str, List[str]] = {
    "vertebrate_class": [
        "Consider the organism '{ent}'. Is '{ent}' a living biological animal rather than a machine or mineral?\nA: Yes, a living animal\nB: No, an electronic device\nC: No, a synthetic plastic\nD: No, an inorganic rock\nReply with one letter only.",
        "Regarding the geographic or ecological presence of '{ent}', which statement is most plausible?\nA: It exists only inside computer code\nB: It is a real biological species found on Earth\nC: It is a chemical solvent\nD: It is a kitchen appliance\nReply with one letter only.",
        "Does '{ent}' possess a backbone (vertebral column) as a multicellular organism?\nA: No, it is a single-celled bacterium\nB: No, it is a flowering plant\nC: Yes, it is a vertebrate animal\nD: No, it is a metal tool\nReply with one letter only.",
        "In everyday human knowledge, how is '{ent}' primarily encountered or studied?\nA: In semiconductor fabrication\nB: In architecture and masonry\nC: In metallurgy\nD: In zoology, ecology, or wildlife biology\nReply with one letter only.",
    ],
    "everyday_material": [
        "Consider the object '{ent}'. Is '{ent}' an inanimate physical item rather than a living animal?\nA: Yes, an inanimate object\nB: No, a wild bird\nC: No, a marine fish\nD: No, a venomous snake\nReply with one letter only.",
        "In which everyday context is '{ent}' typically encountered?\nA: Deep-sea volcanic vents\nB: Household, workshop, or daily human utility\nC: Wild forest canopies as a predator\nD: Subatomic particle accelerators\nReply with one letter only.",
        "Is '{ent}' a human-made or crafted physical artifact?\nA: No, a wild mammal\nB: No, a freshwater fish\nC: Yes, a physical artifact or utensil\nD: No, a songbird\nReply with one letter only.",
        "Does '{ent}' have a tangible solid form used by people?\nA: No, it is an abstract mathematical theorem\nB: No, it is a cloud formation\nC: No, it is a living reptile\nD: Yes, it is a tangible physical object\nReply with one letter only.",
    ],
    "sound_source_mechanism": [
        "Consider the item '{ent}'. Is '{ent}' a musical instrument used to produce sound?\nA: Yes, a musical instrument\nB: No, a predatory mammal\nC: No, a kitchen utensil\nD: No, a building foundation\nReply with one letter only.",
        "In which cultural domain is '{ent}' primarily used?\nA: Deep-sea oil drilling\nB: Music performance, rhythm, or acoustic sound making\nC: Reptile taxonomy\nD: Soil fertilization\nReply with one letter only.",
        "Does a performer interact physically with '{ent}' to produce audible vibrations?\nA: No, it is a wild fish\nB: No, it is an inert rock\nC: Yes, it produces audible musical sound\nD: No, it is a mammal\nReply with one letter only.",
        "Is '{ent}' studied within organology and musicology?\nA: No, marine biology\nB: No, civil engineering\nC: No, organic chemistry\nD: Yes, music and acoustic instruments\nReply with one letter only.",
    ],
}

# Strictly distinct templates for PUBLIC_MONITOR (never shared with TRAIN_RETAIN_RELATION or HIDDEN_AUDIT!)
_MONITOR_RELATION_TEMPLATES: Dict[str, List[str]] = {
    "vertebrate_class": [
        "Public stability check — Is '{ent}' a biological organism studied in biology?\nA: Yes\nB: No, a metal tool\nC: No, a musical instrument\nD: No, a plastic bin\nReply with one letter only.",
        "Public stability check — Does '{ent}' belong to the animal kingdom rather than inanimate objects?\nA: No, inanimate stone\nB: Yes, an animal\nC: No, a wooden box\nD: No, a musical drum\nReply with one letter only.",
    ],
    "everyday_material": [
        "Public stability check — Is '{ent}' a non-living everyday physical object?\nA: Yes\nB: No, a bird\nC: No, a fish\nD: No, a reptile\nReply with one letter only.",
        "Public stability check — Would '{ent}' be found in a home, kitchen, or workshop rather than a zoo?\nA: No, a zoo animal\nB: Yes, a physical object\nC: No, a wild fish\nD: No, a wild bird\nReply with one letter only.",
    ],
    "sound_source_mechanism": [
        "Public stability check — Is '{ent}' used by musicians to create sound or music?\nA: Yes\nB: No, a wild animal\nC: No, a food dish\nD: No, a rock\nReply with one letter only.",
        "Public stability check — Does '{ent}' belong to the family of musical instruments?\nA: No, a vertebrate animal\nB: Yes, a musical instrument\nC: No, a plastic container\nD: No, a tree\nReply with one letter only.",
    ],
}

# Extra generic prompts for W1 (generic-count matched control, strictly non-entity-specific!)
EXTRA_GENERIC_RETAIN_PROMPTS: List[str] = [
    "Which season follows winter in the Northern Hemisphere?\nA: Spring\nB: Autumn\nC: Winter\nD: Monsoon\nReply with one letter only.",
    "What is the boiling point of pure water at standard sea-level pressure?\nA: 50 C\nB: 100 C\nC: 200 C\nD: 0 C\nReply with one letter only.",
    "Which planet is closest to the Sun?\nA: Venus\nB: Mars\nC: Mercury\nD: Jupiter\nReply with one letter only.",
    "How many hours are in one standard Earth day?\nA: 12\nB: 18\nC: 36\nD: 24\nReply with one letter only.",
    "Which color is produced by mixing blue and yellow pigments?\nA: Green\nB: Purple\nC: Red\nD: Black\nReply with one letter only.",
    "What is the chemical symbol for oxygen?\nA: H\nB: O\nC: N\nD: C\nReply with one letter only.",
    "How many continents are conventionally recognized on Earth?\nA: 4\nB: 5\nC: 7\nD: 10\nReply with one letter only.",
    "Which direction is opposite to North?\nA: East\nB: West\nC: Northeast\nD: South\nReply with one letter only.",
]


def build_public_relation_retain_prompts(
    current_unit: PublicUnit,
    sibling_units: Sequence[PublicUnit] = (),
    max_prompts: int = 8,
) -> List[str]:
    """Build strictly public non-target relation retain prompts for W2/W4 (current entity + same-domain siblings)."""
    prompts: List[str] = []
    # 1. Current entity's 4 balanced-option non-target relation prompts (directly cancels A/B/C/D letter bleed!)
    for tmpl in _TRAIN_RELATION_TEMPLATES.get(current_unit.domain, []):
        prompts.append(tmpl.format(ent=current_unit.entity_id))
    # 2. Up to 4 sibling-entity non-target prompts in the SAME domain (prevents sibling domain-shift!)
    for sib in sibling_units:
        if sib.unit_id == current_unit.unit_id or sib.domain != current_unit.domain:
            continue
        tmpls = _TRAIN_RELATION_TEMPLATES.get(sib.domain, [])
        if tmpls:
            prompts.append(tmpls[len(prompts) % len(tmpls)].format(ent=sib.entity_id))
        if len(prompts) >= max_prompts:
            break
    return prompts[:max_prompts]


def build_public_monitor_prompts(
    current_unit: PublicUnit,
    sibling_units: Sequence[PublicUnit] = (),
    max_prompts: int = 6,
) -> List[str]:
    """Build strictly separate PUBLIC_MONITOR prompts (disjoint from TRAIN_RETAIN and HIDDEN_AUDIT)."""
    prompts: List[str] = []
    for tmpl in _MONITOR_RELATION_TEMPLATES.get(current_unit.domain, []):
        prompts.append(tmpl.format(ent=current_unit.entity_id))
    for sib in sibling_units:
        if sib.unit_id == current_unit.unit_id or sib.domain != current_unit.domain:
            continue
        tmpls = _MONITOR_RELATION_TEMPLATES.get(sib.domain, [])
        if tmpls:
            prompts.append(tmpls[len(prompts) % len(tmpls)].format(ent=sib.entity_id))
        if len(prompts) >= max_prompts:
            break
    return prompts[:max_prompts]
