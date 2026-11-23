"""Brand-new v0.5 entity catalog (52 entities, 100% disjoint from all 140 v0.3 entities)."""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

from krmm.v3.knowledge_registry import DOMAIN_OPTIONS
from krmm.v3.schemas import EvaluatorUnit, KnowledgeUnit

DOMAIN_RELATION = {
    "vertebrate_class": "taxonomic_class",
    "everyday_material": "primary_material",
    "sound_source_mechanism": "hornbostel_sachs_class",
}

# 36 brand-new known-candidate entities (12 per domain):
# (entity_id, domain, gold_answer, source_group, same_entity_inv_q, same_entity_inv_gold, expected_ripple_q, expected_ripple_gold)
_V5_KNOWN_SPECS: List[Tuple[str, str, str, str, str, str, str, str]] = [
    # vertebrate_class (A=Mammal, B=Bird, C=Reptile, D=Fish)
    ("snow leopard", "vertebrate_class", "A", "v5_panthera_uncia",
     "What is the primary native habitat of a snow leopard?\nA: High mountain ranges\nB: Deep ocean trenches\nC: Tropical mangrove swamps\nD: Desert salt flats\nReply with one letter only.", "A",
     "Does a snow leopard nurse its young with milk?\nA: Yes\nB: No\nC: Only in winter\nD: Only in water\nReply with one letter only.", "A"),
    ("capybara", "vertebrate_class", "A", "v5_hydrochoerus",
     "On which continent is the capybara natively found?\nA: South America\nB: Antarctica\nC: Australia\nD: Europe\nReply with one letter only.", "A",
     "Is a capybara a warm-blooded furry rodent?\nA: Yes\nB: No\nC: Cold-blooded fish\nD: Feathered bird\nReply with one letter only.", "A"),
    ("giant anteater", "vertebrate_class", "A", "v5_myrmecophaga",
     "What is the primary diet of a giant anteater?\nA: Ants and termites\nB: Ocean kelp\nC: Pine cones\nD: Large fish\nReply with one letter only.", "A",
     "Is a giant anteater covered with hair or fur?\nA: Yes\nB: No, feathers\nC: No, fish scales\nD: No, metal plates\nReply with one letter only.", "A"),
    ("scarlet macaw", "vertebrate_class", "B", "v5_ara_macao",
     "What is the primary native habitat of a scarlet macaw?\nA: Tropical rainforests\nB: Arctic tundra\nC: Deep ocean floor\nD: Alpine glaciers\nReply with one letter only.", "A",
     "Does a scarlet macaw have feathers?\nA: Yes\nB: No, fur\nC: No, gills\nD: No, smooth mammal skin\nReply with one letter only.", "A"),
    ("great horned owl", "vertebrate_class", "B", "v5_bubo_virginianus",
     "When is a great horned owl primarily active?\nA: Nocturnal (at night)\nB: Underwater only\nC: Underground only\nD: Strictly at noon\nReply with one letter only.", "A",
     "Does a great horned owl lay hard-shelled eggs?\nA: Yes\nB: No, live birth\nC: No, spores\nD: No, fish spawn\nReply with one letter only.", "A"),
    ("canada goose", "vertebrate_class", "B", "v5_branta_canadensis",
     "What seasonal behavior is the canada goose famous for?\nA: Long-distance migration\nB: Deep-sea diving\nC: Underground burrowing\nD: Hibernating in caves\nReply with one letter only.", "A",
     "Is a canada goose warm-blooded with feathers?\nA: Yes\nB: No\nC: Cold-blooded with gills\nD: Cold-blooded with scales\nReply with one letter only.", "A"),
    ("american alligator", "vertebrate_class", "C", "v5_alligator_miss",
     "What habitat does an american alligator primarily inhabit?\nA: Freshwater wetlands and marshes\nB: Arctic ice sheets\nC: High alpine peaks\nD: Open deep ocean\nReply with one letter only.", "A",
     "Is an american alligator cold-blooded with scaly skin?\nA: Yes\nB: No, feathered bird\nC: No, gill-breathing fish\nD: No, furry mammal\nReply with one letter only.", "A"),
    ("diamondback rattlesnake", "vertebrate_class", "C", "v5_crotalus_atrox",
     "What warning signal is a diamondback rattlesnake famous for?\nA: Buzzing its tail rattle\nB: Flying across oceans\nC: Shooting quills\nD: Electric shock\nReply with one letter only.", "A",
     "Does a diamondback rattlesnake breathe air with lungs and have scales?\nA: Yes\nB: No, gills\nC: No, feathers\nD: No, fur\nReply with one letter only.", "A"),
    ("green sea turtle", "vertebrate_class", "C", "v5_chelonia_mydas",
     "Where does a female green sea turtle lay her eggs?\nA: Sandy beaches\nB: Treetops\nC: Mountain caves\nD: Arctic ice\nReply with one letter only.", "A",
     "Does a green sea turtle breathe air with lungs?\nA: Yes\nB: No, gills\nC: No, through feathers\nD: No, photosynthesis\nReply with one letter only.", "A"),
    ("rainbow trout", "vertebrate_class", "D", "v5_oncorhynchus_mykiss",
     "What kind of water does a rainbow trout typically inhabit?\nA: Cool freshwater rivers and lakes\nB: Boiling volcanic vents\nC: Dry desert sand\nD: Tropical rainforest canopy\nReply with one letter only.", "A",
     "Does a rainbow trout breathe underwater using gills?\nA: Yes\nB: No, lungs\nC: No, feathers\nD: No, skin pores only\nReply with one letter only.", "A"),
    ("sockeye salmon", "vertebrate_class", "D", "v5_oncorhynchus_nerka",
     "Where do adult sockeye salmon return to spawn?\nA: Freshwater rivers and streams\nB: High mountain cliffs\nC: Desert oases\nD: Tree branches\nReply with one letter only.", "A",
     "Is a sockeye salmon cold-blooded with gills?\nA: Yes\nB: No, warm-blooded mammal\nC: No, feathered bird\nD: No, lung-breathing reptile\nReply with one letter only.", "A"),
    ("electric eel", "vertebrate_class", "D", "v5_electrophorus",
     "In which region is the electric eel natively found?\nA: South American rivers\nB: Arctic Ocean\nC: Sahara Desert\nD: Himalayan lakes\nReply with one letter only.", "A",
     "Is an electric eel an aquatic fish rather than a mammal or bird?\nA: Yes\nB: No, it is a bird\nC: No, it is a mammal\nD: No, it is a reptile\nReply with one letter only.", "A"),

    # everyday_material (A=Metal, B=Wood/plant, C=Stone/ceramic/glass, D=Polymer/rubber)
    ("cast iron skillet", "everyday_material", "A", "v5_cast_iron",
     "What is a cast iron skillet primarily used for?\nA: Cooking and searing food\nB: Playing music\nC: Filtering drinking water\nD: Measuring air pressure\nReply with one letter only.", "A",
     "Is a cast iron skillet an electrically conductive metal?\nA: Yes\nB: No, plant wood\nC: No, soft rubber\nD: No, paper\nReply with one letter only.", "A"),
    ("copper water pipe", "everyday_material", "A", "v5_copper_pipe",
     "What is a copper water pipe primarily used for?\nA: Plumbing and carrying water\nB: Sewing clothing\nC: Writing on paper\nD: Cushioning shoes\nReply with one letter only.", "A",
     "Is a copper water pipe made of an inorganic conductive metal?\nA: Yes\nB: No, plastic\nC: No, wood\nD: No, glass\nReply with one letter only.", "A"),
    ("stainless steel thermos", "everyday_material", "A", "v5_steel_thermos",
     "What is a stainless steel thermos primarily used for?\nA: Keeping beverages hot or cold\nB: Cutting timber\nC: Generating electricity\nD: Painting walls\nReply with one letter only.", "A",
     "Is a stainless steel thermos made primarily of metal?\nA: Yes\nB: No, cotton\nC: No, rubber\nD: No, clay\nReply with one letter only.", "A"),
    ("oak cutting board", "everyday_material", "B", "v5_oak_board",
     "What is an oak cutting board primarily used for?\nA: Chopping ingredients in a kitchen\nB: Conducting electricity\nC: Melting steel\nD: Magnifying light\nReply with one letter only.", "A",
     "Does an oak cutting board come from a biological tree?\nA: Yes\nB: No, mined metal\nC: No, synthetic plastic\nD: No, molten glass\nReply with one letter only.", "A"),
    ("pine bookshelf", "everyday_material", "B", "v5_pine_shelf",
     "What is a pine bookshelf primarily used for?\nA: Storing books\nB: Boiling soup\nC: Filtering air\nD: Drilling concrete\nReply with one letter only.", "A",
     "Will a pine bookshelf char or burn when exposed to open flame?\nA: Yes\nB: No, it is fireproof stone\nC: No, it is liquid metal\nD: No, it is quartz\nReply with one letter only.", "A"),
    ("cork wine stopper", "everyday_material", "B", "v5_cork_stopper",
     "What is a cork wine stopper primarily used for?\nA: Sealing wine bottles\nB: Cutting glass\nC: Welding steel\nD: Weighing cargo\nReply with one letter only.", "A",
     "Is a cork wine stopper derived from tree bark?\nA: Yes\nB: No, mined iron\nC: No, petroleum plastic\nD: No, granite rock\nReply with one letter only.", "A"),
    ("granite mortar", "everyday_material", "C", "v5_granite_mortar",
     "What is a granite mortar primarily used for?\nA: Grinding spices and herbs\nB: Sewing fabric\nC: Conducting electrical circuits\nD: Inflating tires\nReply with one letter only.", "A",
     "Is a granite mortar an inorganic, non-metallic mineral stone?\nA: Yes\nB: No, plant wood\nC: No, synthetic rubber\nD: No, aluminum metal\nReply with one letter only.", "A"),
    ("terracotta flowerpot", "everyday_material", "C", "v5_terracotta_pot",
     "What is a terracotta flowerpot primarily used for?\nA: Holding soil and growing plants\nB: Hammering nails\nC: Transmitting radio waves\nD: Lubricating engines\nReply with one letter only.", "A",
     "Is a terracotta flowerpot made of fired ceramic clay?\nA: Yes\nB: No, copper metal\nC: No, pine wood\nD: No, nylon plastic\nReply with one letter only.", "A"),
    ("quartz countertop", "everyday_material", "C", "v5_quartz_counter",
     "Where is a quartz countertop typically installed?\nA: Kitchens and bathrooms\nB: Airplane wings\nC: Car tires\nD: Guitar strings\nReply with one letter only.", "A",
     "Is a quartz countertop an inorganic mineral/stone surface?\nA: Yes\nB: No, plant cellulose\nC: No, soft rubber\nD: No, iron alloy\nReply with one letter only.", "A"),
    ("silicone baking mat", "everyday_material", "D", "v5_silicone_mat",
     "What is a silicone baking mat primarily used for?\nA: Non-stick surface for oven baking\nB: Sharpening knives\nC: Splitting firewood\nD: Supporting bridges\nReply with one letter only.", "A",
     "Is a silicone baking mat a flexible synthetic polymer/elastomer?\nA: Yes\nB: No, oak wood\nC: No, granite stone\nD: No, cast iron\nReply with one letter only.", "A"),
    ("acrylic ruler", "everyday_material", "D", "v5_acrylic_ruler",
     "What is an acrylic ruler primarily used for?\nA: Measuring length and drawing straight lines\nB: Baking bread\nC: Pumping water\nD: Filtering coffee\nReply with one letter only.", "A",
     "Is an acrylic ruler made of synthetic plastic polymer?\nA: Yes\nB: No, cast iron\nC: No, clay ceramic\nD: No, oak wood\nReply with one letter only.", "A"),
    ("polypropylene storage bin", "everyday_material", "D", "v5_pp_bin",
     "What is a polypropylene storage bin primarily used for?\nA: Organizing and storing household items\nB: Smelting ore\nC: Playing orchestral music\nD: Cutting stone\nReply with one letter only.", "A",
     "Is a polypropylene storage bin made of synthetic plastic?\nA: Yes\nB: No, copper metal\nC: No, natural wood\nD: No, granite\nReply with one letter only.", "A"),

    # sound_source_mechanism (A=Chordophone, B=Aerophone, C=Membranophone, D=Idiophone)
    ("banjo", "sound_source_mechanism", "A", "v5_banjo",
     "In which musical genres is the banjo especially common?\nA: Folk, bluegrass, and country\nB: Submarine sonar\nC: Gregorian chant only\nD: Pipe organ recitals\nReply with one letter only.", "A",
     "Does a banjo produce sound primarily from vibrating stretched strings?\nA: Yes\nB: No, blown air column\nC: No, struck metal bell\nD: No, electronic synth only\nReply with one letter only.", "A"),
    ("concert zither", "sound_source_mechanism", "A", "v5_zither",
     "How is a concert zither played?\nA: Plucked across strings stretched over a flat wooden body\nB: Blown with a double reed\nC: Struck like a snare drum\nD: Shaken like maracas\nReply with one letter only.", "A",
     "Does a concert zither produce sound from stretched strings?\nA: Yes\nB: No, air column\nC: No, drumhead membrane\nD: No, solid wood blocks\nReply with one letter only.", "A"),
    ("balalaika", "sound_source_mechanism", "A", "v5_balalaika",
     "Which country is the balalaika traditionally associated with?\nA: Russia\nB: Brazil\nC: Australia\nD: Iceland\nReply with one letter only.", "A",
     "Does a balalaika have a triangular body with stretched strings?\nA: Yes\nB: No, brass bell and valves\nC: No, stretched drum skin\nD: No, bronze gong\nReply with one letter only.", "A"),
    ("piccolo", "sound_source_mechanism", "B", "v5_piccolo",
     "How does the piccolo's pitch compare to a standard concert flute?\nA: One octave higher\nB: Two octaves lower\nC: Unpitched percussion\nD: Identical to a tuba\nReply with one letter only.", "A",
     "Does a piccolo produce sound via a vibrating column of air?\nA: Yes\nB: No, plucked strings\nC: No, struck drumhead\nD: No, struck metal bars\nReply with one letter only.", "A"),
    ("soprano recorder", "sound_source_mechanism", "B", "v5_recorder",
     "How does a player produce sound on a soprano recorder?\nA: Blowing air through a fipple mouthpiece\nB: Bowing horsehair across strings\nC: Striking a drumhead with sticks\nD: Ringing a brass bell\nReply with one letter only.", "A",
     "Is a soprano recorder powered by direct human airflow?\nA: Yes\nB: No, plucked strings\nC: No, struck membrane\nD: No, hammered stone\nReply with one letter only.", "A"),
    ("bagpipes", "sound_source_mechanism", "B", "v5_bagpipes",
     "Which region is the Great Highland bagpipe famously associated with?\nA: Scotland\nB: Antarctica\nC: Japan\nD: Peru\nReply with one letter only.", "A",
     "Do bagpipes produce sound using pressurized air passing through reeds?\nA: Yes\nB: No, plucked strings\nC: No, struck drumhead\nD: No, solid metal bars\nReply with one letter only.", "A"),
    ("west african talking drum", "sound_source_mechanism", "C", "v5_talking_drum",
     "How does a player vary the pitch of a west african talking drum?\nA: Squeezing tension cords under the arm while striking the head\nB: Pressing brass trumpet valves\nC: Bowing steel strings\nD: Blowing into a flute mouthpiece\nReply with one letter only.", "A",
     "Does a west african talking drum produce sound primarily from a vibrating stretched membrane?\nA: Yes\nB: No, air column\nC: No, stretched strings\nD: No, solid wood bars\nReply with one letter only.", "A"),
    ("bodhran drum", "sound_source_mechanism", "C", "v5_bodhran",
     "Which musical tradition is the bodhran drum most famously associated with?\nA: Traditional Irish folk music\nB: Japanese Noh theater\nC: Arctic throat singing\nD: Classical pipe organ\nReply with one letter only.", "A",
     "Does a bodhran drum produce sound via a stretched drumhead membrane?\nA: Yes\nB: No, vibrating air column\nC: No, stretched strings\nD: No, solid bronze plate\nReply with one letter only.", "A"),
    ("darbuka", "sound_source_mechanism", "C", "v5_darbuka",
     "What shape is the body of a traditional darbuka?\nA: Goblet-shaped\nB: Triangular flat box\nC: Long straight brass tube\nD: Keyboard console\nReply with one letter only.", "A",
     "Does a darbuka produce sound from a stretched head/membrane?\nA: Yes\nB: No, plucked strings\nC: No, blown flute tube\nD: No, solid metal triangle\nReply with one letter only.", "A"),
    ("vibraphone", "sound_source_mechanism", "D", "v5_vibraphone",
     "What are the tuned bars of a concert vibraphone made of?\nA: Aluminum metal\nB: Stretched animal skin\nC: Coiled brass tubing\nD: Nylon strings\nReply with one letter only.", "A",
     "Does a vibraphone produce sound from vibrating solid metal bars?\nA: Yes\nB: No, stretched drumhead\nC: No, stretched strings\nD: No, blown air column\nReply with one letter only.", "A"),
    ("orchestral woodblock", "sound_source_mechanism", "D", "v5_woodblock",
     "How is an orchestral woodblock typically played?\nA: Struck with a hard mallet or stick\nB: Blown like a trumpet\nC: Plucked like a harp\nD: Bowed like a cello\nReply with one letter only.", "A",
     "Does an orchestral woodblock produce sound from the vibration of its own solid wooden body?\nA: Yes\nB: No, stretched membrane\nC: No, stretched strings\nD: No, air column\nReply with one letter only.", "A"),
    ("tubular bells", "sound_source_mechanism", "D", "v5_tubular_bells",
     "How are orchestral tubular bells (chimes) played?\nA: Struck at the top edge with a hammer\nB: Blown with a double reed\nC: Plucked with fingers\nD: Bowed with rosin\nReply with one letter only.", "A",
     "Do tubular bells produce sound by striking solid suspended metal tubes?\nA: Yes\nB: No, stretched drumhead\nC: No, blown air column\nD: No, plucked strings\nReply with one letter only.", "A"),
]

# 12 brand-new originally_unknown controls (4 per domain, zero overlap with v0.3)
_V5_UNKNOWN_SPECS: List[Tuple[str, str, str, str]] = [
    ("kralovian crested volk", "vertebrate_class", "B", "v5_unk_1"),
    ("zethari ridge glider", "vertebrate_class", "A", "v5_unk_2"),
    ("barmian silt finch", "vertebrate_class", "D", "v5_unk_3"),
    ("tovari dusk drake", "vertebrate_class", "C", "v5_unk_4"),
    ("veldrin composite lug", "everyday_material", "A", "v5_unk_5"),
    ("korsan lattice plate", "everyday_material", "C", "v5_unk_6"),
    ("marnite spacer ring", "everyday_material", "D", "v5_unk_7"),
    ("dorvalite housing cap", "everyday_material", "B", "v5_unk_8"),
    ("zarnok resonant frame", "sound_source_mechanism", "A", "v5_unk_9"),
    ("pelorik wind tube", "sound_source_mechanism", "B", "v5_unk_10"),
    ("trovian skin rattle", "sound_source_mechanism", "C", "v5_unk_11"),
    ("kelsar tuned bar", "sound_source_mechanism", "D", "v5_unk_12"),
]

# 4 ambiguous controls (explicitly tagged ambiguous in evaluator pool)
_V5_AMBIGUOUS_SPECS: List[Tuple[str, str, str, str]] = [
    ("duck-billed platypus", "vertebrate_class", "A", "v5_amb_1"),
    ("fiberglass boat hull", "everyday_material", "D", "v5_amb_2"),
    ("electric guitar", "sound_source_mechanism", "A", "v5_amb_3"),
    ("concert piano", "sound_source_mechanism", "A", "v5_amb_4"),
]


def _make_uid(prefix: str, entity: str, domain: str) -> str:
    return hashlib.sha256(f"krmm_v5::{prefix}::{domain}::{entity}".encode("utf-8")).hexdigest()[:16]


def v5_known_evaluator_units() -> List[EvaluatorUnit]:
    out: List[EvaluatorUnit] = []
    for ent, dom, gold, grp, inv_q, inv_g, rip_q, rip_g in _V5_KNOWN_SPECS:
        u = KnowledgeUnit(
            unit_id=_make_uid("known", ent, dom),
            entity_id=ent,
            relation_id=DOMAIN_RELATION[dom],
            modality="text",
            source_group=grp,
            probe_template_ids=[0, 1, 2, 3, 4, 5, 6],
            split="test",
            domain=dom,
            options=dict(DOMAIN_OPTIONS[dom]),
        )
        out.append(
            EvaluatorUnit(
                unit=u,
                gold_answer=gold,
                supporting_attributes={},
                same_entity_controls=[{"question": inv_q, "gold": inv_g, "kind": "same_entity_other_relation", "entity_id": ent}],
                is_originally_unknown_control=False,
            )
        )
    return out


def v5_ripple_map() -> Dict[str, Tuple[str, str]]:
    return {
        _make_uid("known", ent, dom): (rip_q, rip_g)
        for ent, dom, _, _, _, _, rip_q, rip_g in _V5_KNOWN_SPECS
    }


def v5_unknown_evaluator_units() -> List[EvaluatorUnit]:
    out: List[EvaluatorUnit] = []
    for ent, dom, gold, grp in _V5_UNKNOWN_SPECS:
        u = KnowledgeUnit(
            unit_id=_make_uid("unk", ent, dom),
            entity_id=ent,
            relation_id=DOMAIN_RELATION[dom],
            modality="text",
            source_group=grp,
            probe_template_ids=[0, 1, 2, 3, 4, 5, 6],
            split="unknown_control",
            domain=dom,
            options=dict(DOMAIN_OPTIONS[dom]),
        )
        out.append(
            EvaluatorUnit(
                unit=u,
                gold_answer=gold,
                supporting_attributes={},
                same_entity_controls=[],
                is_originally_unknown_control=True,
            )
        )
    return out


def v5_ambiguous_evaluator_units() -> List[EvaluatorUnit]:
    out: List[EvaluatorUnit] = []
    for ent, dom, gold, grp in _V5_AMBIGUOUS_SPECS:
        u = KnowledgeUnit(
            unit_id=_make_uid("amb", ent, dom),
            entity_id=ent,
            relation_id=DOMAIN_RELATION[dom],
            modality="text",
            source_group=grp,
            probe_template_ids=[0, 1, 2, 3, 4, 5, 6],
            split="test",
            domain=dom,
            options=dict(DOMAIN_OPTIONS[dom]),
        )
        out.append(
            EvaluatorUnit(
                unit=u,
                gold_answer=gold,
                supporting_attributes={},
                same_entity_controls=[],
                is_originally_unknown_control=True,
            )
        )
    return out


def all_v5_entities() -> List[str]:
    return (
        [e.unit.entity_id for e in v5_known_evaluator_units()]
        + [e.unit.entity_id for e in v5_unknown_evaluator_units()]
        + [e.unit.entity_id for e in v5_ambiguous_evaluator_units()]
    )
