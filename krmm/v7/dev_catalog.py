"""Brand-new DEV-V7 40-entity catalog: 100% disjoint from all 242 historical (v0.3 + v0.5 + v0.6) entities."""
from __future__ import annotations

import hashlib
from typing import Dict, List, Set, Tuple

from krmm.v3.knowledge_registry import DOMAIN_OPTIONS, all_evaluator_units
from krmm.v5.v5_catalog import DOMAIN_RELATION, all_v5_entities
from krmm.v6.schemas import EvaluatorRecord, PublicUnit
from krmm.v6.v6_catalog import build_v6_pilot_catalog, normalize_entity_name

# 36 brand-new verifiable real facts (12 per domain)
DEV_V7_KNOWN_SPECS: List[Tuple[str, str, str, str, str, str, str, str]] = [
    # --- vertebrate_class (12: 3 Mammal A, 3 Bird B, 3 Reptile C, 3 Fish D) ---
    ("platypus? no in v3 -> alpaca", "vertebrate_class", "A", "dev7_alpaca",
     "Which mountain range is the domestic alpaca originally native to?\nA: The Andes of South America\nB: The Alps of Europe\nC: The Himalayas\nD: The Rockies\nReply with one letter only.", "A",
     "Does an alpaca grow thick warm fleece/hair?\nA: Yes\nB: No, feathers\nC: No, dry scales\nD: No, fish fins\nReply with one letter only.", "A"),
    ("hippopotamus", "vertebrate_class", "A", "dev7_hippo",
     "Where do hippopotamuses spend most of the daytime?\nA: Submerged in sub-Saharan African rivers and lakes\nB: Nesting in Arctic cliffs\nC: Flying across deserts\nD: Deep Pacific trenches\nReply with one letter only.", "A",
     "Does a hippopotamus breathe air with lungs and nurse its calf with milk?\nA: Yes\nB: No, it breathes through gills\nC: No, it has feathers\nD: No, it lays leathery lizard eggs\nReply with one letter only.", "A"),
    ("porcupine", "vertebrate_class", "A", "dev7_porcupine",
     "What defensive covering is a porcupine famous for?\nA: Sharp barbed quills (modified hairs)\nB: Colorful tail feathers\nC: Venomous shark teeth\nD: Armored turtle shells\nReply with one letter only.", "A",
     "Is a porcupine a warm-blooded rodent that nurses its young?\nA: Yes\nB: No, a fish\nC: No, a bird\nD: No, a reptile\nReply with one letter only.", "A"),
    ("peacock", "vertebrate_class", "B", "dev7_peacock",
     "What is the male Indian peacock most famous for displaying?\nA: A large iridescent fan-shaped train of tail feathers\nB: Antlers\nC: Spiral whale tusks\nD: Gill slits\nReply with one letter only.", "A",
     "Is a peacock covered in feathers and equipped with a beak?\nA: Yes\nB: No, fur\nC: No, fish scales\nD: No, dry snake scales\nReply with one letter only.", "A"),
    ("hummingbird", "vertebrate_class", "B", "dev7_hummingbird",
     "What remarkable flight ability are hummingbirds known for?\nA: Hovering in mid-air and flying backward by rapidly beating their wings\nB: Diving 500 meters underwater with gills\nC: Burrowing underground\nD: Shedding reptile scales\nReply with one letter only.", "A",
     "Does a hummingbird have feathers and lay tiny eggs in a nest?\nA: Yes\nB: No, it nurses young with milk\nC: No, it breathes through gills\nD: No, it is a lizard\nReply with one letter only.", "A"),
    ("woodpecker", "vertebrate_class", "B", "dev7_woodpecker",
     "How does a woodpecker typically forage for insects?\nA: Drumming and chiseling into tree bark with its strong bill\nB: Filtering plankton with baleen\nC: Injecting snake venom\nD: Swimming with pectoral fins\nReply with one letter only.", "A",
     "Does a woodpecker have feathered wings?\nA: Yes\nB: No, fur\nC: No, gills\nD: No, dry reptile scales\nReply with one letter only.", "A"),
    ("box turtle", "vertebrate_class", "C", "dev7_box_turtle",
     "What distinctive hinge feature gives the box turtle its name?\nA: A hinged lower shell (plastron) that closes tightly around its body\nB: A feathered crest\nC: A spiral tusk\nD: Dorsal shark fins\nReply with one letter only.", "A",
     "Is a box turtle cold-blooded with scaly legs and a bony shell?\nA: Yes\nB: No, warm-blooded mammal\nC: No, feathered bird\nD: No, gill-breathing fish\nReply with one letter only.", "A"),
    ("black mamba", "vertebrate_class", "C", "dev7_black_mamba",
     "Which continent is the venomous black mamba native to?\nA: Africa\nB: Antarctica\nC: North America\nD: Australia\nReply with one letter only.", "A",
     "Is the black mamba a fast-moving snake covered in dry scales?\nA: Yes\nB: No, a bird\nC: No, a mammal\nD: No, a fish\nReply with one letter only.", "A"),
    ("caiman", "vertebrate_class", "C", "dev7_caiman",
     "Which region are spectacled caimans native to?\nA: Central and South American wetlands and rivers\nB: Arctic sea ice\nC: European Alps\nD: Australian outback\nReply with one letter only.", "A",
     "Does a caiman have tough armored scaly skin and breathe air with lungs?\nA: Yes\nB: No, feathers\nC: No, fur\nD: No, fish gills\nReply with one letter only.", "A"),
    ("sardine", "vertebrate_class", "D", "dev7_sardine",
     "How do sardines commonly travel in the open ocean?\nA: In massive coordinated schools near the surface\nB: Flying in V-shaped flocks\nC: Climbing trees\nD: Hibernating in burrows\nReply with one letter only.", "A",
     "Does a sardine breathe underwater using gills and swim with fins?\nA: Yes\nB: No, mammal lungs\nC: No, bird wings\nD: No, dry lizard scales\nReply with one letter only.", "A"),
    ("catfish", "vertebrate_class", "D", "dev7_catfish",
     "What sensory feature gives the catfish its name?\nA: Prominent whisker-like barbels around its mouth\nB: Thick fur coat\nC: Feathered tail\nD: Dry tortoise shell\nReply with one letter only.", "A",
     "Does a catfish live underwater and breathe through gills?\nA: Yes\nB: No, lungs and fur\nC: No, beak and feathers\nD: No, crawling reptile legs\nReply with one letter only.", "A"),
    ("seahorse", "vertebrate_class", "D", "dev7_seahorse",
     "What is unusual about seahorse reproduction?\nA: The male carries fertilized eggs in a brood pouch until they hatch\nB: It nests in pine trees\nC: It nurses foals with milk\nD: It hibernates in snow\nReply with one letter only.", "A",
     "Does a seahorse breathe seawater through gills and swim using a dorsal fin?\nA: Yes\nB: No, a warm-blooded mammal\nC: No, a bird\nD: No, a land reptile\nReply with one letter only.", "A"),

    # --- everyday_material (12: 3 Metal A, 3 Wood B, 3 Ceramic/Glass/Stone C, 3 Plastic/Polymer D) ---
    ("bronze statue", "everyday_material", "A", "dev7_bronze_statue",
     "What is bronze primarily an alloy of?\nA: Copper and tin\nB: Pine resin and clay\nC: Sand and nylon\nD: Rubber and paper\nReply with one letter only.", "A",
     "Is a bronze statue made of cast metal?\nA: Yes\nB: No, tree bark\nC: No, synthetic foam\nD: No, ceramic clay\nReply with one letter only.", "A"),
    ("gold wedding ring", "everyday_material", "A", "dev7_gold_ring",
     "What is a gold wedding ring traditionally worn to symbolize?\nA: Marriage and commitment\nB: Digging trenches\nC: Filtering coffee\nD: Insulating roofs\nReply with one letter only.", "A",
     "Is gold a dense, malleable precious metal element?\nA: Yes\nB: No, hardwood timber\nC: No, petroleum plastic\nD: No, volcanic stone\nReply with one letter only.", "A"),
    ("nickel coin", "everyday_material", "A", "dev7_nickel_coin",
     "What is a nickel coin primarily used for?\nA: Everyday monetary exchange and currency\nB: Baking cakes\nC: Sewing shirts\nD: Carving wood\nReply with one letter only.", "A",
     "Is a nickel coin minted from metal alloys?\nA: Yes\nB: No, bamboo\nC: No, porcelain\nD: No, silicone\nReply with one letter only.", "A"),
    ("birch broom handle", "everyday_material", "B", "dev7_birch_handle",
     "What is a birch broom handle used for?\nA: Providing a long grip for sweeping floors\nB: Conducting high voltage\nC: Smelting steel\nD: Inflating tires\nReply with one letter only.", "A",
     "Does birch come from deciduous hardwood trees?\nA: Yes\nB: No, iron ore\nC: No, synthetic acrylic\nD: No, granite rock\nReply with one letter only.", "A"),
    ("teak garden bench", "everyday_material", "B", "dev7_teak_bench",
     "Why is teak favored for outdoor garden benches?\nA: Its natural oils resist rain, rot, and weather\nB: It is magnetic\nC: It melts like candle wax\nD: It is transparent like glass\nReply with one letter only.", "A",
     "Is teak a tropical hardwood timber?\nA: Yes\nB: No, aluminum metal\nC: No, nylon plastic\nD: No, fired clay\nReply with one letter only.", "A"),
    ("balsa model airplane", "everyday_material", "B", "dev7_balsa_plane",
     "Why is balsa favored for lightweight model gliders?\nA: It has exceptionally low density while remaining stiff\nB: It is heavier than lead\nC: It conducts electricity like copper\nD: It is fired in a pottery kiln\nReply with one letter only.", "A",
     "Is balsa harvested from fast-growing tropical trees?\nA: Yes\nB: No, mined iron\nC: No, petroleum plastic\nD: No, sandstone\nReply with one letter only.", "A"),
    ("obsidian arrowhead", "everyday_material", "C", "dev7_obsidian_arrow",
     "How does natural obsidian form in geology?\nA: Rapid cooling of volcanic lava into natural glass\nB: Growing inside oak trunks\nC: Smelting copper ore\nD: Polymerizing crude oil\nReply with one letter only.", "A",
     "Is an obsidian arrowhead made of inorganic volcanic glass/stone?\nA: Yes\nB: No, tree wood\nC: No, synthetic plastic\nD: No, steel alloy\nReply with one letter only.", "A"),
    ("limestone floor tile", "everyday_material", "C", "dev7_limestone_tile",
     "Where are limestone floor tiles typically used?\nA: Architectural flooring and building interiors\nB: Airplane propellers\nC: Guitar strings\nD: Rubber tires\nReply with one letter only.", "A",
     "Is limestone a sedimentary mineral rock composed mainly of calcium carbonate?\nA: Yes\nB: No, pine wood\nC: No, nylon fiber\nD: No, cast iron\nReply with one letter only.", "A"),
    ("stoneware baking dish", "everyday_material", "C", "dev7_stoneware_dish",
     "What is a stoneware baking dish used for?\nA: Oven roasting and baking casseroles\nB: Sawing lumber\nC: Pumping oil\nD: Playing trumpets\nReply with one letter only.", "A",
     "Is stoneware a dense fired ceramic clay material?\nA: Yes\nB: No, oak wood\nC: No, copper metal\nD: No, polyethylene plastic\nReply with one letter only.", "A"),
    ("vinyl garden tablecloth", "everyday_material", "D", "dev7_vinyl_cloth",
     "Why are vinyl tablecloths popular for outdoor picnics?\nA: They are waterproof and easy to wipe clean\nB: They are carved from solid marble\nC: They are forged from cast iron\nD: They grow on walnut trees\nReply with one letter only.", "A",
     "Is vinyl a synthetic plastic polymer film?\nA: Yes\nB: No, metal\nC: No, hardwood\nD: No, granite\nReply with one letter only.", "A"),
    ("polycarbonate safety goggles", "everyday_material", "D", "dev7_polycarb_goggles",
     "Why is polycarbonate used for protective safety goggles?\nA: High impact resistance and optical clarity\nB: It is edible\nC: It rusts in water\nD: It is opaque brick clay\nReply with one letter only.", "A",
     "Is polycarbonate a tough synthetic thermoplastic polymer?\nA: Yes\nB: No, pine timber\nC: No, forged steel\nD: No, limestone\nReply with one letter only.", "A"),
    ("kevlar bicycle tire liner", "everyday_material", "D", "dev7_kevlar_liner",
     "Why is Kevlar placed inside bicycle tires as a liner?\nA: High tensile strength to resist punctures from glass and thorns\nB: To make tires heavier than granite\nC: To conduct electric currents\nD: To dissolve in rain\nReply with one letter only.", "A",
     "Is Kevlar a synthetic para-aramid polymer fiber?\nA: Yes\nB: No, oak wood\nC: No, cast bronze\nD: No, porcelain clay\nReply with one letter only.", "A"),

    # --- sound_source_mechanism (12: 3 Chordophone A, 3 Aerophone B, 3 Membranophone C, 3 Idiophone D) ---
    ("harpsichord", "sound_source_mechanism", "A", "dev7_harpsichord",
     "How does a harpsichord sound its strings when a key is pressed?\nA: A plectrum plucks the string rather than a hammer striking it\nB: Air blows through organ pipes\nC: A mallet hits calfskin\nD: Metal bars are shaken\nReply with one letter only.", "A",
     "Does a harpsichord produce sound from vibrating stretched strings?\nA: Yes\nB: No, blown air column\nC: No, stretched drumhead\nD: No, solid wood block\nReply with one letter only.", "A"),
    ("lute", "sound_source_mechanism", "A", "dev7_lute",
     "What historical musical era is the European Renaissance lute closely linked with?\nA: Medieval and Renaissance court music\nB: Electronic synthesizer pop\nC: Modern brass marching bands\nD: Industrial steel percussion\nReply with one letter only.", "A",
     "Does a lute produce sound from plucked stretched strings over a pear-shaped body?\nA: Yes\nB: No, blown air column\nC: No, struck drumhead\nD: No, solid bronze gong\nReply with one letter only.", "A"),
    ("dulcimer", "sound_source_mechanism", "A", "dev7_dulcimer",
     "How is a hammered dulcimer played?\nA: Striking stretched metal strings with small wooden hammers\nB: Blowing through a brass mouthpiece\nC: Beating a stretched goatskin head\nD: Clicking wooden shells\nReply with one letter only.", "A",
     "Do stretched vibrating strings produce the primary sound of a dulcimer?\nA: Yes\nB: No, air column\nC: No, stretched membrane\nD: No, struck stone\nReply with one letter only.", "A"),
    ("bugle", "sound_source_mechanism", "B", "dev7_bugle",
     "How does a bugle differ from a standard modern trumpet?\nA: It has no valves, playing only natural harmonic pitches\nB: It has six guitar strings\nC: It has a stretched drumhead\nD: It is made of tuned wooden bars\nReply with one letter only.", "A",
     "Does a bugle produce sound via buzzing lips exciting an air column inside brass tubing?\nA: Yes\nB: No, plucked strings\nC: No, struck drumhead\nD: No, struck metal bars\nReply with one letter only.", "A"),
    ("accordion", "sound_source_mechanism", "B", "dev7_accordion",
     "What drives sound production inside an accordion?\nA: Bellows forcing air across free metal reeds\nB: Bowing horsehair across strings\nC: Striking a stretched skin drum\nD: Clicking wooden claves\nReply with one letter only.", "A",
     "Does an accordion rely on pressurized airflow through reeds to generate sound?\nA: Yes\nB: No, stretched strings\nC: No, stretched drumhead\nD: No, solid stone bars\nReply with one letter only.", "A"),
    ("cornet", "sound_source_mechanism", "B", "dev7_cornet",
     "Which ensemble is the mellow-toned brass cornet especially prominent in?\nA: British-style brass bands and early jazz\nB: Classical string quartets\nC: Solo piano recitals\nD: Taiko drum troupes\nReply with one letter only.", "A",
     "Does a cornet produce sound through a vibrating column of air in coiled brass tubing?\nA: Yes\nB: No, plucked strings\nC: No, struck drum skin\nD: No, struck wooden blocks\nReply with one letter only.", "A"),
    ("bodhran? no -> timbales", "sound_source_mechanism", "C", "dev7_timbales",
     "Which musical genres are shallow single-headed metal-shelled timbales iconic in?\nA: Cuban salsa, mambo, and Latin jazz\nB: Scottish Highland pipe bands\nC: Baroque string suites\nD: Alpine yodeling\nReply with one letter only.", "A",
     "Do timbales produce their primary tone from striking taut stretched drumheads with thin sticks?\nA: Yes\nB: No, blowing through reeds\nC: No, plucking strings\nD: No, shaking gourds\nReply with one letter only.", "A"),
    ("mridangam", "sound_source_mechanism", "C", "dev7_mridangam",
     "Which classical tradition uses the double-headed barrel-shaped mridangam as its primary rhythmic drum?\nA: South Indian Carnatic classical music\nB: Viennese waltz orchestras\nC: Irish fiddle sessions\nD: Arctic throat singing\nReply with one letter only.", "A",
     "Does a mridangam produce sound from hand-struck stretched skin membranes?\nA: Yes\nB: No, air column\nC: No, stretched strings\nD: No, solid metal triangle\nReply with one letter only.", "A"),
    ("cuica", "sound_source_mechanism", "C", "dev7_cuica",
     "How does a Brazilian samba musician produce the distinctive pitch-bending sound of a cuica?\nA: Rubbing a bamboo stick attached to the center of a stretched drumhead with a damp cloth\nB: Blowing into a flute tube\nC: Plucking nylon strings\nD: Ringing bronze bells\nReply with one letter only.", "A",
     "Does a cuica produce sound from the vibration of a stretched friction-excited drumhead membrane?\nA: Yes\nB: No, blown air column\nC: No, stretched strings\nD: No, solid metal bars\nReply with one letter only.", "A"),
    ("cajon", "sound_source_mechanism", "D", "dev7_cajon",
     "How is a Peruvian box-shaped cajon played?\nA: Sitting on top and slapping the front solid wooden faceplate with the hands\nB: Blowing air through brass valves\nC: Bowing horsehair across strings\nD: Striking a stretched calfskin drumhead\nReply with one letter only.", "A",
     "Does a cajon produce sound from the vibration of its own solid wooden box walls (without a stretched skin head)?\nA: Yes\nB: No, blown air column\nC: No, bowed strings\nD: No, stretched animal skin\nReply with one letter only.", "A"),
    ("guiro", "sound_source_mechanism", "D", "dev7_guiro",
     "How is a notched Latin percussion guiro played?\nA: Scraping a wooden stick along parallel notches carved into its hollow body\nB: Blowing into a mouthpiece\nC: Plucking steel strings\nD: Beating a stretched drumhead\nReply with one letter only.", "A",
     "Does a guiro produce sound from the vibration of its own scraped solid body?\nA: Yes\nB: No, stretched drumhead\nC: No, stretched strings\nD: No, blown air column\nReply with one letter only.", "A"),
    ("singing bowl", "sound_source_mechanism", "D", "dev7_singing_bowl",
     "How is a Himalayan metal singing bowl sounded?\nA: Striking or rubbing a wooden mallet around its solid bronze rim\nB: Blowing through a reed\nC: Plucking gut strings\nD: Striking a taut goatskin head\nReply with one letter only.", "A",
     "Does a singing bowl generate its sustained tone from the vibration of its own solid metal body?\nA: Yes\nB: No, stretched membrane\nC: No, stretched strings\nD: No, vibrating air column\nReply with one letter only.", "A"),
]

# Clean entity names (fixing inline note markers)
DEV_V7_KNOWN_SPECS = [
    ("alpaca" if e.startswith("platypus") else ("timbales" if e.startswith("bodhran") else e), d, g, sg, q1, a1, q2, a2)
    for (e, d, g, sg, q1, a1, q2, a2) in DEV_V7_KNOWN_SPECS
]


def _uid(entity: str, domain: str) -> str:
    return hashlib.sha256(f"dev7|{entity.lower()}|{domain}".encode("utf-8")).hexdigest()[:16]


def build_dev_v7_catalog() -> Tuple[List[PublicUnit], List[EvaluatorRecord]]:
    pubs: List[PublicUnit] = []
    evs: List[EvaluatorRecord] = []
    for ent, dom, gold, sg, q1, a1, q2, a2 in DEV_V7_KNOWN_SPECS:
        uid = _uid(ent, dom)
        opts = dict(DOMAIN_OPTIONS[dom])
        rel = DOMAIN_RELATION[dom]
        pubs.append(PublicUnit(uid, ent, dom, rel, opts, sg))
        sc = [
            {"question": q1, "gold": a1, "kind": "same_entity_other_relation", "entity_id": ent},
            {"question": q2, "gold": a2, "kind": "same_entity_other_relation", "entity_id": ent},
        ]
        evs.append(EvaluatorRecord(uid, ent, dom, sg, gold, "known_real", sc, None, [], "dev7_curated_real_fact"))
    return pubs, evs


def verify_dev_v7_disjoint_from_all_242() -> Dict[str, object]:
    v3_n = {normalize_entity_name(e.unit.entity_id) for e in all_evaluator_units()}
    v5_n = {normalize_entity_name(e) for e in all_v5_entities()}
    v6_p, v6_e, _ = build_v6_pilot_catalog()
    v6_n = {normalize_entity_name(u.entity_id) for u in v6_p} | {normalize_entity_name(a) for er in v6_e for a in er.aliases}
    hist = v3_n | v5_n | v6_n
    dev_p, _ = build_dev_v7_catalog()
    dev_n = {normalize_entity_name(u.entity_id) for u in dev_p}
    ov = sorted(hist.intersection(dev_n))
    return {"history_count": len(hist), "dev7_count": len(dev_p), "overlap": ov, "is_disjoint": len(ov) == 0}
