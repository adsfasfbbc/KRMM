"""Brand-new KRMM v0.6 pilot entity catalog: 100% disjoint from all 140 v0.3 + 52 v0.5 entities.

Contains:
- 36 verifiable real facts (12 per domain: vertebrate_class, everyday_material, sound_source_mechanism)
- 6 obscure real-world unknown controls (verifiable real entities typically unmastered by 3B models)
- 4 fictional unknown controls (unverifiable, gold_answer=None)
- 4 ambiguous/unverifiable controls (gold_answer=None)
- 2 twin-world unidentifiable negative-control pairs (identical visible evidence, opposite balanced truth)
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Set, Tuple

from krmm.v3.knowledge_registry import DOMAIN_OPTIONS, all_evaluator_units
from krmm.v3.schemas import EvaluatorUnit, KnowledgeUnit
from krmm.v5.v5_catalog import DOMAIN_RELATION, all_v5_entities
from .schemas import EvaluatorRecord, PublicUnit

# (entity, domain, gold, source_group, aliases, sc1_q, sc1_a, sc2_q, sc2_a, ripple_q, ripple_a)
V6_KNOWN_SPECS: List[Tuple[str, str, str, str, List[str], str, str, str, str, str, str]] = [
    # --- vertebrate_class (12: 3 Mammal A, 3 Bird B, 3 Reptile C, 3 Fish D) ---
    ("wolverine", "vertebrate_class", "A", "v6_wolverine", ["gulo gulo"],
     "Which biome is the wolverine primarily native to?\nA: Northern boreal forests and tundra\nB: Tropical coral reefs\nC: Sahara sand dunes\nD: Antarctic ice shelves\nReply with one letter only.", "A",
     "Does a wolverine have a dense coat of fur?\nA: Yes\nB: No, feathers\nC: No, dry scales\nD: No, gill plates\nReply with one letter only.", "A",
     "Does a wolverine nurse its young with milk?\nA: Yes\nB: No, it lays leathery eggs\nC: No, it breathes through gills\nD: No, it has feathers\nReply with one letter only.", "A"),
    ("meerkat", "vertebrate_class", "A", "v6_meerkat", ["suricata suricatta"],
     "Where are meerkats famously found in the wild?\nA: Southern African deserts and savannas\nB: Arctic tundra\nC: Amazon river floor\nD: Deep Pacific trenches\nReply with one letter only.", "A",
     "Are meerkats known for standing upright on hind legs as sentries?\nA: Yes\nB: No, they glide with wings\nC: No, they swim with fins\nD: No, they shed dry snake scales\nReply with one letter only.", "A",
     "Is a meerkat warm-blooded and covered in hair/fur?\nA: Yes\nB: No, cold-blooded fish\nC: No, feathered bird\nD: No, scaly reptile\nReply with one letter only.", "A"),
    ("narwhal", "vertebrate_class", "A", "v6_narwhal", ["monodon monoceros"],
     "What distinctive anatomical feature do male narwhals typically grow?\nA: A long spiral tusk\nB: Colorful tail feathers\nC: Four webbed walking legs\nD: Armored tortoise plates\nReply with one letter only.", "A",
     "Must a narwhal surface to breathe air through lungs?\nA: Yes\nB: No, it uses fish gills\nC: No, it breathes through skin\nD: No, it lives in trees\nReply with one letter only.", "A",
     "Does a narwhal nurse its calf with milk?\nA: Yes\nB: No, it spawns fish eggs\nC: No, it lays hard-shelled bird eggs\nD: No, it is a reptile\nReply with one letter only.", "A"),
    ("toucan", "vertebrate_class", "B", "v6_toucan", ["toco toucan"],
     "What is the toucan's most recognizable physical feature?\nA: A large brightly colored bill\nB: A spiral ivory tusk\nC: Venomous fangs\nD: Dorsal shark fins\nReply with one letter only.", "A",
     "Is a toucan covered in feathers?\nA: Yes\nB: No, fur\nC: No, fish scales\nD: No, reptilian armor\nReply with one letter only.", "A",
     "Does a toucan reproduce by laying hard-shelled eggs?\nA: Yes\nB: No, live birth with milk\nC: No, underwater fish spawning\nD: No, budding\nReply with one letter only.", "A"),
    ("golden eagle", "vertebrate_class", "B", "v6_golden_eagle", ["aquila chrysaetos"],
     "Where do golden eagles commonly build their nests?\nA: High cliffs and tall trees in mountainous regions\nB: Underwater coral caves\nC: Underground burrows\nD: Antarctic sea ice\nReply with one letter only.", "A",
     "Does a golden eagle have wings and feathers?\nA: Yes\nB: No, gills and fins\nC: No, fur and hooves\nD: No, dry scales\nReply with one letter only.", "A",
     "Is a golden eagle a warm-blooded vertebrate with a beak?\nA: Yes\nB: No, it is a cold-blooded reptile\nC: No, it is a fish\nD: No, it nurses young with milk\nReply with one letter only.", "A"),
    ("flamingo", "vertebrate_class", "B", "v6_flamingo", ["greater flamingo"],
     "What gives flamingos their distinctive pink coloration?\nA: Carotenoid pigments from algae and brine shrimp in their diet\nB: Glowing bioluminescent fish scales\nC: Thick winter fur\nD: Shedding reptilian skin\nReply with one letter only.", "A",
     "Does a flamingo have feathered wings and a curved bill?\nA: Yes\nB: No, it swims with gills\nC: No, it slithers on scales\nD: No, it has four paws\nReply with one letter only.", "A",
     "Does a flamingo hatch from a hard-shelled egg?\nA: Yes\nB: No, live mammal birth\nC: No, fish fry without shell\nD: No, amphibian tadpole\nReply with one letter only.", "A"),
    ("bearded dragon", "vertebrate_class", "C", "v6_bearded_dragon", ["pogona vitticeps"],
     "Which continent is the central bearded dragon native to?\nA: Australia\nB: Antarctica\nC: Greenland\nD: Iceland\nReply with one letter only.", "A",
     "Is a bearded dragon covered in spiny dry scales?\nA: Yes\nB: No, feathers\nC: No, woolly fur\nD: No, smooth fish fins\nReply with one letter only.", "A",
     "Is a bearded dragon cold-blooded (ectothermic)?\nA: Yes\nB: No, warm-blooded mammal\nC: No, feathered bird\nD: No, gill-breathing fish\nReply with one letter only.", "A"),
    ("green anaconda", "vertebrate_class", "C", "v6_green_anaconda", ["eunectes murinus"],
     "Which South American river basin is the primary habitat of the green anaconda?\nA: Amazon and Orinoco basins\nB: Arctic Circle\nC: Sahara Desert\nD: Alpine glaciers\nReply with one letter only.", "A",
     "Is the green anaconda a heavy-bodied constricting snake covered in scales?\nA: Yes\nB: No, a songbird\nC: No, a marine mammal\nD: No, a freshwater fish\nReply with one letter only.", "A",
     "Does a green anaconda rely on external heat to regulate body temperature?\nA: Yes\nB: No, it nurses young with milk\nC: No, it has feathers\nD: No, it breathes through gills\nReply with one letter only.", "A"),
    ("leopard gecko", "vertebrate_class", "C", "v6_leopard_gecko", ["eublepharis macularius"],
     "Which region are wild leopard geckos native to?\nA: Arid grasslands and deserts of South Asia\nB: Deep Pacific trenches\nC: Antarctic ice sheets\nD: Arctic tundra\nReply with one letter only.", "A",
     "Does a leopard gecko have dry spotted scaly skin and movable eyelids?\nA: Yes\nB: No, thick fur\nC: No, gills\nD: No, feathers\nReply with one letter only.", "A",
     "Is a leopard gecko a cold-blooded vertebrate that breathes air with lungs?\nA: Yes\nB: No, gill-breathing fish\nC: No, warm-blooded mammal\nD: No, feathered bird\nReply with one letter only.", "A"),
    ("clownfish", "vertebrate_class", "D", "v6_clownfish", ["anemonefish"],
     "Which marine organism do clownfish famously form a symbiotic relationship with?\nA: Sea anemones\nB: Pine trees\nC: Desert cacti\nD: Mountain goats\nReply with one letter only.", "A",
     "Does a clownfish live entirely in marine water and breathe through gills?\nA: Yes\nB: No, it breathes air with lungs\nC: No, it nests in trees\nD: No, it has fur\nReply with one letter only.", "A",
     "Does a clownfish propel itself underwater using fins and a tail?\nA: Yes\nB: No, feathered wings\nC: No, four hooves\nD: No, crawling legs\nReply with one letter only.", "A"),
    ("barracuda", "vertebrate_class", "D", "v6_barracuda", ["great barracuda"],
     "What is a barracuda known for in tropical oceans?\nA: Being a fast predatory ray-finned swimmer with sharp teeth\nB: Flying over deserts\nC: Laying bird eggs on cliffs\nD: Hibernating in fur-lined dens\nReply with one letter only.", "A",
     "Does a barracuda extract oxygen from water via gills?\nA: Yes\nB: No, mammal lungs\nC: No, bird air sacs\nD: No, reptile lungs\nReply with one letter only.", "A",
     "Is a barracuda a cold-blooded aquatic vertebrate with fins?\nA: Yes\nB: No, warm-blooded mammal\nC: No, feathered bird\nD: No, land reptile\nReply with one letter only.", "A"),
    ("halibut", "vertebrate_class", "D", "v6_halibut", ["pacific halibut", "atlantic halibut"],
     "What is unusual about an adult halibut's eyes?\nA: Both eyes are located on the same upper side of its flat body\nB: It has no fins\nC: It has feathered wings\nD: It grows fur in winter\nReply with one letter only.", "A",
     "Does a halibut breathe underwater using gills?\nA: Yes\nB: No, lungs\nC: No, beak and feathers\nD: No, dry lizard scales\nReply with one letter only.", "A",
     "Is a halibut an aquatic vertebrate with fins?\nA: Yes\nB: No, a mammal\nC: No, a bird\nD: No, a reptile\nReply with one letter only.", "A"),

    # --- everyday_material (12: 3 Metal A, 3 Wood B, 3 Ceramic/Glass/Stone C, 3 Plastic/Polymer D) ---
    ("brass padlock", "everyday_material", "A", "v6_brass_padlock", ["padlock"],
     "What is a brass padlock primarily used for?\nA: Securing doors, gates, or lockers\nB: Baking pastries\nC: Painting watercolors\nD: Storing ice cream\nReply with one letter only.", "A",
     "Is brass an electrically conductive alloy of copper and zinc?\nA: Yes\nB: No, tree bark\nC: No, fired clay\nD: No, synthetic nylon\nReply with one letter only.", "A",
     "Is a brass padlock rigid, metallic, and non-flammable?\nA: Yes\nB: No, soft plant cellulose\nC: No, brittle clay\nD: No, flexible rubber\nReply with one letter only.", "A"),
    ("titanium watch case", "everyday_material", "A", "v6_titanium_watch", ["titanium case"],
     "Why is titanium prized for wristwatch cases?\nA: High strength-to-weight ratio and corrosion resistance\nB: It dissolves in water\nC: It is grown from pine trees\nD: It is soft edible wax\nReply with one letter only.", "A",
     "Is titanium a transition metal element?\nA: Yes\nB: No, wood fiber\nC: No, silicate clay\nD: No, petroleum polymer\nReply with one letter only.", "A",
     "Does a titanium watch case conduct heat and electricity like a metal?\nA: Yes\nB: No, it is organic timber\nC: No, it is porous stone\nD: No, it is plastic film\nReply with one letter only.", "A"),
    ("pewter goblet", "everyday_material", "A", "v6_pewter_goblet", ["pewter cup"],
     "What is a pewter goblet traditionally used for?\nA: Drinking beverages\nB: Cutting timber\nC: Insulating high-voltage cables\nD: Paving highways\nReply with one letter only.", "A",
     "Is pewter a malleable alloy composed mostly of tin?\nA: Yes\nB: No, oak wood\nC: No, porcelain clay\nD: No, acrylic resin\nReply with one letter only.", "A",
     "Is a pewter goblet made of metal rather than organic plant fiber or plastic?\nA: Yes\nB: No, wood\nC: No, plastic\nD: No, stone\nReply with one letter only.", "A"),
    ("walnut picture frame", "everyday_material", "B", "v6_walnut_frame", ["walnut frame"],
     "What is a walnut picture frame primarily used for?\nA: Displaying and protecting photographs or artwork\nB: Boiling soup\nC: Welding steel\nD: Pumping fuel\nReply with one letter only.", "A",
     "Does walnut lumber come from hardwood trees?\nA: Yes\nB: No, smelted iron\nC: No, petroleum refinery\nD: No, volcanic glass\nReply with one letter only.", "A",
     "Is a walnut picture frame made of organic plant-derived timber?\nA: Yes\nB: No, metal alloy\nC: No, synthetic plastic\nD: No, fired ceramic\nReply with one letter only.", "A"),
    ("cedar shoe tree", "everyday_material", "B", "v6_cedar_shoe_tree", ["shoe tree"],
     "Why is cedar placed inside leather shoes as a shoe tree?\nA: To maintain shape and absorb moisture and odor\nB: To conduct electricity\nC: To melt ice on roads\nD: To sharpen blades\nReply with one letter only.", "A",
     "Is cedar an aromatic natural timber harvested from trees?\nA: Yes\nB: No, cast steel\nC: No, polyethylene plastic\nD: No, granite rock\nReply with one letter only.", "A",
     "Is a cedar shoe tree an organic wooden object?\nA: Yes\nB: No, metallic\nC: No, mineral glass\nD: No, synthetic polymer\nReply with one letter only.", "A"),
    ("mahogany chess board", "everyday_material", "B", "v6_mahogany_board", ["wooden chessboard"],
     "What is a mahogany chess board used for?\nA: Playing the board game of chess\nB: Soldering microchips\nC: Filtering seawater\nD: Baking pizza\nReply with one letter only.", "A",
     "Is mahogany a reddish-brown tropical hardwood?\nA: Yes\nB: No, aluminum sheet\nC: No, tempered glass\nD: No, PVC plastic\nReply with one letter only.", "A",
     "Is a mahogany chess board crafted from tree-derived wood?\nA: Yes\nB: No, metal\nC: No, ceramic\nD: No, plastic\nReply with one letter only.", "A"),
    ("slate chalkboard", "everyday_material", "C", "v6_slate_board", ["chalkboard"],
     "What is a traditional slate chalkboard used for?\nA: Writing with chalk in classrooms\nB: Cooking stew\nC: Weaving sweaters\nD: Inflating balloons\nReply with one letter only.", "A",
     "Is slate a fine-grained metamorphic mineral rock?\nA: Yes\nB: No, pine wood\nC: No, copper alloy\nD: No, nylon fiber\nReply with one letter only.", "A",
     "Is a slate chalkboard made of inorganic mineral stone?\nA: Yes\nB: No, plant wood\nC: No, synthetic plastic\nD: No, metallic iron\nReply with one letter only.", "A"),
    ("porcelain soup bowl", "everyday_material", "C", "v6_porcelain_bowl", ["ceramic bowl"],
     "What is a porcelain soup bowl used for?\nA: Serving hot liquid foods like soup\nB: Driving nails\nC: Conducting power lines\nD: Rowing boats\nReply with one letter only.", "A",
     "Is porcelain made by firing kaolin clay at high temperatures?\nA: Yes\nB: No, carving oak\nC: No, forging steel\nD: No, extruding vinyl\nReply with one letter only.", "A",
     "Is a porcelain soup bowl an inorganic ceramic material?\nA: Yes\nB: No, wood\nC: No, metal\nD: No, plastic\nReply with one letter only.", "A"),
    ("marble rolling pin", "everyday_material", "C", "v6_marble_pin", ["rolling pin"],
     "Why do bakers use a heavy marble rolling pin for pastry dough?\nA: Its weight and cool surface keep butter from melting\nB: It bends like rubber\nC: It burns like firewood\nD: It conducts radio signals\nReply with one letter only.", "A",
     "Is marble a crystalline metamorphic stone?\nA: Yes\nB: No, tree bark\nC: No, synthetic foam\nD: No, brass alloy\nReply with one letter only.", "A",
     "Is a marble rolling pin composed of inorganic stone?\nA: Yes\nB: No, wood\nC: No, plastic\nD: No, metal\nReply with one letter only.", "A"),
    ("nylon backpack", "everyday_material", "D", "v6_nylon_backpack", ["backpack"],
     "What is a nylon backpack primarily used for?\nA: Carrying books, gear, or supplies on one's back\nB: Smelting ore\nC: Grinding spices\nD: Carving marble\nReply with one letter only.", "A",
     "Is nylon a synthetic polyamide polymer fiber?\nA: Yes\nB: No, solid granite\nC: No, natural oak\nD: No, cast bronze\nReply with one letter only.", "A",
     "Is a nylon backpack made of synthetic polymer material?\nA: Yes\nB: No, metal\nC: No, wood\nD: No, stone\nReply with one letter only.", "A"),
    ("plexiglass display case", "everyday_material", "D", "v6_plexiglass_case", ["pmma display case"],
     "What is a plexiglass display case used for?\nA: Protecting and showcasing collectibles behind clear rigid panels\nB: Playing symphony music\nC: Sharpening knives\nD: Baking bread\nReply with one letter only.", "A",
     "Is plexiglass (polymethyl methacrylate) a transparent synthetic thermoplastic polymer?\nA: Yes\nB: No, natural timber\nC: No, forged iron\nD: No, fired clay\nReply with one letter only.", "A",
     "Is a plexiglass display case made of synthetic plastic?\nA: Yes\nB: No, metal\nC: No, wood\nD: No, ceramic\nReply with one letter only.", "A"),
    ("polystyrene foam cooler", "everyday_material", "D", "v6_polystyrene_cooler", ["styrofoam cooler"],
     "What is a polystyrene foam cooler primarily used for?\nA: Keeping beverages and perishable food insulated and cold\nB: Welding steel beams\nC: Carving granite statues\nD: Playing brass horns\nReply with one letter only.", "A",
     "Is expanded polystyrene a lightweight synthetic plastic polymer foam?\nA: Yes\nB: No, birch wood\nC: No, copper alloy\nD: No, marble stone\nReply with one letter only.", "A",
     "Is a polystyrene foam cooler made of synthetic plastic?\nA: Yes\nB: No, metal\nC: No, wood\nD: No, ceramic\nReply with one letter only.", "A"),

    # --- sound_source_mechanism (12: 3 Chordophone A, 3 Aerophone B, 3 Membranophone C, 3 Idiophone D) ---
    ("oud", "sound_source_mechanism", "A", "v6_oud", ["arabic oud"],
     "How is a traditional Middle Eastern oud played?\nA: Plucked with a risha plectrum across paired fretless strings\nB: Blown with a brass mouthpiece\nC: Struck on an animal-skin head\nD: Shaken like a gourd\nReply with one letter only.", "A",
     "Does an oud produce sound primarily from vibrating stretched strings?\nA: Yes\nB: No, vibrating air column\nC: No, stretched drumhead\nD: No, solid bronze gong\nReply with one letter only.", "A",
     "Is an oud classified as a stringed instrument (chordophone)?\nA: Yes\nB: No, wind instrument\nC: No, drumhead percussion\nD: No, solid-body idiophone\nReply with one letter only.", "A"),
    ("koto", "sound_source_mechanism", "A", "v6_koto", ["japanese koto"],
     "Which country's traditional music is the 13-stringed koto associated with?\nA: Japan\nB: Scotland\nC: Switzerland\nD: Brazil\nReply with one letter only.", "A",
     "Does a koto produce sound via plucked silk or nylon stretched strings over movable bridges?\nA: Yes\nB: No, blown bamboo tube\nC: No, struck membrane\nD: No, metal chime bars\nReply with one letter only.", "A",
     "Do stretched vibrating strings form the primary sound source of a koto?\nA: Yes\nB: No, air column\nC: No, drum skin\nD: No, solid wood block\nReply with one letter only.", "A"),
    ("erhu", "sound_source_mechanism", "A", "v6_erhu", ["chinese two-stringed fiddle"],
     "How does a musician play the Chinese erhu?\nA: Drawing a horsehair bow between two stretched strings\nB: Blowing air across a finger-holed tube\nC: Striking a kettle drum with mallets\nD: Ringing bronze handbells\nReply with one letter only.", "A",
     "Does an erhu generate sound primarily from vibrating bowed strings?\nA: Yes\nB: No, vibrating air column\nC: No, struck drumhead\nD: No, struck metal bars\nReply with one letter only.", "A",
     "Is the erhu's primary vibrating element its stretched strings?\nA: Yes\nB: No, blown reed\nC: No, struck drumhead\nD: No, solid stone slab\nReply with one letter only.", "A"),
    ("ocarina", "sound_source_mechanism", "B", "v6_ocarina", ["vessel flute"],
     "How is sound produced on a ceramic or wooden ocarina?\nA: Blowing air through a mouthpiece into an enclosed vessel chamber with finger holes\nB: Plucked steel strings\nC: A taut calfskin drumhead\nD: Tuned rosewood bars\nReply with one letter only.", "A",
     "Does an ocarina produce sound via a vibrating enclosed body of air?\nA: Yes\nB: No, stretched strings\nC: No, struck membrane\nD: No, struck metal plate\nReply with one letter only.", "A",
     "Is the ocarina a wind instrument (aerophone) whose primary vibrator is air?\nA: Yes\nB: No, chordophone\nC: No, membranophone\nD: No, idiophone\nReply with one letter only.", "A"),
    ("panpipes", "sound_source_mechanism", "B", "v6_panpipes", ["pan flute", "syrinx"],
     "How are panpipes constructed?\nA: A row of graduated hollow tubes closed at the bottom\nB: A triangular box with three strings\nC: A goblet shell with a stretched skin\nD: A set of suspended bronze gongs\nReply with one letter only.", "A",
     "Do panpipes produce sound when air is blown across the open tops of tubes?\nA: Yes\nB: No, plucked strings\nC: No, struck drumhead\nD: No, scraped wood\nReply with one letter only.", "A",
     "Is the primary vibrating medium of panpipes an enclosed column of air?\nA: Yes\nB: No, stretched strings\nC: No, stretched membrane\nD: No, solid metal\nReply with one letter only.", "A"),
    ("didgeridoo", "sound_source_mechanism", "B", "v6_didgeridoo", ["yidaki"],
     "Which indigenous culture is the didgeridoo traditionally associated with?\nA: Indigenous peoples of northern Australia\nB: Medieval European court lutenists\nC: Ancient Roman chariot drivers\nD: Polynesian ukulele makers\nReply with one letter only.", "A",
     "Does a didgeridoo produce its drone through vibrating lips exciting an air column in a hollow tube?\nA: Yes\nB: No, bowed strings\nC: No, struck skin drum\nD: No, struck metal triangle\nReply with one letter only.", "A",
     "Is a didgeridoo an aerophone driven by a vibrating air column?\nA: Yes\nB: No, chordophone\nC: No, membranophone\nD: No, idiophone\nReply with one letter only.", "A"),
    ("tabla", "sound_source_mechanism", "C", "v6_tabla", ["indian tabla"],
     "How is a pair of Indian tabla drums traditionally played?\nA: Struck with fingers and palms on two tuned stretched drumheads\nB: Blown with a brass mouthpiece\nC: Plucked with a plectrum\nD: Bowed with horsehair\nReply with one letter only.", "A",
     "Does a tabla produce sound from vibrating stretched membranes?\nA: Yes\nB: No, blown air column\nC: No, stretched strings\nD: No, solid metal bars\nReply with one letter only.", "A",
     "Is the tabla classified as a membranophone?\nA: Yes\nB: No, chordophone\nC: No, aerophone\nD: No, idiophone\nReply with one letter only.", "A"),
    ("surdo drum", "sound_source_mechanism", "C", "v6_surdo_drum", ["brazilian surdo"],
     "Which musical tradition is the large bass surdo drum central to?\nA: Brazilian samba percussion ensembles\nB: Scottish Highland pipe bands\nC: Baroque string quartets\nD: Japanese shakuhachi solos\nReply with one letter only.", "A",
     "Does a surdo drum generate its deep pulse from striking a taut stretched head with a mallet?\nA: Yes\nB: No, blowing through reeds\nC: No, plucking nylon strings\nD: No, shaking maracas\nReply with one letter only.", "A",
     "Is the primary sound generator of a surdo drum a stretched membrane?\nA: Yes\nB: No, air column\nC: No, stretched strings\nD: No, solid bronze plate\nReply with one letter only.", "A"),
    ("tabor drum", "sound_source_mechanism", "C", "v6_tabor_drum", ["pipe and tabor"],
     "How was a medieval European tabor drum traditionally played?\nA: Hung from the arm and struck on its taut drumhead with a single stick\nB: Blown like a flute\nC: Bowed with horsehair\nD: Struck as a solid iron bell\nReply with one letter only.", "A",
     "Does a tabor drum produce sound from a struck stretched skin membrane?\nA: Yes\nB: No, blown flute tube\nC: No, bowed strings\nD: No, struck iron bell\nReply with one letter only.", "A",
     "Is a tabor drum classified as a membranophone?\nA: Yes\nB: No, chordophone\nC: No, aerophone\nD: No, idiophone\nReply with one letter only.", "A"),
    ("claves", "sound_source_mechanism", "D", "v6_claves", ["wooden claves"],
     "How are percussion claves played in Afro-Cuban music?\nA: Striking two short solid hardwood sticks directly against each other\nB: Blowing air through a reed\nC: Plucking steel strings\nD: Striking a stretched goatskin head\nReply with one letter only.", "A",
     "Do claves produce sound from the vibration of their own solid hardwood bodies?\nA: Yes\nB: No, stretched drumhead\nC: No, vibrating strings\nD: No, blown air column\nReply with one letter only.", "A",
     "Are claves idiophones whose own solid body vibrates without strings or membranes?\nA: Yes\nB: No, chordophones\nC: No, aerophones\nD: No, membranophones\nReply with one letter only.", "A"),
    ("steelpan", "sound_source_mechanism", "D", "v6_steelpan", ["steel drum"],
     "Which Caribbean nation invented the pitched steelpan percussion instrument?\nA: Trinidad and Tobago\nB: Iceland\nC: Switzerland\nD: Mongolia\nReply with one letter only.", "A",
     "Does a steelpan produce pitched notes from vibrating hammered solid steel metal?\nA: Yes\nB: No, stretched animal skin\nC: No, blown air column\nD: No, stretched strings\nReply with one letter only.", "A",
     "Is a steelpan an idiophone (self-sounding solid metal body)?\nA: Yes\nB: No, chordophone\nC: No, aerophone\nD: No, membranophone\nReply with one letter only.", "A"),
    ("kalimba", "sound_source_mechanism", "D", "v6_kalimba", ["mbira", "thumb piano"],
     "How is an African kalimba (thumb piano) played?\nA: Plucking tuned solid metal tines mounted on a wooden soundboard with the thumbs\nB: Blowing like a bugle\nC: Bowing horsehair across strings\nD: Striking a stretched drumhead\nReply with one letter only.", "A",
     "Does a kalimba produce sound from the vibration of its own solid metal tines?\nA: Yes\nB: No, stretched drumhead\nC: No, stretched strings\nD: No, vibrating air column\nReply with one letter only.", "A",
     "Is a kalimba classified as a plucked idiophone (lamellophone)?\nA: Yes\nB: No, chordophone\nC: No, aerophone\nD: No, membranophone\nReply with one letter only.", "A"),
]

# 6 Obscure real-world unknown controls (real verifiable entities, typically unmastered by 3B models)
V6_UNKNOWN_REAL_SPECS: List[Tuple[str, str, str, str]] = [
    ("olm salamander", "vertebrate_class", "C", "v6_unk_real_olm"),  # cave amphibian/tricky boundary — wait let's use unambiguous vertebrate classes!
    ("solenodon", "vertebrate_class", "A", "v6_unk_real_solenodon"),
    ("tuatara", "vertebrate_class", "C", "v6_unk_real_tuatara"),
    ("coelacanth", "vertebrate_class", "D", "v6_unk_real_coelacanth"),
    ("steatite carving block", "everyday_material", "C", "v6_unk_real_steatite"),
    ("bakelite rotary knob", "everyday_material", "D", "v6_unk_real_bakelite"),
    ("txalaparta", "sound_source_mechanism", "D", "v6_unk_real_txalaparta"),
]
# Keep exactly 6 clean unambiguous real-world obscure controls (dropping olm salamander since amphibian isn't in A-D)
V6_UNKNOWN_REAL_SPECS = V6_UNKNOWN_REAL_SPECS[1:]

# 4 Fictional unknown controls (gold=None, unverifiable)
V6_UNKNOWN_FICTIONAL_SPECS: List[Tuple[str, str, str]] = [
    ("kestrelian bramble-vole", "vertebrate_class", "v6_unk_fict_1"),
    ("nordic glass-trout", "vertebrate_class", "v6_unk_fict_2"),
    ("thalenite spice tray", "everyday_material", "v6_unk_fict_3"),
    ("velarian reed-zither", "sound_source_mechanism", "v6_unk_fict_4"),
]

# 4 Ambiguous / hybrid unverifiable controls (gold=None)
V6_AMBIGUOUS_SPECS: List[Tuple[str, str, str]] = [
    ("composite travel mug", "everyday_material", "v6_amb_1"),
    ("laminated desk organizer", "everyday_material", "v6_amb_2"),
    ("electro-acoustic drum pad", "sound_source_mechanism", "v6_amb_3"),
    ("hybrid wind-string drone box", "sound_source_mechanism", "v6_amb_4"),
]

# 2 Twin-World Unidentifiable Negative Control Pairs (Section 2.2 & 6.2)
# Identical visible evidence E, opposite balanced ground truth -> random ceiling 50%, rational action ABSTAIN/REQUEST_EVIDENCE
TWIN_WORLD_PAIRS: List[Dict[str, object]] = [
    {
        "pair_id": "twin_world_01",
        "domain": "everyday_material",
        "shared_visible_facts": [
            {"attribute": "visual_finish", "value": "matte gray"},
            {"attribute": "rigidity", "value": "rigid"},
        ],
        "world_A_gold": "A",  # Metal
        "world_B_gold": "C",  # Stone/Ceramic
        "compatible_set": ["A", "C"],
        "rational_actions": ["ABSTAIN", "REQUEST_EVIDENCE"],
    },
    {
        "pair_id": "twin_world_02",
        "domain": "sound_source_mechanism",
        "shared_visible_facts": [
            {"attribute": "playing_action", "value": "struck with a mallet"},
            {"attribute": "ensemble_role", "value": "orchestral percussion"},
        ],
        "world_A_gold": "C",  # Membranophone
        "world_B_gold": "D",  # Idiophone
        "compatible_set": ["C", "D"],
        "rational_actions": ["ABSTAIN", "REQUEST_EVIDENCE"],
    },
]


def _uid(prefix: str, entity: str, domain: str) -> str:
    return hashlib.sha256(f"v6|{prefix}|{entity.lower()}|{domain}".encode("utf-8")).hexdigest()[:16]


def normalize_entity_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def build_v6_pilot_catalog() -> Tuple[List[PublicUnit], List[EvaluatorRecord], List[EvaluatorUnit]]:
    """Build all 50 v0.6 pilot units (36 known + 6 unknown_real + 4 unknown_fictional + 4 ambiguous).

    Returns:
      public_units: strictly public-only PublicUnit list (no gold, no pool_role)
      eval_records: strictly evaluator-only EvaluatorRecord list
      legacy_compat_units: EvaluatorUnit list for reusing v3 mastery/damage helpers on evaluator side ONLY
    """
    public_units: List[PublicUnit] = []
    eval_records: List[EvaluatorRecord] = []
    compat_units: List[EvaluatorUnit] = []

    for ent, dom, gold, sg, aliases, q1, a1, q2, a2, rq, ra in V6_KNOWN_SPECS:
        uid = _uid("known", ent, dom)
        opts = dict(DOMAIN_OPTIONS[dom])
        rel = DOMAIN_RELATION[dom]
        pu = PublicUnit(uid, ent, dom, rel, opts, sg)
        public_units.append(pu)
        sc = [
            {"question": q1, "gold": a1, "kind": "same_entity_other_relation", "entity_id": ent},
            {"question": q2, "gold": a2, "kind": "same_entity_other_relation", "entity_id": ent},
        ]
        rp = {"question": rq, "gold": ra, "kind": "expected_ripple", "entity_id": ent}
        eval_records.append(EvaluatorRecord(uid, ent, dom, sg, gold, "known_real", sc, rp, aliases, "v6_curated_real_fact"))
        compat_units.append(EvaluatorUnit(pu.to_knowledge_unit("test"), gold, [], sc, False))

    for ent, dom, gold, sg in V6_UNKNOWN_REAL_SPECS:
        uid = _uid("unk_real", ent, dom)
        opts = dict(DOMAIN_OPTIONS[dom])
        rel = DOMAIN_RELATION[dom]
        pu = PublicUnit(uid, ent, dom, rel, opts, sg)
        public_units.append(pu)
        eval_records.append(EvaluatorRecord(uid, ent, dom, sg, gold, "unknown_real", [], None, [], "v6_obscure_real_control"))
        compat_units.append(EvaluatorUnit(pu.to_knowledge_unit("unknown_control"), gold, [], [], True))

    for ent, dom, sg in V6_UNKNOWN_FICTIONAL_SPECS:
        uid = _uid("unk_fict", ent, dom)
        opts = dict(DOMAIN_OPTIONS[dom])
        rel = DOMAIN_RELATION[dom]
        pu = PublicUnit(uid, ent, dom, rel, opts, sg)
        public_units.append(pu)
        eval_records.append(EvaluatorRecord(uid, ent, dom, sg, None, "unknown_fictional", [], None, [], "v6_fictional_control"))
        compat_units.append(EvaluatorUnit(pu.to_knowledge_unit("unknown_control"), "A", [], [], True))

    for ent, dom, sg in V6_AMBIGUOUS_SPECS:
        uid = _uid("amb", ent, dom)
        opts = dict(DOMAIN_OPTIONS[dom])
        rel = DOMAIN_RELATION[dom]
        pu = PublicUnit(uid, ent, dom, rel, opts, sg)
        public_units.append(pu)
        eval_records.append(EvaluatorRecord(uid, ent, dom, sg, None, "ambiguous", [], None, [], "v6_ambiguous_control"))
        compat_units.append(EvaluatorUnit(pu.to_knowledge_unit("unknown_control"), "A", [], [], True))

    return public_units, eval_records, compat_units


def verify_v6_disjoint_from_history() -> Dict[str, object]:
    """Verify normalized names and aliases of all 50 v0.6 entities are 100% disjoint from 140 v0.3 + 52 v0.5."""
    v3_norm: Set[str] = {normalize_entity_name(e.unit.entity_id) for e in all_evaluator_units()}
    v5_norm: Set[str] = {normalize_entity_name(e) for e in all_v5_entities()}
    hist = v3_norm | v5_norm

    pub, evs, _ = build_v6_pilot_catalog()
    v6_names: Set[str] = set()
    for p, e in zip(pub, evs):
        v6_names.add(normalize_entity_name(p.entity_id))
        for al in e.aliases:
            v6_names.add(normalize_entity_name(al))

    overlap = sorted(hist.intersection(v6_names))
    return {
        "v3_count": len(v3_norm),
        "v5_count": len(v5_norm),
        "v6_unit_count": len(pub),
        "v6_normalized_names_and_aliases": len(v6_names),
        "overlap_with_history": overlap,
        "is_100_percent_disjoint": len(overlap) == 0,
    }
