"""Entity-disjoint and source-group-disjoint commonsense knowledge registry for KRMM v0.3."""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple

from .schemas import EvaluatorUnit, KnowledgeUnit

DOMAIN_OPTIONS: Dict[str, Dict[str, str]] = {
    "vertebrate_class": {
        "A": "mammal",
        "B": "bird",
        "C": "reptile",
        "D": "fish",
    },
    "everyday_material": {
        "A": "metal (metallic alloy or pure metal)",
        "B": "wood or plant-based fiber (cellulose or timber)",
        "C": "stone, ceramic, or glass (mineral or silicate)",
        "D": "synthetic polymer or elastomer (plastic or rubber)",
    },
    "sound_source_mechanism": {
        "A": "chordophone (vibrating stretched string)",
        "B": "aerophone (vibrating air column or air-driven reed)",
        "C": "membranophone (vibrating stretched membrane)",
        "D": "idiophone (vibrating solid body itself)",
    },
}

DOMAIN_RELATIONS: Dict[str, str] = {
    "vertebrate_class": "taxonomic_class",
    "everyday_material": "primary_material",
    "sound_source_mechanism": "hornbostel_sachs_class",
}

DOMAIN_TEMPLATES: Dict[str, List[str]] = {
    "vertebrate_class": [
        "Which vertebrate taxonomic class does the {entity} belong to?",
        "Classify the {entity} into its biological vertebrate class.",
        "From a zoological perspective, what class of vertebrate is the {entity}?",
        "Assign the {entity} to the correct vertebrate group.",
        "A biology textbook lists the {entity} under which vertebrate class?",
        "Identify the zoological vertebrate class of a typical {entity}.",
    ],
    "everyday_material": [
        "What primary material is a standard {entity} made of?",
        "Classify the typical {entity} by its primary structural material.",
        "Which material category best describes a conventional {entity}?",
        "In materials classification, what is a {entity} primarily composed of?",
        "A hardware inventory groups items by composition. Which material describes a {entity}?",
        "Identify the main constituent material of a traditional {entity}.",
    ],
    "sound_source_mechanism": [
        "Classify the acoustic {entity} in the Hornbostel-Sachs family system.",
        "Which Hornbostel-Sachs family contains the traditional acoustic {entity}?",
        "For a {entity}, choose its organological family based on its sound-producing mechanism.",
        "Assign the acoustic {entity} to one of the four Hornbostel-Sachs families.",
        "A music encyclopedia categorizes a traditional {entity} under which acoustic family?",
        "What is the top-level organological class of an acoustic {entity} under Hornbostel-Sachs?",
    ],
}

DIET_OPTIONS = {
    "A": "herbivore (plant-eater)",
    "B": "carnivore (meat or prey-eater)",
    "C": "omnivore (mixed plant and animal diet)",
    "D": "chemoautotroph (inorganic chemical synthesizer)",
}

FUNCTION_OPTIONS = {
    "A": "kitchen cooking, dining, or food preparation",
    "B": "construction, plumbing, or mechanical hardware",
    "C": "clothing, bedding, or wearable textiles",
    "D": "writing, art, packaging, or optical science",
}

PITCH_OPTIONS = {
    "A": "produces definite musical pitches for melody or harmony",
    "B": "produces mainly indefinite unpitched percussion sound",
    "C": "electronic digital synthesizer with no acoustic resonator",
    "D": "wind-powered weathervane without musical function",
}

RAW_CATALOG: List[Tuple[str, str, str, str, str, Dict[str, str], str, str]] = [
    # ==================== VERTEBRATE CLASS ====================
    # --- dev (12: 3A, 3B, 3C, 3D) ---
    ("african elephant", "vertebrate_class", "vert_proboscidean", "dev", "A", {"thermoregulation": "warm_blooded", "reproduction": "live_birth", "respiration": "lungs", "covering": "hair_skin"}, "diet", "A"),
    ("bottlenose dolphin", "vertebrate_class", "vert_cetacean", "dev", "A", {"thermoregulation": "warm_blooded", "reproduction": "live_birth", "respiration": "lungs", "appendage": "fins"}, "diet", "B"),
    ("gray wolf", "vertebrate_class", "vert_canid", "dev", "A", {"thermoregulation": "warm_blooded", "reproduction": "live_birth", "respiration": "lungs", "covering": "fur"}, "diet", "B"),
    ("bald eagle", "vertebrate_class", "vert_accipitrid", "dev", "B", {"thermoregulation": "warm_blooded", "reproduction": "hard_shelled_eggs", "respiration": "lungs", "covering": "feathers"}, "diet", "B"),
    ("emperor penguin", "vertebrate_class", "vert_spheniscid", "dev", "B", {"thermoregulation": "warm_blooded", "reproduction": "hard_shelled_eggs", "respiration": "lungs", "covering": "feathers"}, "diet", "B"),
    ("barn owl", "vertebrate_class", "vert_strigid", "dev", "B", {"thermoregulation": "warm_blooded", "reproduction": "hard_shelled_eggs", "respiration": "lungs", "covering": "feathers"}, "diet", "B"),
    ("nile crocodile", "vertebrate_class", "vert_crocodilian", "dev", "C", {"thermoregulation": "cold_blooded", "reproduction": "eggs", "respiration": "lungs", "covering": "scales"}, "diet", "B"),
    ("king cobra", "vertebrate_class", "vert_elapid", "dev", "C", {"thermoregulation": "cold_blooded", "reproduction": "eggs", "respiration": "lungs", "covering": "scales"}, "diet", "B"),
    ("galapagos tortoise", "vertebrate_class", "vert_testudinid", "dev", "C", {"thermoregulation": "cold_blooded", "reproduction": "eggs", "respiration": "lungs", "covering": "scales"}, "diet", "A"),
    ("atlantic salmon", "vertebrate_class", "vert_salmonid", "dev", "D", {"thermoregulation": "cold_blooded", "reproduction": "aquatic_eggs", "respiration": "gills", "appendage": "fins"}, "diet", "B"),
    ("great white shark", "vertebrate_class", "vert_lamnid", "dev", "D", {"thermoregulation": "cold_blooded", "reproduction": "live_birth", "respiration": "gills", "appendage": "fins"}, "diet", "B"),
    ("bluefin tuna", "vertebrate_class", "vert_scombrid", "dev", "D", {"thermoregulation": "cold_blooded", "reproduction": "aquatic_eggs", "respiration": "gills", "appendage": "fins"}, "diet", "B"),
    # --- calibration (12: 3A, 3B, 3C, 3D) ---
    ("african lion", "vertebrate_class", "vert_felid", "calibration", "A", {"thermoregulation": "warm_blooded", "reproduction": "live_birth", "respiration": "lungs", "covering": "fur"}, "diet", "B"),
    ("giraffe", "vertebrate_class", "vert_giraffid", "calibration", "A", {"thermoregulation": "warm_blooded", "reproduction": "live_birth", "respiration": "lungs", "covering": "fur"}, "diet", "A"),
    ("red kangaroo", "vertebrate_class", "vert_macropod", "calibration", "A", {"thermoregulation": "warm_blooded", "reproduction": "live_birth", "respiration": "lungs", "covering": "fur"}, "diet", "A"),
    ("peregrine falcon", "vertebrate_class", "vert_falconid", "calibration", "B", {"thermoregulation": "warm_blooded", "reproduction": "hard_shelled_eggs", "respiration": "lungs", "covering": "feathers"}, "diet", "B"),
    ("atlantic puffin", "vertebrate_class", "vert_alcid", "calibration", "B", {"thermoregulation": "warm_blooded", "reproduction": "hard_shelled_eggs", "respiration": "lungs", "covering": "feathers"}, "diet", "B"),
    ("great blue heron", "vertebrate_class", "vert_ardeid", "calibration", "B", {"thermoregulation": "warm_blooded", "reproduction": "hard_shelled_eggs", "respiration": "lungs", "covering": "feathers"}, "diet", "B"),
    ("komodo dragon", "vertebrate_class", "vert_varanid", "calibration", "C", {"thermoregulation": "cold_blooded", "reproduction": "eggs", "respiration": "lungs", "covering": "scales"}, "diet", "B"),
    ("green iguana", "vertebrate_class", "vert_iguanid", "calibration", "C", {"thermoregulation": "cold_blooded", "reproduction": "eggs", "respiration": "lungs", "covering": "scales"}, "diet", "A"),
    ("burmese python", "vertebrate_class", "vert_pythonid", "calibration", "C", {"thermoregulation": "cold_blooded", "reproduction": "eggs", "respiration": "lungs", "covering": "scales"}, "diet", "B"),
    ("hammerhead shark", "vertebrate_class", "vert_sphyrnid", "calibration", "D", {"thermoregulation": "cold_blooded", "reproduction": "live_birth", "respiration": "gills", "appendage": "fins"}, "diet", "B"),
    ("swordfish", "vertebrate_class", "vert_xiphiid", "calibration", "D", {"thermoregulation": "cold_blooded", "reproduction": "aquatic_eggs", "respiration": "gills", "appendage": "fins"}, "diet", "B"),
    ("largemouth bass", "vertebrate_class", "vert_centrarchid", "calibration", "D", {"thermoregulation": "cold_blooded", "reproduction": "aquatic_eggs", "respiration": "gills", "appendage": "fins"}, "diet", "B"),
    # --- test (16: 4A, 4B, 4C, 4D) ---
    ("bengal tiger", "vertebrate_class", "vert_tiger", "test", "A", {"thermoregulation": "warm_blooded", "reproduction": "live_birth", "respiration": "lungs", "covering": "fur"}, "diet", "B"),
    ("plains zebra", "vertebrate_class", "vert_equid", "test", "A", {"thermoregulation": "warm_blooded", "reproduction": "live_birth", "respiration": "lungs", "covering": "hair_skin"}, "diet", "A"),
    ("koala", "vertebrate_class", "vert_phascolarctid", "test", "A", {"thermoregulation": "warm_blooded", "reproduction": "live_birth", "respiration": "lungs", "covering": "fur"}, "diet", "A"),
    ("humpback whale", "vertebrate_class", "vert_balaenopterid", "test", "A", {"thermoregulation": "warm_blooded", "reproduction": "live_birth", "respiration": "lungs", "appendage": "fins"}, "diet", "B"),
    ("wandering albatross", "vertebrate_class", "vert_diomedeid", "test", "B", {"thermoregulation": "warm_blooded", "reproduction": "hard_shelled_eggs", "respiration": "lungs", "covering": "feathers"}, "diet", "B"),
    ("osprey", "vertebrate_class", "vert_pandionid", "test", "B", {"thermoregulation": "warm_blooded", "reproduction": "hard_shelled_eggs", "respiration": "lungs", "covering": "feathers"}, "diet", "B"),
    ("common kingfisher", "vertebrate_class", "vert_alcedinid", "test", "B", {"thermoregulation": "warm_blooded", "reproduction": "hard_shelled_eggs", "respiration": "lungs", "covering": "feathers"}, "diet", "B"),
    ("american white pelican", "vertebrate_class", "vert_pelecanid", "test", "B", {"thermoregulation": "warm_blooded", "reproduction": "hard_shelled_eggs", "respiration": "lungs", "covering": "feathers"}, "diet", "B"),
    ("gila monster", "vertebrate_class", "vert_helodermatid", "test", "C", {"thermoregulation": "cold_blooded", "reproduction": "eggs", "respiration": "lungs", "covering": "scales"}, "diet", "B"),
    ("boa constrictor", "vertebrate_class", "vert_boid", "test", "C", {"thermoregulation": "cold_blooded", "reproduction": "live_birth", "respiration": "lungs", "covering": "scales"}, "diet", "B"),
    ("veiled chameleon", "vertebrate_class", "vert_chamaeleonid", "test", "C", {"thermoregulation": "cold_blooded", "reproduction": "eggs", "respiration": "lungs", "covering": "scales"}, "diet", "B"),
    ("leatherback sea turtle", "vertebrate_class", "vert_dermochelyid", "test", "C", {"thermoregulation": "cold_blooded", "reproduction": "eggs", "respiration": "lungs", "covering": "scales"}, "diet", "B"),
    ("shortfin mako shark", "vertebrate_class", "vert_isurus", "test", "D", {"thermoregulation": "cold_blooded", "reproduction": "live_birth", "respiration": "gills", "appendage": "fins"}, "diet", "B"),
    ("atlantic cod", "vertebrate_class", "vert_gadid", "test", "D", {"thermoregulation": "cold_blooded", "reproduction": "aquatic_eggs", "respiration": "gills", "appendage": "fins"}, "diet", "B"),
    ("mahi-mahi", "vertebrate_class", "vert_coryphaenid", "test", "D", {"thermoregulation": "cold_blooded", "reproduction": "aquatic_eggs", "respiration": "gills", "appendage": "fins"}, "diet", "B"),
    ("northern pike", "vertebrate_class", "vert_esocid", "test", "D", {"thermoregulation": "cold_blooded", "reproduction": "aquatic_eggs", "respiration": "gills", "appendage": "fins"}, "diet", "B"),
    # --- anchor (4) ---
    ("cheetah", "vertebrate_class", "anchor_vert_a", "anchor", "A", {"thermoregulation": "warm_blooded", "reproduction": "live_birth", "respiration": "lungs"}, "diet", "B"),
    ("red-tailed hawk", "vertebrate_class", "anchor_vert_b", "anchor", "B", {"thermoregulation": "warm_blooded", "reproduction": "hard_shelled_eggs", "respiration": "lungs"}, "diet", "B"),
    ("desert tortoise", "vertebrate_class", "anchor_vert_c", "anchor", "C", {"thermoregulation": "cold_blooded", "reproduction": "eggs", "respiration": "lungs"}, "diet", "A"),
    ("red-bellied piranha", "vertebrate_class", "anchor_vert_d", "anchor", "D", {"thermoregulation": "cold_blooded", "reproduction": "aquatic_eggs", "respiration": "gills"}, "diet", "B"),

    # ==================== EVERYDAY MATERIAL ====================
    # --- dev (12: 3A, 3B, 3C, 3D) ---
    ("stainless steel frying pan", "everyday_material", "mat_steel_pan", "dev", "A", {"origin": "inorganic", "conductivity": "conductor", "heat_behavior": "high_melting"}, "func", "A"),
    ("copper electrical wire", "everyday_material", "mat_copper_wire", "dev", "A", {"origin": "inorganic", "conductivity": "conductor", "heat_behavior": "high_melting"}, "func", "B"),
    ("iron carpentry nail", "everyday_material", "mat_iron_nail", "dev", "A", {"origin": "inorganic", "conductivity": "conductor", "heat_behavior": "high_melting"}, "func", "B"),
    ("wooden cutting board", "everyday_material", "mat_wood_board", "dev", "B", {"origin": "biological_plant", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "A"),
    ("bamboo chopsticks", "everyday_material", "mat_bamboo_chop", "dev", "B", {"origin": "biological_plant", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "A"),
    ("cardboard shipping box", "everyday_material", "mat_cardboard", "dev", "B", {"origin": "biological_plant", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "D"),
    ("porcelain teacup", "everyday_material", "mat_porcelain_cup", "dev", "C", {"origin": "inorganic", "conductivity": "insulator", "heat_behavior": "high_melting"}, "func", "A"),
    ("glass windowpane", "everyday_material", "mat_windowpane", "dev", "C", {"origin": "inorganic", "conductivity": "insulator", "heat_behavior": "high_melting"}, "func", "B"),
    ("magnifying glass lens", "everyday_material", "mat_magnifier", "dev", "C", {"origin": "inorganic", "conductivity": "insulator", "heat_behavior": "high_melting"}, "func", "D"),
    ("PVC plumbing pipe", "everyday_material", "mat_pvc_pipe", "dev", "D", {"origin": "synthetic_petro", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "B"),
    ("silicone baking spatula", "everyday_material", "mat_silicone_spat", "dev", "D", {"origin": "synthetic_petro", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "A"),
    ("nylon raincoat", "everyday_material", "mat_nylon_coat", "dev", "D", {"origin": "synthetic_petro", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "C"),
    # --- calibration (12: 3A, 3B, 3C, 3D) ---
    ("aluminum baking sheet", "everyday_material", "mat_alum_sheet", "calibration", "A", {"origin": "inorganic", "conductivity": "conductor", "heat_behavior": "high_melting"}, "func", "A"),
    ("brass doorknob", "everyday_material", "mat_brass_knob", "calibration", "A", {"origin": "inorganic", "conductivity": "conductor", "heat_behavior": "high_melting"}, "func", "B"),
    ("steel wrench", "everyday_material", "mat_steel_wrench", "calibration", "A", {"origin": "inorganic", "conductivity": "conductor", "heat_behavior": "high_melting"}, "func", "B"),
    ("cotton t-shirt", "everyday_material", "mat_cotton_shirt", "calibration", "B", {"origin": "biological_plant", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "C"),
    ("paper notebook", "everyday_material", "mat_paper_book", "calibration", "B", {"origin": "biological_plant", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "D"),
    ("wooden rolling pin", "everyday_material", "mat_wood_pin", "calibration", "B", {"origin": "biological_plant", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "A"),
    ("ceramic dinner plate", "everyday_material", "mat_ceramic_plate", "calibration", "C", {"origin": "inorganic", "conductivity": "insulator", "heat_behavior": "high_melting"}, "func", "A"),
    ("clay brick", "everyday_material", "mat_clay_brick", "calibration", "C", {"origin": "inorganic", "conductivity": "insulator", "heat_behavior": "high_melting"}, "func", "B"),
    ("borosilicate laboratory beaker", "everyday_material", "mat_glass_beaker", "calibration", "C", {"origin": "inorganic", "conductivity": "insulator", "heat_behavior": "high_melting"}, "func", "D"),
    ("car tire", "everyday_material", "mat_rubber_tire", "calibration", "D", {"origin": "synthetic_petro", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "B"),
    ("polyester fleece jacket", "everyday_material", "mat_poly_jacket", "calibration", "D", {"origin": "synthetic_petro", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "C"),
    ("plastic ballpoint pen casing", "everyday_material", "mat_plastic_pen", "calibration", "D", {"origin": "synthetic_petro", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "D"),
    # --- test (16: 4A, 4B, 4C, 4D) ---
    ("silver soup spoon", "everyday_material", "mat_silver_spoon", "test", "A", {"origin": "inorganic", "conductivity": "conductor", "heat_behavior": "high_melting"}, "func", "A"),
    ("cast-iron Dutch oven", "everyday_material", "mat_cast_iron_pot", "test", "A", {"origin": "inorganic", "conductivity": "conductor", "heat_behavior": "high_melting"}, "func", "A"),
    ("steel hex bolt", "everyday_material", "mat_steel_bolt", "test", "A", {"origin": "inorganic", "conductivity": "conductor", "heat_behavior": "high_melting"}, "func", "B"),
    ("titanium orthopedic screw", "everyday_material", "mat_titanium_screw", "test", "A", {"origin": "inorganic", "conductivity": "conductor", "heat_behavior": "high_melting"}, "func", "B"),
    ("linen bedsheet", "everyday_material", "mat_linen_sheet", "test", "B", {"origin": "biological_plant", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "C"),
    ("kraft paper envelope", "everyday_material", "mat_paper_env", "test", "B", {"origin": "biological_plant", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "D"),
    ("cedarwood pencil shaft", "everyday_material", "mat_cedar_pencil", "test", "B", {"origin": "biological_plant", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "D"),
    ("maple wooden spoon", "everyday_material", "mat_maple_spoon", "test", "B", {"origin": "biological_plant", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "A"),
    ("crystal wine glass", "everyday_material", "mat_wine_glass", "test", "C", {"origin": "inorganic", "conductivity": "insulator", "heat_behavior": "high_melting"}, "func", "A"),
    ("granite kitchen mortar", "everyday_material", "mat_granite_mortar", "test", "C", {"origin": "inorganic", "conductivity": "insulator", "heat_behavior": "high_melting"}, "func", "A"),
    ("terracotta roof tile", "everyday_material", "mat_roof_tile", "test", "C", {"origin": "inorganic", "conductivity": "insulator", "heat_behavior": "high_melting"}, "func", "B"),
    ("optical glass microscope slide", "everyday_material", "mat_micro_slide", "test", "C", {"origin": "inorganic", "conductivity": "insulator", "heat_behavior": "high_melting"}, "func", "D"),
    ("rubber garden hose", "everyday_material", "mat_rubber_hose", "test", "D", {"origin": "synthetic_petro", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "B"),
    ("neoprene diving wetsuit", "everyday_material", "mat_neoprene_suit", "test", "D", {"origin": "synthetic_petro", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "C"),
    ("polyethylene bubble wrap", "everyday_material", "mat_bubble_wrap", "test", "D", {"origin": "synthetic_petro", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "D"),
    ("plastic measuring cup", "everyday_material", "mat_plastic_cup", "test", "D", {"origin": "synthetic_petro", "conductivity": "insulator", "heat_behavior": "low_heat_degrade"}, "func", "A"),
    # --- anchor (4) ---
    ("aluminum soda can", "everyday_material", "anchor_mat_a", "anchor", "A", {"origin": "inorganic", "conductivity": "conductor"}, "func", "A"),
    ("oak barrel", "everyday_material", "anchor_mat_b", "anchor", "B", {"origin": "biological_plant", "conductivity": "insulator"}, "func", "A"),
    ("ceramic mug", "everyday_material", "anchor_mat_c", "anchor", "C", {"origin": "inorganic", "conductivity": "insulator"}, "func", "A"),
    ("latex balloon", "everyday_material", "anchor_mat_d", "anchor", "D", {"origin": "synthetic_petro", "conductivity": "insulator"}, "func", "D"),

    # ==================== SOUND SOURCE MECHANISM ====================
    # --- dev (12: 3A, 3B, 3C, 3D) ---
    ("acoustic guitar", "sound_source_mechanism", "snd_guitar", "dev", "A", {"oscillator": "stretched_string", "excitation": "plucked"}, "pitch", "A"),
    ("concert harp", "sound_source_mechanism", "snd_harp", "dev", "A", {"oscillator": "stretched_string", "excitation": "plucked"}, "pitch", "A"),
    ("cello", "sound_source_mechanism", "snd_cello", "dev", "A", {"oscillator": "stretched_string", "excitation": "bowed"}, "pitch", "A"),
    ("trumpet", "sound_source_mechanism", "snd_trumpet", "dev", "B", {"oscillator": "air_column", "excitation": "lip_buzz_air"}, "pitch", "A"),
    ("clarinet", "sound_source_mechanism", "snd_clarinet", "dev", "B", {"oscillator": "flexible_reed", "excitation": "direct_air_flow"}, "pitch", "A"),
    ("harmonica", "sound_source_mechanism", "snd_harmonica", "dev", "B", {"oscillator": "flexible_reed", "excitation": "direct_air_flow"}, "pitch", "A"),
    ("snare drum", "sound_source_mechanism", "snd_snare", "dev", "C", {"oscillator": "stretched_membrane", "excitation": "struck"}, "pitch", "B"),
    ("bongo drum", "sound_source_mechanism", "snd_bongo", "dev", "C", {"oscillator": "stretched_membrane", "excitation": "struck"}, "pitch", "B"),
    ("conga drum", "sound_source_mechanism", "snd_conga", "dev", "C", {"oscillator": "stretched_membrane", "excitation": "struck"}, "pitch", "B"),
    ("xylophone", "sound_source_mechanism", "snd_xylophone", "dev", "D", {"oscillator": "solid_body", "excitation": "struck"}, "pitch", "A"),
    ("orchestral triangle", "sound_source_mechanism", "snd_triangle", "dev", "D", {"oscillator": "solid_body", "excitation": "struck"}, "pitch", "B"),
    ("crash cymbal", "sound_source_mechanism", "snd_cymbal", "dev", "D", {"oscillator": "solid_body", "excitation": "struck"}, "pitch", "B"),
    # --- calibration (12: 3A, 3B, 3C, 3D) ---
    ("violin", "sound_source_mechanism", "snd_violin", "calibration", "A", {"oscillator": "stretched_string", "excitation": "bowed"}, "pitch", "A"),
    ("five-string banjo", "sound_source_mechanism", "snd_banjo", "calibration", "A", {"oscillator": "stretched_string", "excitation": "plucked"}, "pitch", "A"),
    ("mandolin", "sound_source_mechanism", "snd_mandolin", "calibration", "A", {"oscillator": "stretched_string", "excitation": "plucked"}, "pitch", "A"),
    ("concert flute", "sound_source_mechanism", "snd_flute", "calibration", "B", {"oscillator": "air_column", "excitation": "direct_air_flow"}, "pitch", "A"),
    ("alto saxophone", "sound_source_mechanism", "snd_sax", "calibration", "B", {"oscillator": "flexible_reed", "excitation": "direct_air_flow"}, "pitch", "A"),
    ("tenor trombone", "sound_source_mechanism", "snd_trombone", "calibration", "B", {"oscillator": "air_column", "excitation": "lip_buzz_air"}, "pitch", "A"),
    ("concert bass drum", "sound_source_mechanism", "snd_bass_drum", "calibration", "C", {"oscillator": "stretched_membrane", "excitation": "struck"}, "pitch", "B"),
    ("timpani", "sound_source_mechanism", "snd_timpani", "calibration", "C", {"oscillator": "stretched_membrane", "excitation": "struck"}, "pitch", "A"),
    ("tom-tom drum", "sound_source_mechanism", "snd_tomtom", "calibration", "C", {"oscillator": "stretched_membrane", "excitation": "struck"}, "pitch", "B"),
    ("marimba", "sound_source_mechanism", "snd_marimba", "calibration", "D", {"oscillator": "solid_body", "excitation": "struck"}, "pitch", "A"),
    ("glockenspiel", "sound_source_mechanism", "snd_glockenspiel", "calibration", "D", {"oscillator": "solid_body", "excitation": "struck"}, "pitch", "A"),
    ("tuning fork", "sound_source_mechanism", "snd_tuning_fork", "calibration", "D", {"oscillator": "solid_body", "excitation": "struck"}, "pitch", "A"),
    # --- test (16: 4A, 4B, 4C, 4D) ---
    ("viola", "sound_source_mechanism", "snd_viola", "test", "A", {"oscillator": "stretched_string", "excitation": "bowed"}, "pitch", "A"),
    ("double bass", "sound_source_mechanism", "snd_double_bass", "test", "A", {"oscillator": "stretched_string", "excitation": "bowed"}, "pitch", "A"),
    ("ukulele", "sound_source_mechanism", "snd_ukulele", "test", "A", {"oscillator": "stretched_string", "excitation": "plucked"}, "pitch", "A"),
    ("sitar", "sound_source_mechanism", "snd_sitar", "test", "A", {"oscillator": "stretched_string", "excitation": "plucked"}, "pitch", "A"),
    ("oboe", "sound_source_mechanism", "snd_oboe", "test", "B", {"oscillator": "flexible_reed", "excitation": "direct_air_flow"}, "pitch", "A"),
    ("bassoon", "sound_source_mechanism", "snd_bassoon", "test", "B", {"oscillator": "flexible_reed", "excitation": "direct_air_flow"}, "pitch", "A"),
    ("french horn", "sound_source_mechanism", "snd_french_horn", "test", "B", {"oscillator": "air_column", "excitation": "lip_buzz_air"}, "pitch", "A"),
    ("tuba", "sound_source_mechanism", "snd_tuba", "test", "B", {"oscillator": "air_column", "excitation": "lip_buzz_air"}, "pitch", "A"),
    ("kettledrum", "sound_source_mechanism", "snd_kettledrum", "test", "C", {"oscillator": "stretched_membrane", "excitation": "struck"}, "pitch", "A"),
    ("djembe", "sound_source_mechanism", "snd_djembe", "test", "C", {"oscillator": "stretched_membrane", "excitation": "struck"}, "pitch", "B"),
    ("taiko drum", "sound_source_mechanism", "snd_taiko", "test", "C", {"oscillator": "stretched_membrane", "excitation": "struck"}, "pitch", "B"),
    ("bodhran", "sound_source_mechanism", "snd_bodhran", "test", "C", {"oscillator": "stretched_membrane", "excitation": "struck"}, "pitch", "B"),
    ("castanets", "sound_source_mechanism", "snd_castanets", "test", "D", {"oscillator": "solid_body", "excitation": "struck"}, "pitch", "B"),
    ("tam-tam gong", "sound_source_mechanism", "snd_gong", "test", "D", {"oscillator": "solid_body", "excitation": "struck"}, "pitch", "B"),
    ("cowbell", "sound_source_mechanism", "snd_cowbell", "test", "D", {"oscillator": "solid_body", "excitation": "struck"}, "pitch", "B"),
    ("brass handbell", "sound_source_mechanism", "snd_handbell", "test", "D", {"oscillator": "solid_body", "excitation": "struck"}, "pitch", "A"),
    # --- anchor (4) ---
    ("grand piano", "sound_source_mechanism", "anchor_snd_a", "anchor", "A", {"oscillator": "stretched_string", "excitation": "struck"}, "pitch", "A"),
    ("pipe organ", "sound_source_mechanism", "anchor_snd_b", "anchor", "B", {"oscillator": "air_column", "excitation": "direct_air_flow"}, "pitch", "A"),
    ("frame drum", "sound_source_mechanism", "anchor_snd_c", "anchor", "C", {"oscillator": "stretched_membrane", "excitation": "struck"}, "pitch", "B"),
    ("wooden maraca", "sound_source_mechanism", "anchor_snd_d", "anchor", "D", {"oscillator": "solid_body", "excitation": "struck"}, "pitch", "B"),

    # ==================== ORIGINALLY UNKNOWN CONTROL (FOR M2 FPR ONLY) ====================
    ("akukuadwo", "sound_source_mechanism", "unk_akukuadwo", "unknown_control", "D", {}, "pitch", "B"),
    ("atwreshie", "sound_source_mechanism", "unk_atwreshie", "unknown_control", "D", {}, "pitch", "B"),
    ("shudraga", "sound_source_mechanism", "unk_shudraga", "unknown_control", "B", {}, "pitch", "A"),
    ("tsenatsil", "sound_source_mechanism", "unk_tsenatsil", "unknown_control", "D", {}, "pitch", "B"),
    ("chipendani", "sound_source_mechanism", "unk_chipendani", "unknown_control", "A", {}, "pitch", "A"),
    ("inanga", "sound_source_mechanism", "unk_inanga", "unknown_control", "A", {}, "pitch", "A"),
    ("kā'eke'eke", "sound_source_mechanism", "unk_kaekeeke", "unknown_control", "D", {}, "pitch", "B"),
    ("mukupela", "sound_source_mechanism", "unk_mukupela", "unknown_control", "C", {}, "pitch", "B"),
]

TRAIN_GENERAL_CONTROLS: List[Dict[str, str]] = [
    {"uid": "train-gen-00", "question": "What is 7 plus 8?\nA: 14\nB: 15\nC: 16\nD: 13\nReply with one letter only.", "gold": "B"},
    {"uid": "train-gen-01", "question": "What is the capital of France?\nA: Lyon\nB: Marseille\nC: Paris\nD: Nice\nReply with one letter only.", "gold": "C"},
    {"uid": "train-gen-02", "question": "Which planet is known as the Red Planet?\nA: Mars\nB: Venus\nC: Jupiter\nD: Mercury\nReply with one letter only.", "gold": "A"},
    {"uid": "train-gen-03", "question": "What is the chemical symbol for oxygen?\nA: Ox\nB: Og\nC: H\nD: O\nReply with one letter only.", "gold": "D"},
    {"uid": "train-gen-04", "question": "What is 9 times 4?\nA: 36\nB: 32\nC: 40\nD: 27\nReply with one letter only.", "gold": "A"},
    {"uid": "train-gen-05", "question": "How many hours are in one day?\nA: 12\nB: 24\nC: 48\nD: 36\nReply with one letter only.", "gold": "B"},
    {"uid": "train-gen-06", "question": "Which gas do plants absorb from the air for photosynthesis?\nA: Nitrogen\nB: Helium\nC: Carbon dioxide\nD: Argon\nReply with one letter only.", "gold": "C"},
    {"uid": "train-gen-07", "question": "What is the freezing point of pure water at standard pressure?\nA: 100 degrees Celsius\nB: 32 degrees Celsius\nC: -10 degrees Celsius\nD: 0 degrees Celsius\nReply with one letter only.", "gold": "D"},
]

AUDIT_GENERAL_CONTROLS: List[Dict[str, str]] = [
    {"uid": "audit-gen-00", "question": "What is 12 times 12?\nA: 124\nB: 144\nC: 132\nD: 154\nReply with one letter only.", "gold": "B"},
    {"uid": "audit-gen-01", "question": "What is 81 divided by 9?\nA: 9\nB: 8\nC: 7\nD: 6\nReply with one letter only.", "gold": "A"},
    {"uid": "audit-gen-02", "question": "What is 17 plus 26?\nA: 33\nB: 53\nC: 43\nD: 42\nReply with one letter only.", "gold": "C"},
    {"uid": "audit-gen-03", "question": "What is 100 minus 37?\nA: 73\nB: 67\nC: 53\nD: 63\nReply with one letter only.", "gold": "D"},
    {"uid": "audit-gen-04", "question": "What is 2 to the power of 5?\nA: 16\nB: 64\nC: 32\nD: 25\nReply with one letter only.", "gold": "C"},
    {"uid": "audit-gen-05", "question": "How many sides does a triangle have?\nA: 3\nB: 4\nC: 5\nD: 6\nReply with one letter only.", "gold": "A"},
    {"uid": "audit-gen-06", "question": "Which number is odd?\nA: 8\nB: 9\nC: 12\nD: 14\nReply with one letter only.", "gold": "B"},
    {"uid": "audit-gen-07", "question": "Which letter follows M in the English alphabet?\nA: N\nB: O\nC: L\nD: M\nReply with one letter only.", "gold": "A"},
    {"uid": "audit-gen-08", "question": "Which word is a synonym of large?\nA: tiny\nB: big\nC: slow\nD: sour\nReply with one letter only.", "gold": "B"},
    {"uid": "audit-gen-09", "question": "Which word is an antonym of hot?\nA: warm\nB: burning\nC: cold\nD: sunny\nReply with one letter only.", "gold": "C"},
    {"uid": "audit-gen-10", "question": "What is three quarters as a decimal?\nA: 0.25\nB: 0.5\nC: 1.25\nD: 0.75\nReply with one letter only.", "gold": "D"},
    {"uid": "audit-gen-11", "question": "What is 6 times 7?\nA: 36\nB: 48\nC: 54\nD: 42\nReply with one letter only.", "gold": "D"},
    {"uid": "audit-gen-12", "question": "What is 9 plus 14?\nA: 23\nB: 25\nC: 21\nD: 24\nReply with one letter only.", "gold": "A"},
    {"uid": "audit-gen-13", "question": "Which day comes immediately after Monday?\nA: Sunday\nB: Tuesday\nC: Friday\nD: Saturday\nReply with one letter only.", "gold": "B"},
    {"uid": "audit-gen-14", "question": "How many metres are in one kilometre?\nA: 100\nB: 10\nC: 1000\nD: 10000\nReply with one letter only.", "gold": "C"},
    {"uid": "audit-gen-15", "question": "How many interior angles does a square have?\nA: 3\nB: 5\nC: 6\nD: 4\nReply with one letter only.", "gold": "D"},
    {"uid": "audit-gen-16", "question": "What is the first month of the calendar year?\nA: January\nB: June\nC: May\nD: December\nReply with one letter only.", "gold": "A"},
    {"uid": "audit-gen-17", "question": "What is 5 squared?\nA: 10\nB: 25\nC: 125\nD: 15\nReply with one letter only.", "gold": "B"},
    {"uid": "audit-gen-18", "question": "What is 40 divided by 5?\nA: 10\nB: 6\nC: 8\nD: 12\nReply with one letter only.", "gold": "C"},
    {"uid": "audit-gen-19", "question": "What is decimal ten in binary?\nA: 101\nB: 111\nC: 1100\nD: 1010\nReply with one letter only.", "gold": "D"},
]


def _make_uid(entity_id: str, domain: str) -> str:
    return hashlib.sha256(f"{domain}::{entity_id}".encode("utf-8")).hexdigest()[:16]


def _build_same_entity_control(entity_id: str, domain: str, ctrl_type: str, ctrl_gold: str) -> Dict[str, str]:
    if ctrl_type == "diet":
        opts = DIET_OPTIONS
        stem = f"What is the primary natural diet of an adult {entity_id}?"
    elif ctrl_type == "func":
        opts = FUNCTION_OPTIONS
        stem = f"In which everyday context is a {entity_id} primarily used?"
    elif ctrl_type == "pitch":
        opts = PITCH_OPTIONS
        stem = f"How does a traditional acoustic {entity_id} function musically?"
    else:
        raise ValueError(f"Unknown ctrl_type: {ctrl_type}")
    prompt = stem + "\n" + "\n".join(f"{k}: {v}" for k, v in opts.items()) + "\nReply with one letter only."
    return {
        "uid": f"ctrl-{_make_uid(entity_id, domain)}-{ctrl_type}",
        "entity_id": entity_id,
        "domain": domain,
        "question": prompt,
        "gold": ctrl_gold,
        "kind": "same_entity_other_relation",
    }


def all_evaluator_units(split: Optional[str] = None, domain: Optional[str] = None) -> List[EvaluatorUnit]:
    """Evaluator-only full records. Never import or call from reconstruction, selector, or repairer."""
    units: List[EvaluatorUnit] = []
    for entity_id, dom, group, sp, gold, attrs, ctrl_type, ctrl_gold in RAW_CATALOG:
        if split is not None and sp != split:
            continue
        if domain is not None and dom != domain:
            continue
        pub = KnowledgeUnit(
            unit_id=_make_uid(entity_id, dom),
            entity_id=entity_id,
            relation_id=DOMAIN_RELATIONS[dom],
            modality="text",
            source_group=group,
            probe_template_ids=[0, 1, 2, 3, 4, 5],
            split=sp,
            domain=dom,
            options=dict(DOMAIN_OPTIONS[dom]),
        )
        ctrl = _build_same_entity_control(entity_id, dom, ctrl_type, ctrl_gold)
        units.append(
            EvaluatorUnit(
                unit=pub,
                gold_answer=gold,
                supporting_attributes=dict(attrs),
                same_entity_controls=[ctrl],
                is_originally_unknown_control=(sp == "unknown_control"),
            )
        )
    return units


def public_units(split: Optional[str] = None, domain: Optional[str] = None) -> List[KnowledgeUnit]:
    """Algorithm-facing public units without ground truth or damage plans."""
    return [ev.unit for ev in all_evaluator_units(split=split, domain=domain)]


def question_prompt(
    unit: KnowledgeUnit,
    variant: int = 0,
    permutation: Optional[Sequence[str]] = None,
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """Build MCQ prompt and option mappings."""
    stems = DOMAIN_TEMPLATES[unit.domain]
    stem = stems[variant].format(entity=unit.entity_id)
    order = tuple(permutation) if permutation is not None else ("A", "B", "C", "D")
    if set(order) != {"A", "B", "C", "D"} or len(order) != 4:
        raise ValueError(f"Invalid option permutation: {order}")
    disp_keys = ("A", "B", "C", "D")
    disp_to_canon = {d: c for d, c in zip(disp_keys, order)}
    canon_to_disp = {c: d for d, c in disp_to_canon.items()}
    lines = [stem] + [f"{d}: {unit.options[c]}" for d, c in zip(disp_keys, order)] + ["Reply with one letter only."]
    return "\n".join(lines), disp_to_canon, canon_to_disp


def train_retain_prompts(include_multi_template: bool = True) -> List[str]:
    """Training-time retention prompts using ONLY anchor entities (never dev/cal/test) and train general items."""
    prompts: List[str] = []
    anchors = public_units(split="anchor")
    variants = (0, 1, 2, 3) if include_multi_template else (2,)
    for u in anchors:
        for v in variants:
            p, _, _ = question_prompt(u, variant=v)
            prompts.append(p)
    for g in TRAIN_GENERAL_CONTROLS:
        prompts.append(g["question"])
    return prompts


def audit_controls_for_split(split: str, exclude_entity: Optional[str] = None) -> List[Dict[str, str]]:
    """Independent audit controls across same_entity_other_relation, other_entity_unseen_template, near_neighbor, and general."""
    controls: List[Dict[str, str]] = []
    ev_units = all_evaluator_units(split=split)
    for ev in ev_units:
        if exclude_entity is not None and ev.unit.entity_id == exclude_entity:
            for c in ev.same_entity_controls:
                controls.append(dict(c))
            continue
        for v in (4, 5):
            p, _, _ = question_prompt(ev.unit, variant=v)
            controls.append(
                {
                    "uid": f"{ev.unit.unit_id}-v{v}",
                    "entity_id": ev.unit.entity_id,
                    "domain": ev.unit.domain,
                    "question": p,
                    "gold": ev.gold_answer,
                    "kind": "other_entity_unseen_template",
                }
            )
    for ev in ev_units:
        if ev.unit.entity_id != exclude_entity:
            for c in ev.same_entity_controls:
                controls.append(dict(c, kind="near_neighbor_invariant"))
    for g in AUDIT_GENERAL_CONTROLS:
        controls.append(dict(g, entity_id="", domain="general", kind="general_commonsense"))
    return controls


def verify_split_integrity() -> Dict[str, object]:
    """Verify entity_id and source_group disjointness across all splits and train/audit isolation."""
    splits = ["dev", "calibration", "test", "anchor", "unknown_control"]
    by_split = {s: all_evaluator_units(split=s) for s in splits}
    seen_entities: Dict[str, str] = {}
    seen_groups: Dict[str, str] = {}
    for s, items in by_split.items():
        for ev in items:
            e = ev.unit.entity_id.lower().strip()
            g = ev.unit.source_group.lower().strip()
            if e in seen_entities and seen_entities[e] != s:
                raise AssertionError(f"Entity {e} leaks across {seen_entities[e]} and {s}")
            if g in seen_groups and seen_groups[g] != s:
                raise AssertionError(f"Source group {g} leaks across {seen_groups[g]} and {s}")
            seen_entities[e] = s
            seen_groups[g] = s
    train_set = set(train_retain_prompts(include_multi_template=True))
    for s in ["dev", "calibration", "test"]:
        audit_qs = {c["question"] for c in audit_controls_for_split(s)}
        overlap = train_set.intersection(audit_qs)
        if overlap:
            raise AssertionError(f"Train retain overlaps audit controls for {s}: {overlap}")
    return {
        "counts": {s: len(items) for s, items in by_split.items()},
        "unique_entities": len(seen_entities),
        "unique_source_groups": len(seen_groups),
        "train_retain_count": len(train_set),
        "disjoint_verified": True,
    }


def manifest() -> Dict[str, object]:
    integrity = verify_split_integrity()
    payload = json.dumps(RAW_CATALOG, sort_keys=True)
    return {
        "catalog_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "integrity": integrity,
    }
