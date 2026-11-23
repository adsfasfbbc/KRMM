"""Verify selective policy abstention, request_evidence, nonparametric fallback, and repair actions."""
from __future__ import annotations

from krmm.v3.schemas import AtomicFact
from krmm.v4.evidence_graph import EvidenceGraphAnalyzer
from krmm.v4.selective_policy import decide_repair


def test_selective_policy_all_four_actions():
    an = EvidenceGraphAnalyzer("vertebrate_class")
    sus = {"status": "suspected"}

    # 1. Conflict -> ABSTAIN
    c_conf = an.assess([
        AtomicFact("x", "thermoregulation", "warm_blooded", "p", "text", 0.9, "m"),
        AtomicFact("x", "respiration", "gills", "p", "text", 0.9, "m"),
    ])
    assert decide_repair(sus, c_conf, {"remaining_repairs": 5})["action"] == "ABSTAIN"

    # 2. Insufficient -> REQUEST_EXTERNAL_EVIDENCE
    c_ins = an.assess([AtomicFact("x", "thermoregulation", "warm_blooded", "p", "text", 0.9, "m")])
    assert decide_repair(sus, c_ins, {"remaining_repairs": 5})["action"] == "REQUEST_EXTERNAL_EVIDENCE"

    # 3. Corroborated with budget -> ATTEMPT_PARAMETRIC_REPAIR
    c_ok = an.assess([
        AtomicFact("x", "covering", "feathers", "p", "text", 0.9, "m"),
        AtomicFact("x", "thermoregulation", "warm_blooded", "p", "text", 0.9, "m"),
        AtomicFact("x", "reproduction", "hard_shelled_eggs", "p", "text", 0.9, "m"),
    ])
    assert decide_repair(sus, c_ok, {"remaining_repairs": 5})["action"] == "ATTEMPT_PARAMETRIC_REPAIR"

    # 4. Corroborated with 0 parametric budget -> NONPARAMETRIC_COMPENSATION
    assert decide_repair(sus, c_ok, {"remaining_repairs": 0, "allow_nonparametric": True})["action"] == "NONPARAMETRIC_COMPENSATION"
