"""Verify strict Evaluator vs Research Method boundary in krmm/v4/."""
from __future__ import annotations

import ast
from pathlib import Path
import pytest

from krmm.v3.schemas import AtomicFact
from krmm.v4.candidate_reconstruction import reconstruct_evidence_gated
from krmm.v4.candidate_reliability import extract_evidence_features
from krmm.v4.evidence_graph import EvidenceGraphAnalyzer
from krmm.v4.schemas import FORBIDDEN_METHOD_KEYS
from krmm.v4.selective_policy import decide_repair

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_v4_method_modules_do_not_import_evaluator_or_omni():
    v4_method_files = [
        "minimal_support.py",
        "evidence_dependence.py",
        "evidence_graph.py",
        "candidate_reconstruction.py",
        "candidate_reliability.py",
        "selective_policy.py",
        "blind_discovery.py",
    ]
    forbidden = {"damage_truth_audit", "locality_audit", "evaluator", "damage", "mastery", "model"}
    for fn in v4_method_files:
        tree = ast.parse((REPO_ROOT / "krmm/v4" / fn).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[-1]
                assert mod not in forbidden, f"{fn} imports forbidden {mod}"
                for a in node.names:
                    assert a.name not in {"Omni", "all_evaluator_units"}, f"{fn} imports {a.name}"


def test_internal_only_blocks_external_facts():
    f_int = AtomicFact("wolf", "covering", "fur_or_hair", "p", "text", 0.9, "damaged_model_internal")
    f_ext = AtomicFact("wolf", "respiration", "lungs", "ext", "text", 0.9, "external_kb")
    with pytest.raises(ValueError):
        reconstruct_evidence_gated("vertebrate_class", [f_int], study_condition="internal_only", external_facts=[f_ext])


@pytest.mark.parametrize("bad_key", sorted(FORBIDDEN_METHOD_KEYS))
def test_method_functions_reject_forbidden_keys(bad_key):
    assess = EvidenceGraphAnalyzer("vertebrate_class").assess([])
    with pytest.raises(ValueError):
        extract_evidence_features(assess, 0.5, 0, extra_meta={bad_key: "secret"})
    with pytest.raises(ValueError):
        decide_repair({"status": "suspected", bad_key: "secret"}, assess, {"remaining_repairs": 1})
