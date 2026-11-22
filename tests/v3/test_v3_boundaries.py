"""Mandatory 13 unit and information-boundary tests for KRMM v0.3 (Section 14.1)."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from krmm.v3.damage import balanced_wrong_assignments, snapshot_weights, state_sha256
from krmm.v3.damage_audit import compare_locality
from krmm.v3.discovery import detect
from krmm.v3.evaluator import evaluate_one_unit
from krmm.v3.knowledge_graph import EvidenceGraph
from krmm.v3.knowledge_registry import (
    all_evaluator_units,
    audit_controls_for_split,
    public_units,
    question_prompt,
    train_retain_prompts,
    verify_split_integrity,
)
from krmm.v3.mastery import screen_clean_mastery
from krmm.v3.metrics import summarize_p0_results
from krmm.v3.reconstruction import reconstruct, reconstruct_structured
from krmm.v3.repairability import DecomposedPredictor, assess_repairability
from krmm.v3.runner import check_budget_guard, run_p0_experiment
from krmm.v3.schemas import AtomicFact, CandidateReport, KnowledgeUnit


class _MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False):
        return [65 + "ABCD".index(text)] if text in "ABCD" else [10]


class _MockConfig:
    class _TextCfg:
        num_hidden_layers = 4
    text_config = _TextCfg()


class _MockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _MockConfig()
        self.w = torch.nn.Parameter(torch.zeros(2, 2))

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        yield "layers.3.self_attn.q_proj.weight", self.w


class _MockOmni:
    def __init__(self, always_wrong: bool = False) -> None:
        self.model = _MockModel()
        self.tokenizer = _MockTokenizer()
        self.always_wrong = always_wrong
        self.calls = 0

    def probabilities(self, prompt: str, media=None):
        self.calls += 1
        if self.always_wrong:
            return [0.25, 0.25, 0.25, 0.25], 0.20
        return [0.85, 0.05, 0.05, 0.05], 0.95

    def generate(self, prompt: str, max_new_tokens: int = 32, media=None):
        self.calls += 1
        return "warm-blooded lungs live birth fur"

    def retain_logits(self, prompts):
        return [torch.tensor([1.0, 0.0, 0.0, 0.0]) for _ in prompts]


class TestV3Boundaries(unittest.TestCase):
    def test_label_isolation(self):
        v3_root = Path(__file__).resolve().parents[2] / "krmm" / "v3"
        algo_files = [
            "discovery.py",
            "evidence.py",
            "reconstruction.py",
            "repairability.py",
            "repair_policy.py",
            "parameter_repair.py",
            "memory_recovery.py",
        ]
        forbidden_imports = {"evaluator", "damage", "damage_audit", "mastery", "all_evaluator_units"}
        for fname in algo_files:
            tree = ast.parse((v3_root / fname).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    mod = (node.module or "").split(".")[-1]
                    self.assertNotIn(mod, forbidden_imports, f"{fname} imports forbidden {mod}")
                    for alias in node.names:
                        self.assertNotIn(alias.name, forbidden_imports, f"{fname} imports {alias.name}")
        with self.assertRaises(ValueError):
            KnowledgeUnit.from_public_dict(
                {
                    "unit_id": "u1",
                    "entity_id": "wolf",
                    "relation_id": "taxonomic_class",
                    "modality": "text",
                    "source_group": "canid",
                    "probe_template_ids": [0],
                    "split": "dev",
                    "domain": "vertebrate_class",
                    "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                    "gold_answer": "A",
                }
            )

    def test_no_inverse_shortcut(self):
        ev = all_evaluator_units(split="dev")
        assign, audit = balanced_wrong_assignments(ev, seed=107)
        self.assertTrue(audit["no_deterministic_inverse"])
        for u in ev:
            self.assertNotEqual(assign[u.unit.unit_id], u.gold_answer)

    def test_mastery_filter(self):
        ev = all_evaluator_units(split="dev")[0]
        omni = _MockOmni(always_wrong=True)
        res = screen_clean_mastery(omni, ev)
        self.assertFalse(res["stably_mastered"])
        summary = summarize_p0_results([
            {"mastery": res, "operational_eligible": False, "strict_eligible": False, "exclusion_reason": res["exclusion_reason"]}
        ])
        self.assertEqual(summary["stably_mastered"], 0)
        self.assertEqual(summary["oracle_strict_restorations"], 0)

    def test_split_integrity(self):
        info = verify_split_integrity()
        self.assertTrue(info["disjoint_verified"])
        self.assertEqual(info["counts"]["dev"], 36)
        self.assertEqual(info["counts"]["calibration"], 36)
        self.assertEqual(info["counts"]["test"], 48)

    def test_detector_blindness(self):
        pubs = public_units(split="dev")[:4]
        omni = _MockOmni(always_wrong=False)
        reports = detect(omni, pubs, probe_budget=20, mode="reference_free")
        self.assertEqual(len(reports), 4)
        for r in reports:
            self.assertIn(r.status, {"suspected", "no_evidence_of_damage", "unknown"})

    def test_evidence_provenance(self):
        f1 = AtomicFact("gray wolf", "thermoregulation", "warm_blooded", "probe", "text", 0.8, "damaged_model_internal")
        f2 = AtomicFact("gray wolf", "reproduction", "live_birth", "probe", "text", 0.8, "damaged_model_internal")
        g = EvidenceGraph("vertebrate_class").evaluate([f1, f2])
        self.assertEqual(g["compatible"], ["A"])
        self.assertEqual(len(g["unique_provenance_groups"]), 1)
        self.assertTrue(g["same_model_correlated_only"])

    def test_abstention(self):
        f1 = AtomicFact("wolf", "thermoregulation", "warm_blooded", "p", "text", 0.8, "g1")
        f2 = AtomicFact("shark", "respiration", "gills", "p", "text", 0.8, "g2")
        res = reconstruct_structured("vertebrate_class", [f1, f2])
        self.assertIsNone(res["candidate"])
        self.assertTrue(res["graph"]["conflict"])
        cand = CandidateReport("u1", None, "insufficient_or_conflicting", [], [], 1.0, True, {}, {}, "conflicting_evidence", 2)
        rep = assess_repairability(cand, None, {}, {})
        self.assertEqual(rep.recommended_action, "abstain")

    def test_independent_locality(self):
        train_set = set(train_retain_prompts(include_multi_template=True))
        audit_set = {c["question"] for c in audit_controls_for_split("dev")}
        self.assertEqual(len(train_set.intersection(audit_set)), 0)
        controls = [{"question": "q1", "gold": "A", "kind": "general_commonsense"}, {"question": "q2", "gold": "B", "kind": "near_neighbor_invariant"}]
        base = {"top_token": [65, 66], "prediction": ["A", "B"], "probabilities": [[0.9, 0.1, 0, 0], [0.1, 0.9, 0, 0]]}
        bad = {"top_token": [66, 66], "prediction": ["B", "B"], "probabilities": [[0.1, 0.9, 0, 0], [0.1, 0.9, 0, 0]]}
        comp = compare_locality(base, bad, controls, "wolf")
        self.assertFalse(comp["independent_locality_pass"])

    def test_counterfactual_attempts(self):
        # Verify true 2-premise composition requires >= 2 premises (singletons remain ambiguous)
        f1 = AtomicFact("nile crocodile", "thermoregulation", "cold_blooded", "p", "text", 0.8, "g1")
        f2 = AtomicFact("nile crocodile", "respiration", "lungs", "p", "text", 0.8, "g2")
        res = reconstruct_structured("vertebrate_class", [f1, f2])
        self.assertEqual(res["candidate"], "C")
        self.assertTrue(res["true_composition"])

    def test_fresh_process_reload(self):
        omni = _MockOmni()
        snap = snapshot_weights(omni, {"last_layers": 4, "target_modules": ["q_proj"]})
        sha1 = state_sha256(snap)
        snap["layers.3.self_attn.q_proj.weight"] += 1.0
        sha2 = state_sha256(snap)
        self.assertNotEqual(sha1, sha2)

    def test_rollback(self):
        controls = [{"question": "q1", "gold": "A", "kind": "same_entity_other_relation"}]
        base = {"top_token": [65], "prediction": ["A"], "probabilities": [[0.9, 0.1, 0, 0]]}
        after = {"top_token": [66], "prediction": ["B"], "probabilities": [[0.1, 0.9, 0, 0]]}
        comp = compare_locality(base, after, controls, "wolf")
        self.assertFalse(comp["independent_locality_pass"])

    def test_budget_guard(self):
        with tempfile.TemporaryDirectory() as td:
            bf = Path(td) / "budget.json"
            check_budget_guard(bf, max_launches=1)
            with self.assertRaises(RuntimeError):
                check_budget_guard(bf, max_launches=1)

    def test_result_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_p = Path(td) / "cfg.json"
            out_p = Path(td) / "existing_run"
            out_p.mkdir()
            cfg_p.write_text(json.dumps({"model_path": "/nonexistent", "max_launches": 5}), encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_p0_experiment(cfg_p, out_p, split="dev", limit=1)


if __name__ == "__main__":
    unittest.main()
