"""Entry point for v0.4 end-to-end blind repair matrix (CPU mock supported; GPU blocked without auth)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cpu_mock", "gpu"], default="cpu_mock")
    ap.add_argument("--authorized-gpu-plan", default=None)
    ap.add_argument("--out", default="results/v4-new/repair_matrix_mock.json")
    args = ap.parse_args()

    if args.mode == "gpu" and not args.authorized_gpu_plan:
        raise RuntimeError(
            "GPU execution is blocked in PLAN/CPU VALIDATION phase. "
            "Requires explicit GPU authorization block (Section 11)."
        )

    from krmm.v3.knowledge_registry import all_evaluator_units
    from krmm.v3.schemas import AtomicFact
    from krmm.v4.damage_truth_audit import classify_pool_truth
    from krmm.v4.end_to_end_pipeline import run_blind_end_to_end

    evs = all_evaluator_units(split="dev")[:6]
    pub = [e.unit for e in evs]
    # Simulate 2 damaged_mastered, 2 healthy_mastered, 2 originally_unknown
    truths = [
        classify_pool_truth(evs[0].unit.unit_id, True, True, True, 1.0, 0.0, locality_pass=True).to_dict(),
        classify_pool_truth(evs[1].unit.unit_id, True, True, True, 1.0, 0.0, locality_pass=True).to_dict(),
        classify_pool_truth(evs[2].unit.unit_id, True, False, False, 1.0, 1.0).to_dict(),
        classify_pool_truth(evs[3].unit.unit_id, True, False, False, 1.0, 1.0).to_dict(),
        classify_pool_truth(evs[4].unit.unit_id, False, False, False, 0.25, 0.25).to_dict(),
        classify_pool_truth(evs[5].unit.unit_id, False, False, False, 0.25, 0.25).to_dict(),
    ]

    class _MockBackend:
        def probabilities(self, prompt: str, media=None):
            if evs[0].unit.entity_id in prompt or evs[1].unit.entity_id in prompt:
                return [0.05, 0.85, 0.05, 0.05], 0.95
            if evs[4].unit.entity_id in prompt or evs[5].unit.entity_id in prompt:
                return [0.25, 0.25, 0.25, 0.25], 0.20
            return [0.85, 0.05, 0.05, 0.05], 0.95

        def generate(self, prompt: str, max_new_tokens: int = 24, media=None):
            return "warm-blooded lungs live birth fur"

    def _extractor(u):
        return [
            AtomicFact(u.entity_id, "covering", "fur_or_hair", "probe", "text", 0.9, "damaged_model_internal"),
            AtomicFact(u.entity_id, "thermoregulation", "warm_blooded", "probe", "text", 0.9, "damaged_model_internal"),
            AtomicFact(u.entity_id, "reproduction", "live_birth", "probe", "text", 0.9, "damaged_model_internal"),
        ]

    gold_map = {e.unit.unit_id: e.gold_answer for e in evs}

    def _auditor(uid, action, tr):
        match = tr.get("candidate_answer") == gold_map[uid]
        return {
            "candidate_matches_gold": match,
            "strict_parametric_recovery": bool(action == "ATTEMPT_PARAMETRIC_REPAIR" and match),
            "healthy_harmed": False,
            "oracle_repairable": True,
        }

    res = run_blind_end_to_end(
        public_units=pub,
        model_backend=_MockBackend(),
        fact_extractor=_extractor,
        discovery_budget=36,
        repair_budget=6,
        discovery_method="full_m2",
        evaluator_auditor=_auditor,
        evaluator_truth_records=truths,
    )
    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["evaluator_end_to_end_metrics"], indent=2))


if __name__ == "__main__":
    main()
