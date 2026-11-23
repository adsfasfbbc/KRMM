"""Entry point for v0.4 equal-budget blind discovery (CPU mock supported; GPU blocked without auth)."""
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
    ap.add_argument("--out", default="results/v4-new/blind_discovery_mock.json")
    args = ap.parse_args()

    if args.mode == "gpu" and not args.authorized_gpu_plan:
        raise RuntimeError(
            "GPU execution is blocked in PLAN/CPU VALIDATION phase. "
            "Requires explicit GPU authorization block (Section 11)."
        )

    from krmm.v3.discovery import BudgetedQueryProxy
    from krmm.v3.knowledge_registry import public_units
    from krmm.v4.blind_discovery import run_blind_detector

    class _MockBackend:
        def probabilities(self, prompt: str, media=None):
            return [0.85, 0.05, 0.05, 0.05], 0.95

        def generate(self, prompt: str, max_new_tokens: int = 24, media=None):
            return "warm-blooded lungs live birth fur"

    units = public_units(split="dev")[:6]
    budget = 18
    results = {}
    for method in ("direct_confidence", "self_consistency", "direct_paraphrases", "constraint_conflict_only", "full_m2"):
        proxy = BudgetedQueryProxy(backend=_MockBackend(), query_budget=budget)
        res = run_blind_detector(proxy, units, query_budget=budget, method=method)
        results[method] = {"total_queries": res["total_queries"], "query_budget": budget, "n_reports": len(res["reports"])}

    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({"mode": "cpu_mock", "methods": results}, indent=2), encoding="utf-8")
    print(json.dumps({"mode": "cpu_mock", "methods": results}, indent=2))


if __name__ == "__main__":
    main()
