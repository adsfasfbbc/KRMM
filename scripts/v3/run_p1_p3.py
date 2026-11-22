"""CLI entrypoint for KRMM v0.3 P1-P3 experiments (M2 discovery, E0-E7 matrix, calibration/evaluation)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from krmm.model import Omni
from krmm.v3.damage import snapshot_weights
from krmm.v3.knowledge_registry import train_retain_prompts
from krmm.v3.pipeline_p1_p3 import evaluate_selector_comparisons, run_m2_discovery_eval
from krmm.v3.repairability import DecomposedPredictor
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["m2_discovery", "selector_eval"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=18)
    ap.add_argument("--train-results", nargs="*", default=[])
    ap.add_argument("--eval-results", nargs="*", default=[])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    if args.mode == "m2_discovery":
        omni = Omni(str(cfg["model_path"]))
        clean_state = snapshot_weights(omni, cfg)
        retains = train_retain_prompts(include_multi_template=bool(cfg.get("multi_template_retain", True)))
        teacher_logits = omni.retain_logits(retains)
        res = run_m2_discovery_eval(
            omni, cfg, args.split, clean_state, retains, teacher_logits, limit=args.limit
        )
        res["resources"] = omni.state()
        (out / "m2_discovery_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2), flush=True)
    else:
        def load_recs(paths):
            recs = []
            for p in paths:
                for f in sorted(Path(p).glob("*.json")):
                    if f.name in {"metrics.json", "run_meta.json", "damage-randomization-audit.json"}:
                        continue
                    r = json.loads(f.read_text(encoding="utf-8"))
                    if "unit" in r and "repairs" in r:
                        recs.append(r)
            return recs

        train_recs = load_recs(args.train_results)
        eval_recs = load_recs(args.eval_results)
        predictor = DecomposedPredictor()
        valid_tr = [r for r in train_recs if r.get("strict_eligible") and "structured" in r.get("repairs", {})]
        if valid_tr:
            X = np.asarray([r["diagnostic_features"] for r in valid_tr], dtype=np.float64)
            yc = [int(r["reconstruction"]["candidate_answer"] == r["gold_evaluator_only"]) for r in valid_tr]
            yw = [int(r["repairs"]["structured"]["strict_restoration"]) for r in valid_tr]
            predictor.fit(X, yc, yw)
        res = {
            "calibration_status": predictor.calibration_status,
            "train_n": len(valid_tr),
            "eval": evaluate_selector_comparisons(eval_recs, budget_fraction=0.5, predictor=predictor),
        }
        (out / "selector_eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
