"""Experiment V3-01: Training-free clean mastery, control baseline, and attribute-graph survey on dev."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from krmm.model import Omni
from krmm.v3.damage_audit import measure_controls
from krmm.v3.evidence import extract_atomic_facts
from krmm.v3.knowledge_registry import (
    AUDIT_GENERAL_CONTROLS,
    TRAIN_GENERAL_CONTROLS,
    all_evaluator_units,
)
from krmm.v3.mastery import screen_clean_mastery, summarize_mastery
from krmm.v3.reconstruction import reconstruct_structured


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/rivermind-data/models/Qwen2.5-Omni-3B")
    ap.add_argument("--out", default="results/v3-new/exp01-mastery-survey-dev")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    omni = Omni(args.model)

    split_units = all_evaluator_units(split=args.split)
    anchor_units = all_evaluator_units(split="anchor")
    unknown_units = all_evaluator_units(split="unknown_control")

    dev_records = []
    attr_correct = 0
    for ev in split_units:
        m = screen_clean_mastery(omni, ev, min_choice_mass=0.25)
        facts, _ = extract_atomic_facts(omni, ev.unit, max_new_tokens=32)
        struct = reconstruct_structured(ev.unit.domain, facts)
        ctrl_res = measure_controls(omni, ev.same_entity_controls)
        ctrl_ok = all(p == c["gold"] for p, c in zip(ctrl_res["prediction"], ev.same_entity_controls))
        cand_ok = struct["candidate"] == ev.gold_answer
        if cand_ok:
            attr_correct += 1
        rec = {
            "unit_id": ev.unit.unit_id,
            "entity_id": ev.unit.entity_id,
            "domain": ev.unit.domain,
            "gold": ev.gold_answer,
            "mastery": m,
            "same_entity_ctrl_ok": ctrl_ok,
            "same_entity_ctrl_pred": ctrl_res["prediction"],
            "clean_structured_candidate": struct["candidate"],
            "clean_structured_correct": cand_ok,
            "clean_true_composition": struct["true_composition"],
            "clean_source_type": struct["source_type"],
            "facts": [f.to_dict() for f in facts],
        }
        dev_records.append(rec)
        print(
            ev.unit.domain,
            ev.unit.entity_id,
            "mastered=", m["stably_mastered"],
            "heldout=", m["heldout_clean_mastered"],
            "ctrl=", ctrl_ok,
            "struct=", struct["candidate"],
            "gold=", ev.gold_answer,
            flush=True,
        )

    anchor_records = [screen_clean_mastery(omni, ev, min_choice_mass=0.25) for ev in anchor_units]
    unk_records = [screen_clean_mastery(omni, ev, min_choice_mass=0.25) for ev in unknown_units]

    train_gen_res = measure_controls(omni, TRAIN_GENERAL_CONTROLS)
    audit_gen_res = measure_controls(omni, AUDIT_GENERAL_CONTROLS)
    train_gen_acc = sum(p == c["gold"] for p, c in zip(train_gen_res["prediction"], TRAIN_GENERAL_CONTROLS)) / len(TRAIN_GENERAL_CONTROLS)
    audit_gen_acc = sum(p == c["gold"] for p, c in zip(audit_gen_res["prediction"], AUDIT_GENERAL_CONTROLS)) / len(AUDIT_GENERAL_CONTROLS)

    by_domain = {}
    for dom in sorted({r["domain"] for r in dev_records}):
        sub = [r for r in dev_records if r["domain"] == dom]
        by_domain[dom] = {
            "summary": summarize_mastery([r["mastery"] for r in sub]),
            "same_entity_ctrl_acc": sum(r["same_entity_ctrl_ok"] for r in sub) / len(sub),
            "clean_structured_acc": sum(r["clean_structured_correct"] for r in sub) / len(sub),
        }

    summary = {
        "split": args.split,
        "overall": summarize_mastery([r["mastery"] for r in dev_records]),
        "same_entity_ctrl_acc": sum(r["same_entity_ctrl_ok"] for r in dev_records) / len(dev_records),
        "clean_structured_acc": attr_correct / len(dev_records),
        "by_domain": by_domain,
        "anchor_mastery": summarize_mastery(anchor_records),
        "unknown_control_mastery": summarize_mastery(unk_records),
        "train_general_acc": train_gen_acc,
        "audit_general_acc": audit_gen_acc,
        "resources": omni.state(),
    }
    (out / "records.json").write_text(json.dumps(dev_records, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
