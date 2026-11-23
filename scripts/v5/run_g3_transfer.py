"""Batch 3 (G3): Bounded D2_suppression + Second Model Family Generalization on brand-new v0.5 entities."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from peft import LoraConfig, get_peft_model

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from krmm.v3.damage import attach_lora, balanced_wrong_assignments, inject_damage, restore_weights, snapshot_weights
from krmm.v3.damage_audit import compare_locality, measure_controls
from krmm.v3.evidence import extract_atomic_facts
from krmm.v3.knowledge_registry import AUDIT_GENERAL_CONTROLS, question_prompt, train_retain_prompts
from krmm.v3.mastery import HELDOUT_PERM, HELDOUT_VARIANTS, evaluate_unit_target, screen_clean_mastery
from krmm.v3.parameter_repair import repair as run_parameter_repair
from krmm.v5.selective_repair import REPAIR_STRATEGIES_V5, select_candidate_for_strategy
from krmm.v5.support_structure import analyze_support_structure
from krmm.v5.v5_catalog import v5_known_evaluator_units


def _choose(probs: List[float]) -> str:
    return "ABCD"[int(np.argmax(probs))]


def inject_bounded_d2(omni, unit, gold_answer: str, retains, teacher_logits, cfg: Dict[str, object], max_unlikelihood: float = 2.0):
    """Bounded D2 response suppression: clamps target unlikelihood at max_unlikelihood with retain KL=50.0."""
    params = attach_lora(omni, cfg)
    lr = float(cfg.get("damage_lr", 5e-4))
    steps = int(cfg.get("damage_steps", 48))
    retain_weight = float(cfg.get("damage_retain_weight", 50.0))
    l2_weight = float(cfg.get("damage_l2_weight", 1e-4))
    retain_batch = int(cfg.get("retain_batch_size", 24))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    perm_schedule = [("A", "B", "C", "D"), ("D", "C", "B", "A")]
    variants = (0, 1)

    for step in range(steps):
        omni.model.train()
        opt.zero_grad(set_to_none=True)
        var = variants[step % len(variants)]
        perm = perm_schedule[(step // len(variants)) % len(perm_schedule)]
        prompt, _, c2d = question_prompt(unit, variant=var, permutation=perm)
        disp_gold = c2d[gold_answer]
        ce = omni.model(**omni.encode(prompt, disp_gold)).loss
        raw_ul = -torch.log1p(-torch.exp(-ce).clamp(max=1.0 - 1e-6))
        target_loss = raw_ul.clamp(max=max_unlikelihood)

        n_ret = len(retains)
        bsize = min(retain_batch, n_ret)
        idxs = [(step * bsize + k) % n_ret for k in range(bsize)]
        r_prompts = [retains[j] for j in idxs]
        logits = omni.model(**omni.encode_many(r_prompts)).logits[:, -1].float()
        t = torch.stack([teacher_logits[j] for j in idxs]).to(logits.device)
        kl = torch.nn.functional.kl_div(logits.log_softmax(-1), t.softmax(-1), reduction="batchmean")
        l2 = sum(p.float().square().sum() for p in params)
        loss = target_loss + retain_weight * kl + l2_weight * l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    omni.model = omni.model.merge_and_unload()
    for p in omni.model.parameters():
        p.requires_grad_(False)
    omni.model.eval()


class _CausalLMWrapper:
    """Generic HuggingFace AutoModelForCausalLM wrapper matching Omni's interface for second-model G3-B."""

    def __init__(self, model_path: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
        )
        # Expose text_config alias for compatibility with snapshot_weights / attach_lora
        if not hasattr(self.model.config, "text_config"):
            self.model.config.text_config = self.model.config
        self.model.eval()
        self.choice_ids = [self.tokenizer.encode(c, add_special_tokens=False)[-1] for c in "ABCD"]

    def _fmt(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        return prompt + "\nAnswer:"

    def encode(self, prompt: str, answer: str):
        p_txt = self._fmt(prompt)
        p_ids = self.tokenizer(p_txt, return_tensors="pt").input_ids
        a_ids = self.tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids
        inp = torch.cat([p_ids, a_ids], dim=1).to("cuda")
        labels = inp.clone()
        labels[:, : p_ids.shape[1]] = -100
        return {"input_ids": inp, "attention_mask": torch.ones_like(inp), "labels": labels}

    def encode_many(self, prompts: List[str]):
        txts = [self._fmt(p) for p in prompts]
        enc = self.tokenizer(txts, return_tensors="pt", padding=True).to("cuda")
        return enc

    @torch.no_grad()
    def probabilities(self, prompt: str, media=None):
        enc = self.tokenizer(self._fmt(prompt), return_tensors="pt").to("cuda")
        logits = self.model(**enc).logits[0, -1].float()
        full_p = torch.softmax(logits, dim=-1)
        c_probs = [float(full_p[i]) for i in self.choice_ids]
        mass = sum(c_probs)
        norm = [p / mass for p in c_probs] if mass > 1e-12 else [0.25] * 4
        return norm, float(mass)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 24, media=None) -> str:
        enc = self.tokenizer(self._fmt(prompt), return_tensors="pt").to("cuda")
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0][enc.input_ids.shape[1] :], skip_special_tokens=True).strip()

    @torch.no_grad()
    def retain_logits(self, prompts: List[str]):
        out = []
        for i in range(0, len(prompts), 16):
            batch = prompts[i : i + 16]
            logits = self.model(**self.encode_many(batch)).logits[:, -1].float().cpu()
            for row in logits:
                out.append(row)
        return out


def _build_controls(target_ev, all_evs):
    controls = [dict(sc) for sc in target_ev.same_entity_controls]
    for ev in all_evs:
        if ev.unit.unit_id == target_ev.unit.unit_id:
            continue
        q, _, _ = question_prompt(ev.unit, variant=4)
        controls.append({"question": q, "gold": ev.gold_answer, "kind": "near_neighbor_invariant" if ev.unit.domain == target_ev.unit.domain else "other_entity_unseen_template", "entity_id": ev.unit.entity_id})
    for g in AUDIT_GENERAL_CONTROLS:
        controls.append({"question": g["question"], "gold": g["gold"], "kind": "general_commonsense", "entity_id": "__general__"})
    return controls


def _eval_cohort(omni, evs, cfg, damage_mode: str) -> dict:
    clean_state = snapshot_weights(omni, cfg)
    retains = train_retain_prompts(include_multi_template=True)
    teacher_logits = omni.retain_logits(retains)
    assign, _ = balanced_wrong_assignments(evs, seed=int(cfg["seed"]))

    n_mast = n_eff = n_elig = 0
    rows = []
    for idx, ev in enumerate(evs, 1):
        restore_weights(omni, clean_state)
        u = ev.unit
        m = screen_clean_mastery(omni, ev, min_choice_mass=0.25)
        if not m["stably_mastered"]:
            continue
        n_mast += 1
        controls = _build_controls(ev, evs)
        clean_ctrl = measure_controls(omni, controls)

        if damage_mode == "D2_bounded":
            inject_bounded_d2(omni, u, ev.gold_answer, retains, teacher_logits, cfg, max_unlikelihood=2.0)
        else:
            inject_damage(omni, u, ev.gold_answer, assign[u.unit_id], retains, teacher_logits, cfg, damage_type="D1_wrong_association")

        dmg_held = evaluate_unit_target(omni, ev.unit, ev.gold_answer, variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,))
        dmg_ctrl = measure_controls(omni, controls)
        dmg_loc = compare_locality(clean_ctrl, dmg_ctrl, controls, u.entity_id)
        if dmg_held["accuracy"] > 0.5:
            continue
        n_eff += 1
        if not dmg_loc["independent_locality_pass"]:
            continue
        n_elig += 1

        dmg_state = snapshot_weights(omni, cfg)
        dmg_teacher = omni.retain_logits(retains)
        p2, _, _ = question_prompt(u, variant=2)
        dir_cand = _choose(omni.probabilities(p2)[0])
        facts, _ = extract_atomic_facts(omni, u, max_new_tokens=24)
        sup = analyze_support_structure(u.domain, facts)
        obs = "\n".join(f"- {f.attribute}: {f.raw_output or f.value}" for f in facts)
        opts = "\n".join(f"{k}: {v}" for k, v in u.options.items())
        gen_cand = _choose(omni.probabilities(f"Infer the correct category using ONLY these observations:\n{obs}\n{opts}\nReply with one letter only.")[0])

        branches = {}
        cand_cache = {}
        for strat in REPAIR_STRATEGIES_V5:
            sel = select_candidate_for_strategy(strat, dir_cand, gen_cand, sup, oracle_gold=ev.gold_answer)
            if not sel["attempt"] or sel["candidate"] is None:
                branches[strat] = {"attempted": False, "strict_restoration": False, "wrong_write": False}
                continue
            cand = str(sel["candidate"])
            if cand not in cand_cache:
                import tempfile
                restore_weights(omni, dmg_state)
                with tempfile.TemporaryDirectory() as td:
                    run_parameter_repair(omni, u, cand, retains, dmg_teacher, cfg, save_dir=str(Path(td) / cand))
                rh = evaluate_unit_target(omni, ev.unit, ev.gold_answer, variants=HELDOUT_VARIANTS, permutations=(HELDOUT_PERM,))
                rc = measure_controls(omni, controls)
                rl = compare_locality(dmg_ctrl, rc, controls, u.entity_id)
                if hasattr(omni.model, "unload"):
                    omni.model = omni.model.unload()
                cand_cache[cand] = {
                    "target_acc": float(rh["accuracy"]),
                    "locality_pass": bool(rl["independent_locality_pass"]),
                    "strict_restoration": bool(rh["accuracy"] >= 0.99 and rl["independent_locality_pass"]),
                }
            branches[strat] = {
                "attempted": True,
                "candidate": cand,
                "wrong_write": cand != ev.gold_answer,
                **cand_cache[cand],
            }
        rows.append({"unit_id": u.unit_id, "entity_id": u.entity_id, "support_class": sup.support_class, "branches": branches})
        print(f"G3_{damage_mode} {idx}/{len(evs)} {u.entity_id:24s} str={branches['structured']['strict_restoration']} eg={branches['evidence_gated']['strict_restoration']} orc={branches['oracle']['strict_restoration']}", flush=True)

    restore_weights(omni, clean_state)
    summary = {}
    for strat in REPAIR_STRATEGIES_V5:
        att = [r for r in rows if r["branches"][strat]["attempted"]]
        succ = sum(int(r["branches"][strat]["strict_restoration"]) for r in rows)
        ww = sum(int(r["branches"][strat]["wrong_write"]) for r in att)
        summary[strat] = {
            "n_strict_eligible": len(rows),
            "n_attempted": len(att),
            "n_strict_restored": succ,
            "rate_eligible": succ / max(1, len(rows)),
            "wrong_write_count": ww,
        }
    return {
        "n_catalog": len(evs),
        "n_mastered": n_mast,
        "n_damage_effective": n_eff,
        "n_strict_eligible": n_elig,
        "strategy_summary": summary,
        "per_unit": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["d2_bounded", "second_model"], required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--limit", type=int, default=18)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    all_evs = v5_known_evaluator_units()
    # Take balanced subset across 3 domains
    per_dom = args.limit // 3
    evs = []
    for dom in ("vertebrate_class", "everyday_material", "sound_source_mechanism"):
        evs.extend([e for e in all_evs if e.unit.domain == dom][:per_dom])

    cfg = {
        "model_path": args.model_path,
        "last_layers": 4,
        "target_modules": ["q_proj", "v_proj", "down_proj"],
        "rank": 8,
        "alpha": 16,
        "damage_lr": 5e-4,
        "damage_steps": 48 if args.mode == "d2_bounded" else 64,
        "damage_retain_weight": 50.0,
        "damage_l2_weight": 1e-4,
        "repair_lr": 5e-4,
        "repair_steps": 32,
        "repair_retain_weight": 50.0,
        "repair_l2_weight": 1e-4,
        "retain_batch_size": 24,
        "seed": args.seed,
    }
    if args.mode == "d2_bounded":
        from krmm.model import Omni
        model = Omni(args.model_path)
        res = _eval_cohort(model, evs, cfg, "D2_bounded")
    else:
        model = _CausalLMWrapper(args.model_path)
        res = _eval_cohort(model, evs, cfg, "D1_second_model")

    res["mode"] = args.mode
    res["model_path"] = args.model_path
    res["elapsed_seconds"] = time.time() - t0
    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"G3_{args.mode}_COMPLETE elapsed={res['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
