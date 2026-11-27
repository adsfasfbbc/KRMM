#!/usr/bin/env python3
"""GPU entry points for the frozen DEV-V8 evaluator and Oracle writer."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.v8.rebuild_replay import read, write

PUBLIC = ROOT / "data/v8/public/dev_v8.json"
PRIVATE = ROOT / "data/v8/evaluator_private/dev_v8.json"
PROTOCOL = ROOT / "configs/v8/dev_protocol.json"
LEGACY = ROOT / "configs/v6/pilot.json"


def data_paths(cfg):
    return ROOT / cfg.get("public_path", str(PUBLIC)), ROOT / cfg.get("private_path", str(PRIVATE))


def event(path, **data):
    with path.open("a") as stream:
        stream.write(json.dumps(data, allow_nan=False) + "\n")


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configs():
    old = read(LEGACY)
    anchor = {**old["lora"], "model_path": old["model_path"]}
    return old, {**anchor, **old["damage_cfg"]}, {**anchor, **old["repair_cfg"]}


def prepare(out, cfg):
    import math
    import torch
    from krmm.model import Omni
    from krmm.v3.damage import balanced_wrong_assignments, inject_damage, snapshot_weights, state_sha256
    from krmm.v3.knowledge_registry import train_retain_prompts
    from krmm.v3.schemas import EvaluatorUnit
    from krmm.v8.dev_eval import anchor_truth, control_specs, controls, knowledge, screen, select_damage, targets

    old, damage, repair = configs()
    public_path, private_path = data_paths(cfg)
    public = read(public_path)
    private = {row["unit_id"]: row for row in read(private_path)}
    hidden = out / "private"
    exchange = out / "oracle_input"
    set_seed(cfg["seed"])
    model = Omni(old["model_path"])
    clean = screen(model, public, private)
    write(hidden / "clean_screen.json", clean)
    planned, by_domain = select_damage(public, clean, old["damage_fraction"], cfg["seed"])
    mastered = sum(by_domain.values())
    base_gate = {"n": len(public), "mastered": mastered, "by_domain": by_domain, "planned": len(planned)}
    if mastered < cfg["min_clean_mastered"]:
        write(hidden / "g0.json", {**base_gate, "status": "FAIL_CLEAN_MASTERY"})
        print("DEV G0 clean mastery failed", base_gate, flush=True)
        return

    units = [EvaluatorUnit(knowledge(row), private[row["unit_id"]]["gold"], [], [], False) for row in planned]
    assignments, _ = balanced_wrong_assignments(units, seed=cfg["seed"])
    write(hidden / "damage_assignments.json", assignments)
    retain = train_retain_prompts(include_multi_template=True)
    teacher = model.retain_logits(retain)
    for index, unit in enumerate(units, 1):
        set_seed(cfg["seed"] + index)
        event(hidden / "damage_fits.jsonl", fit_start=index, unit_id=unit.unit.unit_id, steps=damage["damage_steps"])
        before = model.compute_tokens
        trace = inject_damage(model, unit.unit, unit.gold_answer, assignments[unit.unit.unit_id], retain, teacher, damage, damage_type="D1_wrong_association")
        if not all(math.isfinite(value) for row in trace for value in row.values()):
            raise FloatingPointError("Nonfinite damage fit")
        event(hidden / "damage_fits.jsonl", fit_end=index, tokens=model.compute_tokens - before)
        print("DEV damage", index, "/", len(units), flush=True)

    damaged = targets(model, public, private)
    write(hidden / "damaged_targets.json", damaged)
    truth = anchor_truth(public, private, clean, damaged, set(assignments))
    write(hidden / "truth.json", truth)
    degraded = [row["unit_id"] for row in truth if row["is_degraded_target"]]
    if len(degraded) < cfg["min_degraded"]:
        write(hidden / "g0.json", {**base_gate, "degraded": len(degraded), "status": "FAIL_EFFECTIVE_DAMAGE"})
        return
    random.Random(cfg["order_seed"]).shuffle(degraded)
    oracle_targets = degraded[:cfg["oracle_targets"]]
    specs = control_specs(public, private)
    first = controls(model, specs, False)
    anchor = controls(model, specs, True)
    if any(a["probabilities"] != b["probabilities"] for a, b in zip(first, anchor, strict=True)):
        raise RuntimeError("B0 fixed-control probabilities drifted")
    write(hidden / "m0_controls.json", anchor)
    weights = snapshot_weights(model, repair)
    state_hash = state_sha256(weights)
    torch.save(weights, exchange / "m0.pt")
    write(exchange / "requests.json", {
        "state_sha256": state_hash,
        "requests": [{"unit_id": uid, "gold": private[uid]["gold"]} for uid in oracle_targets],
    })
    write(hidden / "g0.json", {**base_gate, "degraded": len(degraded), "status": "PASS", "m0_sha256": state_hash})
    print("DEV G0 PASS", base_gate, "degraded", len(degraded), flush=True)


def isolate(out, cfg):
    from krmm.v8.isolation import require_denied, restrict
    model = read(LEGACY)["model_path"]
    public_path, private_path = data_paths(cfg)
    allowed = [
        "/usr", "/opt", "/lib", "/lib64", "/etc", "/sys", "/proc/self", "/proc/cpuinfo",
        "/proc/meminfo", "/proc/driver", "/proc/sys", ROOT / "krmm", ROOT / "scripts",
        public_path, LEGACY, out / "oracle_input", model, Path(sys.prefix),
    ]
    workspace = out / "writer"
    os.environ["HOME"] = str(workspace / "scratch")
    os.environ["TMPDIR"] = str(workspace / "scratch")
    os.environ["XDG_CACHE_HOME"] = str(workspace / "scratch")
    abi = restrict(allowed, [workspace], ["/dev"])
    require_denied(private_path)
    if private_path != PRIVATE:
        require_denied(PRIVATE)
    require_denied(out / "private" / "g0.json")
    return abi


def writer(out, cfg):
    abi = isolate(out, cfg)
    import torch
    from krmm.model import Omni
    from krmm.v3.damage import restore_weights, snapshot_weights, state_sha256
    from krmm.v3.knowledge_registry import question_prompt, train_retain_prompts
    from krmm.v7.relation_retain import EXTRA_GENERIC_RETAIN_PROMPTS
    from krmm.v8.constrained_writer import audit_merged, collect_basis, fit_update, random_bases
    from krmm.v8.dev_eval import knowledge

    old, _, repair = configs()
    root = out / "writer"
    public = {row["unit_id"]: row for row in read(data_paths(cfg)[0])}
    oracle = read(out / "oracle_input" / "requests.json")
    requests = oracle["requests"]
    if len(requests) != cfg["oracle_targets"] or len({row["unit_id"] for row in requests}) != len(requests):
        raise RuntimeError("Invalid scoped Oracle input")
    write(root / "isolation.json", {"landlock_abi": abi, "private_manifest_denied": True, "private_run_denied": True})
    model = Omni(old["model_path"])
    anchor = torch.load(out / "oracle_input" / "m0.pt", map_location="cpu", weights_only=True)
    restore_weights(model, anchor)
    expected = oracle["state_sha256"]
    if state_sha256(snapshot_weights(model, repair)) != expected:
        raise RuntimeError("Oracle M0 hash mismatch")
    retains = train_retain_prompts(include_multi_template=True)
    teacher = model.retain_logits(retains)
    extra = list(EXTRA_GENERIC_RETAIN_PROMPTS)
    extra_teacher = model.retain_logits(extra)
    references = {}
    fits = 0

    for name in cfg["branches"]:
        folder = root / name
        folder.mkdir()
        restore_weights(model, anchor)
        if state_sha256(snapshot_weights(model, repair)) != expected:
            raise RuntimeError("Branch did not reset to Oracle M0")
        committed, attempts = [], []
        for index, request in enumerate(requests, 1):
            if name in {"WN", "WR"} and index not in references:
                break
            if fits >= cfg["max_writer_fits"]:
                raise RuntimeError("Writer fit cap reached")
            row = public[request["unit_id"]]
            unit = knowledge(row)
            before = snapshot_weights(model, repair)
            pre_sha = state_sha256(before)
            peers = [item["public_protection"] for item in public.values() if item["domain"] == row["domain"] and item["unit_id"] != row["unit_id"]][:2]
            previous = [question_prompt(knowledge(public[item["unit_id"]]), variant=2)[0] for item in committed[-2:]]
            prompts = [*retains[:4], row["public_protection"], *peers, *previous]
            basis = collect_basis(model, before, prompts, cfg["rank_cap"])
            if set(basis) != set(before):
                raise RuntimeError("Incomplete protected activation bank")
            applied = basis if name == "WP" else (random_bases(basis, cfg["order_seed"] + index * 100) if name == "WR" else {})
            match = references.get(index) if name in {"WN", "WR"} else None
            set_seed(cfg["seed"] + 1000 + index)
            fits += 1
            event(root / "fits.jsonl", fit_start=fits, branch=name, step=index, unit_id=unit.unit_id, pre_sha256=pre_sha)
            measures, tokens = fit_update(
                model, unit, request["gold"], repair, before, retains, teacher, extra if name != "W0" else [],
                extra_teacher if name != "W0" else [], applied, match["effective"] if match else None,
            )
            path = folder / f"step-{index:02}"
            path.mkdir()
            model.model.save_pretrained(str(path), save_embedding_layers=False)
            model.model = model.model.merge_and_unload(safe_merge=True)
            model.model.eval()
            after = snapshot_weights(model, repair)
            audit_merged(before, after, measures, basis, applied, match["merged"] if match else None)
            if name == "WP":
                references[index] = {
                    "effective": {key: value["effective_norm"] for key, value in measures.items()},
                    "merged": {key: value["merged_norm"] for key, value in measures.items()},
                }

            analytic = max((row.get("analytic_residual", 0) for row in measures.values()), default=0)
            merged = max((row.get("merged_residual", 0) for row in measures.values()), default=0)
            match_error = max((row.get("matched_relative_error", 0) for row in measures.values()), default=0)
            reject = name in {"WP", "WR"} and (
                analytic > cfg["analytic_residual"] or merged > cfg["bf16_residual"]
                or any(row["merged_norm"] <= cfg["min_layer_delta"] for row in measures.values())
            )
            if match_error > cfg["norm_tolerance"]:
                reject = True
            record = {"branch": name, "step": index, "unit_id": unit.unit_id, "pre_sha256": pre_sha,
                      "candidate_sha256": state_sha256(after), "adapter": str(path.relative_to(root)),
                      "committed": not reject, "train_tokens": tokens, "analytic_residual": analytic,
                      "merged_residual": merged, "max_norm_error": match_error, "layer_audit": measures,
                      "public_prompt_sha256": [hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts]}
            if reject:
                restore_weights(model, before)
                if state_sha256(snapshot_weights(model, repair)) != pre_sha:
                    raise RuntimeError("Rejected update could not restore original weights")
            else:
                committed.append(record)
            attempts.append(record)
            event(root / "fits.jsonl", fit_end=fits, **record)
            print("DEV", name, index, "commit", not reject, "rho", merged, "norm_err", match_error, flush=True)
            if reject:
                break
        probes = []
        for prompt in retains[:4]:
            probabilities, _ = model.probabilities(prompt)
            probes.append({"prompt": prompt, "probabilities": probabilities})
        write(folder / "trajectory.json", {
            "attempts": attempts, "committed": committed, "public_probes": probes,
            "final_sha256": state_sha256(snapshot_weights(model, repair)),
        })
    write(root / "complete.json", {"fits": fits, "compute": model.state()})
