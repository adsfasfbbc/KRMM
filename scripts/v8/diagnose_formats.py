#!/usr/bin/env python3
"""Replay format measurements and compare actual unmerged, merged, and fresh forwards."""
import argparse
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.v8.rebuild_replay import command, digest, gpu_usage, read, verify_inputs, write

CONFIG = ROOT / "configs/v8/format_merge_protocol.json"
QUEUE = ROOT / "configs/v8/format_merge_queue.json"
FACTS = ROOT / "configs/v8/format_merge_facts.json"
LETTERS = "ABCD"
MODES = ("primary", "fresh")


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    text = text.strip(" .!?,;:\"'“”‘’")
    return re.sub(r"^(?:the|an|a) ", "", " ".join(text.split()))


def open_score(text, fact):
    value = normalize(text)
    if value in {normalize(answer) for answer in fact["answers"]}:
        return "correct"
    if value in {normalize(answer) for answer in fact["wrong"]}:
        return "wrong"
    return "unclassified"


def parse_options(question):
    first, *rows, instruction = question.splitlines()
    if instruction != "Reply with one letter only." or len(rows) != 4:
        raise ValueError(f"Unexpected legacy question format: {question}")
    return first, dict(zip(LETTERS, (row[3:] for row in rows), strict=True))


def choice_prompt(stem, choices, order):
    lines = [f"{letter}: {choices[canonical]}" for letter, canonical in zip(LETTERS, order, strict=True)]
    return "\n".join([stem, *lines, "Reply with one letter only."])


def make_questions(cfg, source):
    base = [row for row in source if row["branch"] == "M0"]
    permutations = cfg["option_permutations"]
    if len(permutations) != 4 or any(set(order) != set(LETTERS) for order in permutations):
        raise ValueError("Four valid permutations required")
    if any({order[i] for order in permutations} != set(LETTERS) for i in range(4)):
        raise ValueError("Unbalanced displayed positions")

    cases = []
    for entry in read(FACTS):
        row = base[entry["index"]]
        if row["entity_id"] != entry["entity"] or not row["correct"] or row["gold"] != "A":
            raise ValueError(f"Unexpected anchored baseline row: {entry['id']}")
        if {normalize(a) for a in entry["answers"]} & {normalize(a) for a in entry["wrong"]}:
            raise ValueError(f"Ambiguous alias rule: {entry['id']}")
        stem, options = parse_options(row["question"])
        paraphrase = dict(zip(LETTERS, entry["choices"], strict=True))
        cases.append((entry, row, stem, options, paraphrase))
    if len(cases) != 6 or len({case[0]["id"] for case in cases}) != len(cases):
        raise ValueError("Invalid frozen fact set")
    return cases


def score_choices(backend, prompt, order, gold):
    probs, mass = backend.probabilities(prompt)
    if len(probs) != 4 or not all(math.isfinite(value) for value in [*probs, mass]):
        raise FloatingPointError("Invalid choice probabilities")
    best = max(range(4), key=probs.__getitem__)
    second = sorted(probs, reverse=True)[1]
    return {
        "prompt": prompt, "order": order, "displayed_label": LETTERS[best],
        "canonical_answer": order[best], "correct": order[best] == gold,
        "probabilities": probs, "choice_mass": mass,
        "top2_log_margin": math.log(max(probs[best], 1e-38)) - math.log(max(second, 1e-38)),
    }


def score_state(backend, cases, cfg):
    result = {}
    for fact, row, stem, original, paraphrase in cases:
        gold = row["gold"]
        text = backend.generate(fact["open"], max_new_tokens=cfg["open_max_tokens"])
        result[fact["id"]] = {
            "role": fact["role"],
            "F0": score_choices(backend, row["question"], LETTERS, gold),
            "F1": [
                score_choices(backend, choice_prompt(stem, original, order), order, gold)
                for order in cfg["option_permutations"]
            ],
            "F2": {"prompt": fact["open"], "text": text, "classification": open_score(text, fact)},
            "F3": [
                score_choices(backend, choice_prompt(fact["paraphrase"], paraphrase, order), order, gold)
                for order in cfg["option_permutations"]
            ],
        }
    return result


def squared(tensor):
    import torch

    return float(torch.linalg.vector_norm(tensor.double()).square())


def prepared_delta(adapter, initial):
    layers = {}
    base = adapter.get_base_model()
    for name, layer in base.named_modules():
        if not all(hasattr(layer, attribute) for attribute in ("lora_A", "lora_B", "get_delta_weight")):
            continue
        key = name + ".weight"
        if key not in initial or layer.fan_in_fan_out or layer.use_dora["default"]:
            raise RuntimeError(f"Unexpected adapter orientation: {name}")
        weight = initial[key]
        a = layer.lora_A["default"].weight.detach().float()
        b = layer.lora_B["default"].weight.detach().float()
        delta_fp32 = (b @ a).mul_(layer.scaling["default"]).cpu()
        native = layer.get_delta_weight("default").detach().cpu()
        projected = (weight.float() + native.float()).to(weight.dtype)
        if not projected.isfinite().all() or not delta_fp32.isfinite().all():
            raise FloatingPointError(f"Nonfinite update in {name}")
        layers[key] = (delta_fp32, native, projected)
    if not layers or sum(squared(values[0]) for values in layers.values()) <= 0:
        raise RuntimeError("Adapter has no nonzero effective update")
    return layers


def merge_metrics(initial, final, layers):
    results = {}
    unexpected = [key for key in initial.keys() - layers.keys() if not initial[key].equal(final[key])]
    if unexpected or initial.keys() != final.keys():
        raise RuntimeError("Merge changed unexpected parameters")
    for key, (delta_fp32, native, rounded) in layers.items():
        if not rounded.equal(final[key]):
            raise RuntimeError(f"Observed merge differs from native rounded accumulation: {key}")
        observed = final[key].float() - initial[key].float()
        results[key] = {
            "fp32_delta_frobenius": math.sqrt(squared(delta_fp32)),
            "native_delta_frobenius": math.sqrt(squared(native)),
            "observed_delta_frobenius": math.sqrt(squared(observed)),
            "base_frobenius": math.sqrt(squared(initial[key])),
            "native_vs_fp32_frobenius": math.sqrt(squared(native.float() - delta_fp32)),
            "rounded_weight_error_frobenius": math.sqrt(squared(observed - delta_fp32)),
            "max_abs_rounded_weight_error": float((observed - delta_fp32).abs().max()),
        }
    if not any(row["observed_delta_frobenius"] > 0 for row in results.values()):
        raise RuntimeError("Nonzero adapter rounded to no actual model update")
    combined = {
        key: math.sqrt(sum(layer[key] ** 2 for layer in results.values()))
        for key in next(iter(results.values())) if key != "max_abs_rounded_weight_error"
    }
    combined["relative_observed_delta"] = combined["observed_delta_frobenius"] / combined["base_frobenius"]
    return {"overall": combined, "layers": results}


def compare_states(before, after):
    probs = []
    flips, canonical_flips, open_edits, open_text_edits, mass_diffs = [], [], [], [], []
    for fact, original in before.items():
        changed = after[fact]
        for field in ("F0", "F1", "F3"):
            pairs = [(original[field], changed[field])] if field == "F0" else zip(original[field], changed[field], strict=True)
            for left, right in pairs:
                if left["prompt"] != right["prompt"]:
                    raise RuntimeError("Probe question mismatch")
                pair = (fact, field, left["order"])
                probs.extend(abs(a - b) for a, b in zip(left["probabilities"], right["probabilities"], strict=True))
                mass_diffs.append(abs(left["choice_mass"] - right["choice_mass"]))
                if left["displayed_label"] != right["displayed_label"]:
                    flips.append(pair)
                if left["canonical_answer"] != right["canonical_answer"]:
                    canonical_flips.append(pair)
        if original["F2"]["classification"] != changed["F2"]["classification"]:
            open_edits.append(fact)
        if normalize(original["F2"]["text"]) != normalize(changed["F2"]["text"]):
            open_text_edits.append(fact)
    return {
        "choice_probes": len(probs) // 4, "max_probability_difference": max(probs), "mean_probability_difference": sum(probs) / len(probs),
        "max_choice_mass_difference": max(mass_diffs), "displayed_label_flips": flips, "canonical_flips": canonical_flips,
        "open_classification_changes": open_edits, "open_text_changes": open_text_edits,
    }


def evaluate_gpu(mode, cfg, paths, out):
    import torch
    from peft import PeftModel
    from krmm.model import Omni
    from krmm.v3.damage import restore_weights, snapshot_weights, state_sha256

    source, _ = paths
    replay = source / "replay"
    old = read(ROOT / "configs/v6/pilot.json")
    backend = Omni(old["model_path"])
    lora = old["lora"]
    initial = torch.load(replay / "damaged_state_M0.pt", map_location="cpu", weights_only=True)
    if state_sha256(initial) != cfg["m0_sha256"]:
        raise RuntimeError("State digest mismatch for rebuilt M0")
    if not initial or not all(weight.dtype == torch.bfloat16 for weight in initial.values()):
        raise RuntimeError("Unexpected M0 dtype")
    cases = make_questions(cfg, [json.loads(line) for line in (source / "private_predictions.jsonl").read_text().splitlines()])

    def reset():
        restore_weights(backend, initial)
        if state_sha256(snapshot_weights(backend, lora)) != cfg["m0_sha256"]:
            raise RuntimeError("M0 reset does not match the state digest")

    def attach(adapter_path):
        backend.model = PeftModel.from_pretrained(backend.model, str(adapter_path), is_trainable=False)
        backend.model.eval()

    reset()
    if mode == "primary":
        baseline = score_state(backend, cases, cfg)
        for fact, row, *_ in cases:
            stats = baseline[fact["id"]]["F0"]
            if stats["displayed_label"] != row["pred"] or max(abs(a - b) for a, b in zip(stats["probabilities"], row["probabilities"], strict=True)) > cfg["tolerance"]:
                raise RuntimeError("F0 cannot reproduce the rebuild M0")
        write(out / "m0_predictions.json", baseline)

    for branch in cfg["branches"]:
        reset()
        name = branch["label"]
        attach(replay / branch["adapter"])
        if mode == "primary":
            with torch.inference_mode():
                effective = prepared_delta(backend.model, initial)
            unmerged = score_state(backend, cases, cfg)
            write(out / f"{name}_unmerged_predictions.json", unmerged)
        backend.model = backend.model.merge_and_unload(safe_merge=True)
        backend.model.eval()
        merged = snapshot_weights(backend, lora)
        state_hash = state_sha256(merged)
        predictions = score_state(backend, cases, cfg)
        write(out / f"{name}_{mode}_predictions.json", predictions)
        if mode == "primary":
            result = {"merged_state_sha256": state_hash, "weights": merge_metrics(initial, merged, effective), "forward": compare_states(unmerged, predictions)}
            del effective
        else:
            anchor = read(out / f"{name}_primary_audit.json")
            comparison = compare_states(read(out / f"{name}_primary_predictions.json"), predictions)
            passed = state_hash == anchor["merged_state_sha256"] and comparison["max_probability_difference"] <= cfg["tolerance"] and not comparison["displayed_label_flips"]
            result = {"merged_state_sha256": state_hash, "matches_primary": passed, "forward": comparison}
        write(out / f"{name}_{mode}_audit.json", result)
        print(mode, name, "scored", flush=True)

    if mode == "primary":
        reset()
        for adapter in sorted((replay / cfg["d3_final_dir"]).iterdir()):
            if adapter.is_dir():
                attach(adapter)
                backend.model = backend.model.merge_and_unload(safe_merge=True)
        backend.model.eval()
        write(out / "d3_final_predictions.json", score_state(backend, cases, cfg))
        print("primary d3-final scored", flush=True)
    write(out / f"{mode}_compute.json", backend.state())


def assess(out, cfg):
    anchor = read(out / "m0_predictions.json")
    observations = {}
    for label in [item["label"] for item in cfg["branches"]] + ["d3_final"]:
        final = read(out / (f"{label}_primary_predictions.json" if label != "d3_final" else "d3_final_predictions.json"))
        observed = {}
        for fact_id, before in anchor.items():
            after = final[fact_id]
            f1 = [sum(row["correct"] for row in state["F1"]) for state in (before, after)]
            f3 = [sum(row["correct"] for row in state["F3"]) for state in (before, after)]
            f2 = [state["F2"]["classification"] for state in (before, after)]
            fixed_harm = before["F0"]["correct"] and not after["F0"]["correct"]
            category = "no_fixed_harm"
            if f1[0] >= 3 and f1[1] <= 1 and f3[0] >= 3 and f3[1] <= 1 and f2 == ["correct", "wrong"]:
                category = "semantic_harm_supported"
            elif fixed_harm:
                supported = min(*f1, *f3) >= 3 and f2 == ["correct", "correct"]
                category = "fixed_format_only_supported" if supported else "mixed_or_inconclusive"
            observed[fact_id] = {"role": before["role"], "fixed_harm": fixed_harm, "F1_correct_of_4": f1, "F3_correct_of_4": f3, "F2": f2, "category": category}
        observations[label] = observed
    reloads = {item["label"]: read(out / f'{item["label"]}_fresh_audit.json') for item in cfg["branches"]}
    write(out / "summary.json", {"replay_only": True, "format_observations": observations, "strict_merge_reload_pass": all(row["matches_primary"] for row in reloads.values()), "reloads": reloads})


def verify(cfg, source):
    verify_inputs(read(ROOT / "configs/v8/protocol.json"))
    manifest = source / "SHA256SUMS"
    if digest(manifest) != cfg["rebuild_manifest_sha256"]:
        raise RuntimeError("Rebuild artifact manifest digest mismatch")
    subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=source, check=True)
    make_questions(cfg, [json.loads(line) for line in (source / "private_predictions.jsonl").read_text().splitlines()])
    prior = read(source / "budget_usage.json")
    contract = read(ROOT / "configs/v8/protocol.json")
    cap = contract["stage_caps"]["V8-A"]
    if (
        sum(stage["max_seconds"] for stage in cfg["queue"]) / 3600 + prior["conservative_gpu_hours_upper_bound"] > cap["gpu_hours"]
        or prior["primary_launches"] + 1 > cap["primary_launches"]
        or prior["reload_children"] + 1 > cap["reload_children"]
        or [stage["kind"] for stage in cfg["queue"]] != ["primary", "independent_gpu_reload"]
    ):
        raise RuntimeError("Frozen diagnostic would overrun V8-A budget")
    print("Frozen inputs, facts, and 73 rebuild artifacts verified", flush=True)


def supervise(cfg, paths, out):
    source, parent = paths
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["gpu_uuid"]:
        raise RuntimeError("Unexpected GPU assignment")
    verify(cfg, source)
    os.umask(0o077)
    with (parent / ".rebuild.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial_free = shutil.disk_usage(parent).free
        if initial_free < cfg["minimum_free_bytes"] + cfg["max_added_bytes"]:
            raise RuntimeError("Insufficient disk headroom for frozen launch")
        out.mkdir()
        stats = []
        status = "FAILED"
        try:
            write(out / "provenance.json", {
                "source_commit": command("git", "rev-parse", "HEAD"),
                "protocol_sha256": digest(CONFIG), "queue_sha256": digest(QUEUE), "facts_sha256": digest(FACTS),
                "m0_sha256": cfg["m0_sha256"], "rebuild_manifest_sha256": cfg["rebuild_manifest_sha256"],
            })
            (out / "command.txt").write_text(shlex.join(cfg["command"]) + "\n")
            for mode, stage in zip(MODES, cfg["queue"], strict=True):
                free_vram = command("nvidia-smi", "-i", cfg["gpu_uuid"], "--query-gpu=memory.free", "--format=csv,noheader,nounits")
                if gpu_usage(cfg["gpu_uuid"]) or int(free_vram) < cfg["minimum_start_free_mib"]:
                    raise RuntimeError("GPU is busy or free VRAM under the frozen threshold")
                started = time.monotonic()
                launch = {"mode": mode, "kind": stage["kind"], "started_unix": time.time(), "argv": stage["argv"]}
                with (out / f"{mode}.stdout.log").open("x") as log, (out / f"{mode}.stderr.log").open("x") as err:
                    proc = subprocess.Popen(stage["argv"], cwd=ROOT, stdout=log, stderr=err, start_new_session=True)
                    stats.append(launch)
                    try:
                        while proc.poll() is None:
                            free = shutil.disk_usage(parent).free
                            if free < cfg["minimum_free_bytes"] or initial_free - free > cfg["max_added_bytes"]:
                                raise RuntimeError("Disk limit reached")
                            if time.monotonic() - started >= stage["max_seconds"]:
                                raise TimeoutError(f"{mode} timed out")
                            time.sleep(1)
                    finally:
                        if proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGTERM)
                            try:
                                proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid, signal.SIGKILL)
                                proc.wait()
                        launch.update({"seconds": time.monotonic() - started, "exit": proc.returncode})
                    if proc.returncode != 0:
                        raise subprocess.CalledProcessError(proc.returncode, stage["argv"])
            assess(out, cfg)
            status = "COMPLETE" if read(out / "summary.json")["strict_merge_reload_pass"] else "RELOAD_FAIL"
        finally:
            write(out / "budget_usage.json", {
                "status": status, "stages": stats, "fits": 0, "gpu_hours_upper_bound": sum(s["seconds"] for s in stats) / 3600,
                "added_bytes": initial_free - shutil.disk_usage(parent).free,
            })
            with (out / "SHA256SUMS").open("x") as manifest:
                for artifact in sorted(out.iterdir()):
                    if artifact.is_file() and artifact != out / "SHA256SUMS":
                        manifest.write(f"{digest(artifact)}  {artifact.name}\n")
        if status != "COMPLETE":
            raise RuntimeError(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify", "supervise", *MODES), required=True)
    mode = parser.parse_args().mode
    cfg, queue = read(CONFIG), read(QUEUE)
    if any(cfg[key] != queue[key] for key in ("run_id", "version")):
        raise RuntimeError("Protocol and queue mismatch")
    cfg.update(queue)
    paths = ROOT / cfg["rebuild_run"], ROOT / "results/v8-new"
    out = paths[1] / cfg["run_id"]
    if mode == "verify":
        verify(cfg, paths[0])
    elif mode == "supervise":
        supervise(cfg, paths, out)
    else:
        evaluate_gpu(mode, cfg, paths, out)


if __name__ == "__main__":
    main()
