#!/usr/bin/env python3
"""Frozen queue for a first-step merge pilot and conditional DEV writer matrix."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.v8.dev_workers import data_paths, writer
from scripts.v8.precision_pilot_worker import compare_step, gate
from scripts.v8.rebuild_replay import command, digest, gpu_usage, read, verify_inputs, write

CONFIG = ROOT / "configs/v8/precision_pilot/protocol.json"
QUEUE = ROOT / "configs/v8/precision_pilot/run_queue.json"
DEV_CONFIG = ROOT / "configs/v8/dev_pilot02_resume/protocol.json"
RESULTS = ROOT / "results/v8-new"
MODES = ("pilot", "pilot-fresh", "writer", "observe")


def verify(precision, cfg, queue):
    verify_inputs(read(ROOT / "configs/v8/protocol.json"), ("krmm/v8/", "data/v8/", "tests/v8/"))
    if digest(DEV_CONFIG) != precision["dev_config_sha256"] or cfg["bf16_residual"] != precision["residual_threshold"]:
        raise RuntimeError("Prior thresholds or parameters differ")
    for path, key in zip(data_paths(cfg), ("public_sha256", "private_sha256"), strict=True):
        if digest(path) != cfg[key]:
            raise RuntimeError("Frozen prior dataset differs")
    prior = read(ROOT / "reports/v8/dev_v8_pilot02.json")
    if prior["ow_pilot"] != "FAIL" or prior["g0"]["m0_sha256"] != precision["m0_sha256"]:
        raise RuntimeError("Prior gate or M0 provenance was not retained")
    for run, record in prior["runs"].items():
        directory = RESULTS / run
        if digest(directory / "SHA256SUMS") != record["sha256"]:
            raise RuntimeError(f"Previous run manifest differs: {run}")
        subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=directory, check=True)
    source = RESULTS / precision["archive_run"]
    requests = read(source / "oracle_input/requests.json")
    trajectory = read(source / "writer/WP/trajectory.json")
    first = trajectory["attempts"][0]
    if (
        requests["state_sha256"] != precision["m0_sha256"]
        or requests["requests"][0]["unit_id"] != precision["first_unit_id"]
        or first["pre_sha256"] != precision["m0_sha256"] or first["committed"]
        or first["public_prompt_sha256"] != precision["public_prompt_sha256"]
        or first["candidate_sha256"] != precision["archived_bf16_candidate_sha256"]
        or first["adapter"] != "WP/step-01"
        or digest(source / "writer/WP/step-01/adapter_model.safetensors") != precision["adapter_weights_sha256"]
    ):
        raise RuntimeError("First-step lineage differs")
    prior_a = read(ROOT / "reports/v8/format_merge.json")["v8_a_cumulative"]
    prior_b = prior["pilot_budget"]
    contract = read(ROOT / "configs/v8/protocol.json")
    total = contract["contract_caps"]
    stage = contract["stage_caps"]["V8-B"]
    max_hours = sum(item["seconds"] for item in queue["stages"]) / 3600
    used_hours = prior["v8_b_cumulative_gpu_hours"]
    used_b_main = prior_b["primary_launches"] + read(ROOT / "reports/v8/dev_v8.json")["budget"]["primary_launches"]
    used_b_child = prior_b["reload_children"]
    if (
        tuple(entry["mode"] for entry in queue["stages"]) != MODES
        or max_hours + used_hours > stage["gpu_hours"]
        or used_b_main + 2 > stage["primary_launches"] or used_b_child + 2 > stage["reload_children"]
        or max_hours + used_hours + prior_a["gpu_hours"] > total["gpu_hours"]
        or prior_a["primary_launches"] + used_b_main + 2 > total["primary_launches"]
        or prior_a["reload_children"] + used_b_child + 2 > total["reload_children"]
        or prior_a["fits"] + prior_b["damage_fits"] + prior_b["writer_fits"] + cfg["max_writer_fits"] > total["fits"]
    ):
        raise RuntimeError("Precision queue would exceed the remaining contract")
    print("Earlier failures, model shards, dataset, M0 and first-step adapter verified", flush=True)


def prepare_archive(precision, out):
    source = RESULTS / precision["archive_run"]
    for name in (
        "private/g0.json", "private/truth.json", "private/damaged_targets.json", "private/m0_controls.json",
        "oracle_input/m0.pt", "oracle_input/requests.json",
    ):
        shutil.copyfile(source / name, out / name)
        if digest(source / name) != digest(out / name):
            raise RuntimeError(f"Archived input transfer failed: {name}")
    original = source / "writer/WP/step-01"
    local = out / "oracle_input/archived_wp"
    local.mkdir()
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copyfile(original / filename, local / filename)
        if digest(original / filename) != digest(local / filename):
            raise RuntimeError(f"Archived adapter transfer failed: {filename}")
    write(out / "oracle_input/first_request.json", read(out / "oracle_input/requests.json")["requests"][0])


def execute(stage, out, initial_disk, cfg):
    uuid = cfg["gpu_uuid"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid or gpu_usage(uuid):
        raise RuntimeError("Assigned GPU is not isolated or is occupied")
    free_mib = command("nvidia-smi", "-i", uuid, "--query-gpu=memory.free", "--format=csv,noheader,nounits")
    if int(free_mib) < 23000:
        raise RuntimeError("Insufficient free GPU memory")
    kind = stage["mode"]
    with (out / f"{kind}.stdout.log").open("x") as stdout, (out / f"{kind}.stderr.log").open("x") as stderr:
        process = subprocess.Popen(stage["argv"], cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
        start = time.monotonic()
        try:
            while process.poll() is None:
                free = shutil.disk_usage(RESULTS).free
                if free < cfg["disk_floor_bytes"] or initial_disk - free > cfg["add_disk_bytes"]:
                    raise RuntimeError("Precision pilot disk limit exceeded")
                if time.monotonic() - start > stage["seconds"]:
                    raise TimeoutError(f"{kind} exceeded the frozen wall budget")
                time.sleep(1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            with (out / "launches.jsonl").open("a") as log:
                log.write(json.dumps({"mode": kind, "exit": process.returncode,
                                      "seconds": time.monotonic() - start, "argv": stage["argv"]}) + "\n")
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, stage["argv"])


def supervise(precision, cfg, queue, out):
    verify(precision, cfg, queue)
    os.umask(0o077)
    with (RESULTS / ".rebuild.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial_disk = shutil.disk_usage(RESULTS).free
        if initial_disk < cfg["disk_floor_bytes"] + cfg["add_disk_bytes"]:
            raise RuntimeError("Insufficient workspace headroom")
        out.mkdir()
        for folder in ("private", "oracle_input", "writer", "writer/scratch"):
            (out / folder).mkdir()
        status = "FAILED"
        try:
            prepare_archive(precision, out)
            write(out / "provenance.json", {
                "source_commit": command("git", "rev-parse", "HEAD"),
                "protocol_sha256": digest(CONFIG), "queue_sha256": digest(QUEUE),
                "dev_config_sha256": digest(DEV_CONFIG), "m0_sha256": precision["m0_sha256"],
            })
            for stage in queue["stages"]:
                execute(stage, out, initial_disk, cfg)
                if stage["mode"] == "pilot-fresh":
                    decision = gate(read(out / "writer/pilot_compare.json"), read(out / "writer/pilot_fresh.json"),
                                    precision["residual_threshold"], cfg["reload_tolerance"])
                    write(out / "single_step_gate.json", decision)
                    print("Single-step gate", decision, flush=True)
                    if not decision["eligible_merge_dtype"]:
                        status = "STOP_SINGLE_STEP_GATE"
                        break
            else:
                status = "COMPLETE"
        finally:
            log = out / "launches.jsonl"
            launches = [json.loads(row) for row in log.read_text().splitlines()] if log.exists() else []
            fits = out / "writer/fits.jsonl"
            fit_count = sum("fit_start" in json.loads(row) for row in fits.read_text().splitlines()) if fits.exists() else 0
            write(out / "budget_usage.json", {
                "status": status, "stages": launches, "fits": fit_count,
                "gpu_hours": sum(item["seconds"] for item in launches) / 3600,
                "primary_launches": sum(item["mode"] in ("pilot", "writer") for item in launches),
                "reload_children": sum(item["mode"] in ("pilot-fresh", "observe") for item in launches),
                "added_bytes": initial_disk - shutil.disk_usage(RESULTS).free,
            })
            with (out / "SHA256SUMS").open("x") as manifest:
                for path in sorted(out.rglob("*")):
                    if path.is_file() and path.name != "SHA256SUMS":
                        manifest.write(f"{digest(path)}  {path.relative_to(out)}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify", "supervise", *MODES), required=True)
    mode = parser.parse_args().mode
    precision, cfg, queue = read(CONFIG), read(DEV_CONFIG), read(QUEUE)
    if precision["run_id"] != queue["run_id"]:
        raise RuntimeError("Precision protocol and queue disagree")
    out = RESULTS / precision["run_id"]
    cfg.update(branches=["WP", "WN", "WR"], max_writer_fits=24)
    if mode == "verify":
        verify(precision, cfg, queue)
    elif mode == "supervise":
        supervise(precision, cfg, queue, out)
    elif mode in ("pilot", "pilot-fresh"):
        compare_step(out, cfg, precision, independent=mode == "pilot-fresh")
    else:
        decision = read(out / "single_step_gate.json")
        dtype = decision["eligible_merge_dtype"]
        if decision["status"] != "PASS" or dtype not in {"FP32", "BF16"} or not decision["routes"][dtype]["passed"]:
            raise RuntimeError("No verified merge dtype passed the single-step gate")
        cfg["merge_dtype"] = dtype
        if mode == "writer":
            writer(out, cfg)
        else:
            from scripts.v8.dev_observer import observe
            observe(out, cfg)


if __name__ == "__main__":
    main()
