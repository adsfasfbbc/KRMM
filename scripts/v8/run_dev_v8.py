#!/usr/bin/env python3
"""Bounded DEV-V8 queue; evaluates gates before opening downstream GPU stages."""
from argparse import ArgumentParser
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.v8.dev_workers import PRIVATE, PROTOCOL, PUBLIC, data_paths, prepare, writer
from scripts.v8.rebuild_replay import command, digest, gpu_usage, read, verify_inputs, write

QUEUE = ROOT / "configs/v8/dev_run_queue.json"
STAGES = ("prepare", "writer", "observe")
RESULTS = ROOT / "results/v8-new"


def verify(cfg, queue):
    from krmm.v3.knowledge_registry import all_evaluator_units
    from krmm.v5.v5_catalog import all_v5_entities
    from krmm.v6.v6_catalog import build_v6_pilot_catalog, normalize_entity_name
    from krmm.v7.dev_catalog import build_dev_v7_catalog

    verify_inputs(read(ROOT / "configs/v8/protocol.json"), ("krmm/v8/", "data/v8/", "tests/v8/"))
    public_path, private_path = data_paths(cfg)
    for path, key in ((public_path, "public_sha256"), (private_path, "private_sha256")):
        if digest(path) != cfg[key]:
            raise RuntimeError(f"Frozen manifest mismatch: {path}")
    public, hidden = read(public_path), read(private_path)
    if [row["unit_id"] for row in public] != [row["unit_id"] for row in hidden] or len({row["unit_id"] for row in public}) != 36:
        raise RuntimeError("Manifest IDs disagree")
    if sorted(Counter(row["domain"] for row in public).values()) != [12, 12, 12]:
        raise RuntimeError("Expected 12 entities per historical domain")
    if any(len(row["controls"]) != 2 or not row["source"].startswith("https://") for row in hidden):
        raise RuntimeError("Missing immutable controls or factual provenance")

    v6, evaluator, _ = build_v6_pilot_catalog()
    history = (
        {row.unit.entity_id for row in all_evaluator_units()} | set(all_v5_entities())
        | {row.entity_id for row in v6} | {alias for row in evaluator for alias in row.aliases}
        | {row.entity_id for row in build_dev_v7_catalog()[0]}
    )
    def normalize(name):
        folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
        return normalize_entity_name(folded)

    earlier = cfg.get("earlier_failed_pilot")
    if earlier:
        earlier_catalog = read(PUBLIC)
        earlier_ids = {row["unit_id"] for row in earlier_catalog if row["domain"] == "sound_source_mechanism"}
        earlier_answers = read(PRIVATE)
        unchanged = [row for row in public if row["domain"] != "sound_source_mechanism"]
        unchanged_ids = {row["unit_id"] for row in unchanged}
        if (
            unchanged != [row for row in earlier_catalog if row["unit_id"] not in earlier_ids]
            or [row for row in hidden if row["unit_id"] in unchanged_ids]
            != [row for row in earlier_answers if row["unit_id"] not in earlier_ids]
            or earlier_ids.intersection(row["unit_id"] for row in public)
        ):
            raise RuntimeError("Pilot should only replace the earlier 12 instrument entities")
        history.update(row["entity_id"] for row in earlier_catalog if row["unit_id"] in earlier_ids)
        history.update(alias for row in earlier_answers if row["unit_id"] in earlier_ids for alias in row["aliases"])
        earlier_run = RESULTS / earlier
        if digest(earlier_run / "SHA256SUMS") != cfg["earlier_manifest_sha256"]:
            raise RuntimeError("Earlier pilot checksum manifest was changed")
        subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=earlier_run, check=True)

    old_names = {normalize(name) for name in history}
    new_names = {normalize(row["entity_id"]) for row in public}
    new_names.update(normalize(alias) for row in hidden for alias in row["aliases"])
    overlap = sorted(old_names & new_names)
    if overlap:
        raise RuntimeError(f"Normalized historical overlap: {overlap}")
    prior = read(ROOT / "reports/v8/format_merge.json")["v8_a_cumulative"]
    contract = read(ROOT / "configs/v8/protocol.json")
    caps = contract["contract_caps"]
    stage = contract["stage_caps"]["V8-B"]
    wall_hours = sum(entry["seconds"] for entry in queue["stages"]) / 3600
    prepared = cfg.get("resume_from")
    preceding = [read(RESULTS / earlier / "budget_usage.json")] if earlier else []
    if prepared:
        source = RESULTS / prepared
        if digest(source / "SHA256SUMS") != cfg["prepared_manifest_sha256"]:
            raise RuntimeError("Prepared episode manifest changed")
        subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=source, check=True)
        g0 = read(source / "private/g0.json")
        exchange = read(source / "oracle_input/requests.json")
        old_budget = read(source / "budget_usage.json")
        originals = read(ROOT / "configs/v8/dev_pilot02/protocol.json")
        if (
            g0["status"] != "PASS" or g0["m0_sha256"] != exchange["state_sha256"]
            or old_budget["status"] != "FAILED" or old_budget["fits"]["writer"] != 0
            or any(cfg[key] != value for key, value in originals.items() if key != "id")
            or g0["m0_sha256"] != cfg["prepared_m0_sha256"]
        ):
            raise RuntimeError("Resume requires the original passing G0 and zero earlier writer fits")
        preceding.append(old_budget)
        for probe in cfg["cuda_probes"]:
            directory = ROOT / probe
            seconds = float((directory / "end").read_text()) - float((directory / "start").read_text())
            preceding.append({"gpu_hours": seconds / 3600, "primary_launches": 1, "reload_children": 0, "fits": {}})

    used_hours = sum(item["gpu_hours"] for item in preceding)
    used_main = sum(item["primary_launches"] for item in preceding)
    used_fresh = sum(item["reload_children"] for item in preceding)
    used_fits = sum(sum(item["fits"].values()) for item in preceding)
    planned_modes = list(STAGES[1:] if prepared else STAGES)
    new_main = len(planned_modes) - 1
    if (
        [entry["mode"] for entry in queue["stages"]] != planned_modes
        or wall_hours + used_hours > stage["gpu_hours"]
        or used_main + new_main > stage["primary_launches"]
        or used_fresh + 1 > stage["reload_children"]
        or wall_hours + used_hours + prior["gpu_hours"] > caps["gpu_hours"]
        or prior["primary_launches"] + used_main + new_main > caps["primary_launches"]
        or prior["reload_children"] + used_fresh + 1 > caps["reload_children"]
        or prior["fits"] + used_fits + cfg["max_writer_fits"] + (0 if prepared else len(public)) > caps["fits"]
    ):
        raise RuntimeError("DEV queue would exceed its frozen budget")
    return {"old_normalized_entities_aliases": len(old_names), "new_normalized_entities_aliases": len(new_names),
            "source_groups": len({row["source_group"] for row in public}), "overlap": overlap}


def supervise(cfg, queue, out, protocol_path, queue_path):
    overlap = verify(cfg, queue)
    os.umask(0o077)
    with (RESULTS / ".rebuild.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = shutil.disk_usage(RESULTS).free
        if initial < cfg["disk_floor_bytes"] + cfg["add_disk_bytes"]:
            raise RuntimeError("Insufficient storage for DEV")
        out.mkdir()
        for folder in ("private", "oracle_input", "writer", "writer/scratch"):
            (out / folder).mkdir()
        prepared = cfg.get("resume_from")
        if prepared:
            source = RESULTS / prepared
            for name in (
                "private/g0.json", "private/truth.json", "private/damaged_targets.json", "private/m0_controls.json",
                "oracle_input/m0.pt", "oracle_input/requests.json",
            ):
                shutil.copyfile(source / name, out / name)
                if digest(source / name) != digest(out / name):
                    raise RuntimeError(f"Prepared input copy failed: {name}")
        write(out / "provenance.json", {
            "source_commit": command("git", "rev-parse", "HEAD"),
            "protocol_sha256": digest(protocol_path), "queue_sha256": digest(queue_path),
            "public_sha256": cfg["public_sha256"], "private_sha256": cfg["private_sha256"],
            "overlap_audit": overlap, "prepared_from": prepared,
            "prepared_manifest_sha256": cfg.get("prepared_manifest_sha256"),
        })
        track = []
        status = "FAILED"
        try:
            for stage in queue["stages"]:
                mode = stage["mode"]
                uuid = cfg["gpu_uuid"]
                if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid or gpu_usage(uuid):
                    raise RuntimeError("Visible device mismatch or GPU occupied")
                free_mib = command("nvidia-smi", "-i", uuid, "--query-gpu=memory.free", "--format=csv,noheader,nounits")
                if int(free_mib) < 23000:
                    raise RuntimeError("GPU memory floor not met")
                with (out / f"{mode}.stdout.log").open("x") as stdout, (out / f"{mode}.stderr.log").open("x") as stderr:
                    proc = subprocess.Popen(stage["argv"], cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
                    started = time.monotonic()
                    line = {"mode": mode, "started_unix": time.time(), "argv": stage["argv"]}
                    track.append(line)
                    try:
                        while proc.poll() is None:
                            free = shutil.disk_usage(RESULTS).free
                            if free < cfg["disk_floor_bytes"] or initial - free > cfg["add_disk_bytes"]:
                                raise RuntimeError("DEV disk guard triggered")
                            if time.monotonic() - started > stage["seconds"]:
                                raise TimeoutError(f"{mode} exceeded wall budget")
                            time.sleep(1)
                    finally:
                        if proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGTERM)
                            try:
                                proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid, signal.SIGKILL)
                                proc.wait()
                        line.update(seconds=time.monotonic() - started, exit=proc.returncode)
                    if proc.returncode:
                        raise subprocess.CalledProcessError(proc.returncode, stage["argv"])
                if mode == "prepare" and read(out / "private/g0.json")["status"] != "PASS":
                    status = read(out / "private/g0.json")["status"]
                    write(out / "dev_summary.json", {
                        "g0": read(out / "private/g0.json"), "branches": {}, "ow_pilot": "NOT_RUN_GATED",
                        "decision": "DO_NOT_ENTER_TEST", "ow_confirm": "NOT_RUN",
                    })
                    break
            else:
                status = "COMPLETE"
        finally:
            fits = {}
            for key, relative in (("damage", "private/damage_fits.jsonl"), ("writer", "writer/fits.jsonl")):
                log = out / relative
                fits[key] = sum("fit_start" in json.loads(line) for line in log.read_text().splitlines()) if log.exists() else 0
            write(out / "budget_usage.json", {
                "status": status, "stages": track, "fits": fits,
                "gpu_hours": sum(stage["seconds"] for stage in track) / 3600,
                "primary_launches": sum(stage["mode"] != "observe" for stage in track),
                "reload_children": sum(stage["mode"] == "observe" for stage in track),
                "added_bytes": initial - shutil.disk_usage(RESULTS).free,
            })
            with (out / "SHA256SUMS").open("x") as manifest:
                for path in sorted(out.rglob("*")):
                    if path.is_file() and path.name != "SHA256SUMS":
                        manifest.write(f"{digest(path)}  {path.relative_to(out)}\n")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify", "supervise", *STAGES), required=True)
    parser.add_argument("--pilot", choices=("01", "02", "02-resume"), default="01")
    args = parser.parse_args()
    mode = args.mode
    config_root = ROOT / ("configs/v8/dev_pilot02_resume" if args.pilot == "02-resume" else "configs/v8/dev_pilot02")
    protocol_path, queue_path = (PROTOCOL, QUEUE) if args.pilot == "01" else (
        config_root / "protocol.json", config_root / "run_queue.json",
    )
    cfg, queue = read(protocol_path), read(queue_path)
    if cfg["id"] != queue["id"]:
        raise RuntimeError("Frozen DEV protocol and queue disagree")
    out = RESULTS / cfg["id"]
    if mode == "verify":
        print(verify(cfg, queue))
    elif mode == "supervise":
        supervise(cfg, queue, out, protocol_path, queue_path)
    elif mode == "prepare":
        prepare(out, cfg)
    elif mode == "writer":
        writer(out, cfg)
    else:
        from scripts.v8.dev_observer import observe
        observe(out, cfg)


if __name__ == "__main__":
    main()
