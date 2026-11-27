#!/usr/bin/env python3
"""Bounded, evaluator-owned reconstruction of the historical KRMM Oracle replay."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import signal
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GIB = 1 << 30
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, record):
    with path.open("x") as output:
        json.dump(record, output, indent=2, allow_nan=False)
        output.write("\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(*argv):
    return subprocess.check_output(argv, cwd=ROOT, text=True).strip()


def verify_inputs(protocol, additional_paths=()):
    base = protocol["base_sha"]
    tracked = command("git", "diff", "--name-only", base, "--")
    allowed = ("scripts/v8/", "configs/v8/", "docs/v8/", "reports/v8/", *additional_paths)
    if any(not path.startswith(allowed) for path in tracked.splitlines()):
        raise RuntimeError("Protected source differs from the frozen parent")
    if command("git", "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Commit staged protocol and executable before launch")
    for rel, expected in protocol["files"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError(f"Input digest mismatch: {rel}")
    model = protocol["model"]
    directory = Path(model["path"])
    manifest_path = directory / "download.complete.json"
    if digest(manifest_path) != model["download_manifest_sha256"]:
        raise RuntimeError("Model manifest digest mismatch")
    manifest = read(manifest_path)
    if manifest["revision"] != model["revision"]:
        raise RuntimeError("Model revision mismatch")
    for name, item in manifest["files"].items():
        path = directory / name
        if path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
            raise RuntimeError(f"Cached shard mismatch: {name}")
    print("Verified frozen inputs and complete cached model SHA256", flush=True)


def gpu_usage(uuid):
    return command(
        "nvidia-smi", "-i", uuid,
        "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits",
    )


class Run:
    def __init__(self, root, protocol, entry):
        self.root = root
        self.replay = root / "replay"
        self.protocol = protocol
        self.entry = entry
        self.started = time.monotonic()
        self.fits = 0
        self.children = 0
        self.child_seconds = 0.0
        self.initial_free = shutil.disk_usage(ROOT).free
        self.floor = protocol["contract_caps"]["minimum_free_disk_gib"] * GIB
        self.disk_limit = entry["max_added_disk_gib"] * GIB
        self.last_controls = []

    def event(self, event, **fields):
        with (self.root / "step_ledger.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": event, "unix_s": time.time(), **fields}, allow_nan=False) + "\n")

    def check_resources(self):
        free = shutil.disk_usage(ROOT).free
        if free < self.floor or self.initial_free - free > self.disk_limit:
            raise RuntimeError("Disk limit reached")
        if time.monotonic() - self.started >= self.entry["max_seconds"]:
            raise TimeoutError("Primary wall clock limit reached")

    @contextmanager
    def fit(self, family, backend, steps, **details):
        import torch

        self.check_resources()
        if self.fits >= self.entry["max_fits"]:
            raise RuntimeError("Fit cap reached")
        self.fits += 1
        index = self.fits
        start = time.monotonic()
        before = backend.state()
        rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        cuda_rng = hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
        self.event("fit_start", fit=index, family=family, planned_steps=steps, torch_rng=rng, cuda_rng=cuda_rng, **details)
        status = "FAILED"
        try:
            yield
            status = "COMPLETE"
        finally:
            after = backend.state()
            self.event(
                "fit_end", fit=index, status=status, seconds=time.monotonic() - start,
                tokens=after["decoder_tokens_processed"] - before["decoder_tokens_processed"],
                forwards=after["model_forward_calls"] - before["model_forward_calls"],
                peak_allocated_gib=after["peak_vram_gb"],
            )
        self.check_resources()

    def fresh(self, job_path, python_exe):
        self.check_resources()
        if self.children >= len(self.entry["reloads"]):
            raise RuntimeError("Unapproved reload")
        spec = self.entry["reloads"][self.children]
        argv = [python_exe, str(ROOT / "scripts/v6/verify_fresh_subprocess.py"), "--job", str(job_path)]
        if argv != spec["argv"] or not self.last_controls:
            raise RuntimeError("Frozen reload mismatch or empty prediction probes")

        replay_job = read(job_path)
        replay_job["consistency_probes"] = [
            {"prompt": row["question"], "expected_probs": row["probabilities"]}
            for row in self.last_controls if row["branch"] == spec["label"]
        ]
        if not replay_job["consistency_probes"]:
            raise RuntimeError("Missing final predictions for reload")
        job_path.write_text(json.dumps(replay_job, indent=2, allow_nan=False) + "\n")
        self.children += 1
        start = time.monotonic()
        self.event("reload_start", label=spec["label"], argv=argv)
        status = "FAILED"
        try:
            with (self.root / f'{spec["label"]}.stdout.log').open("x") as out:
                with (self.root / f'{spec["label"]}.stderr.log').open("x") as err:
                    subprocess.run(argv, cwd=ROOT, stdout=out, stderr=err, check=True, timeout=spec["max_seconds"])
            result = read(replay_job["output"])
            if not all(result.get(key) for key in ("damaged_sha_verified", "fresh_reload_passed", "tolerance_passed")):
                raise RuntimeError(f'Reload failed frozen probability/state checks: {spec["label"]}')
            status = "COMPLETE"
            return {**result, "fresh_subprocess_ok": True}
        finally:
            elapsed = time.monotonic() - start
            self.child_seconds += elapsed
            self.event("reload_end", label=spec["label"], status=status, seconds=elapsed)

    def finish(self, status):
        elapsed = time.monotonic() - self.started
        write(self.root / "budget_usage.json", {
            "status": status, "primary_launches": 1, "reload_children": self.children,
            "fits_started": self.fits, "primary_wall_seconds": elapsed,
            "reload_seconds_included_in_primary_wall": self.child_seconds,
            "conservative_gpu_hours_upper_bound": (elapsed + self.child_seconds) / 3600,
            "filesystem_space_delta_bytes": self.initial_free - shutil.disk_usage(ROOT).free,
        })


def reconstruct(run):
    import numpy as np
    import torch
    from scripts.v7 import run_oracle_diagnosis as legacy

    seed = run.protocol["seeds"]["rebuild"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    damage, sequential, joint = legacy.inject_damage, legacy.fit_continual_step, legacy.fit_joint_multitarget
    phase = "M0"
    baseline_probs = None
    checked = False

    def inject(backend, unit, correct, incorrect, prompts, logits, cfg, **kwargs):
        with run.fit("damage", backend, cfg["damage_steps"], unit_id=unit.unit_id):
            result = damage(backend, unit, correct, incorrect, prompts, logits, cfg, **kwargs)
            if not all(math.isfinite(value) for row in result for value in row.values()):
                raise FloatingPointError("Nonfinite damage fit trace")
        return result

    def sequence(backend, unit, gold, prompts, logits, cfg, out, **kwargs):
        nonlocal phase
        phase = "D3_sequential_v6" if out.parent.name == "D3_seq_adapters" else "D2_single"
        with run.fit(phase, backend, cfg["repair_steps"], unit_id=unit.unit_id):
            result = sequential(backend, unit, gold, prompts, logits, cfg, out, **kwargs)
            if not math.isfinite(result["effective_delta_w_frobenius"]):
                raise FloatingPointError("Nonfinite effective update")
        return result

    def joint_fit(backend, pairs, prompts, logits, cfg, out):
        nonlocal phase
        phase = "D4_joint_multitarget"
        with run.fit(phase, backend, cfg["repair_steps"] * len(pairs)):
            result = joint(backend, pairs, prompts, logits, cfg, out)
            if not math.isfinite(result["effective_delta_w_frobenius"]):
                raise FloatingPointError("Nonfinite joint effective update")
        return result

    def controls(backend, prompts):
        nonlocal phase, baseline_probs
        if phase == "M0" and baseline_probs is not None:
            phase = "D0"
        # One forward per original control. Save the actual prediction and probability vector.
        results, probes = [], []
        for control in prompts:
            probs, mass = backend.probabilities(control["question"])
            if not all(math.isfinite(value) for value in [*probs, mass]):
                raise FloatingPointError("Nonfinite scored prediction")
            label = "ABCD"[int(np.argmax(probs))]
            row = {**control, "pred": label, "correct": label == control["gold"], "mass": mass}
            results.append(row)
            probes.append({**row, "branch": phase, "probabilities": probs})
        probability_snapshot = [row["probabilities"] for row in probes]
        if phase == "M0":
            baseline_probs = probability_snapshot
        elif phase == "D0" or phase.startswith("noop_"):
            if probability_snapshot != baseline_probs:
                raise RuntimeError(f"Fixed controls drifted under {phase}")
        run.check_resources()
        run.event("prediction_batch", branch=phase, count=len(probes), compute=backend.state())
        with (run.root / "private_predictions.jsonl").open("a") as stream:
            for row in probes:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        if phase in {"D3_sequential_v6", "D4_joint_multitarget"}:
            run.last_controls = [row for row in run.last_controls if row["branch"] != phase] + probes
        return results

    noop = legacy.run_noop_lora_merge

    def noop_check(backend, cfg, n_merges):
        nonlocal phase
        phase = f"noop_{n_merges}x"
        run.check_resources()
        result = noop(backend, cfg, n_merges=n_merges)
        if not result["hash_exact_match"] or result["max_effective_delta_w"] != 0:
            raise RuntimeError(f"Historical no-op control failed after {n_merges} merges")
        return result

    def check_g0(backend, units, eval_map, target_uids):
        nonlocal checked
        if not checked and (run.replay / "truth_records.json").exists():
            truth = read(run.replay / "truth_records.json")
            known = [row for row in truth if row["pool_role"] == "known_real"]
            ready = sum(row["clean_screen_passed"] and row["clean_heldout_accuracy"] >= 0.999 for row in known)
            degraded = sum(row["is_degraded_target"] for row in truth)
            ratio = ready / len(known)
            run.event("g0_replay_screen", mastered=ready, eligible=len(known), degraded=degraded)
            if ratio < run.protocol["validity"]["minimum_clean_mastery"] or degraded < run.protocol["validity"]["minimum_degraded_targets"]:
                raise RuntimeError(f"Replay G0 failed: clean mastery {ready}/{len(known)}, D={degraded}")
            checked = True
        return heldout(backend, units, eval_map, target_uids)

    heldout = legacy._eval_heldout_map
    legacy.inject_damage = inject
    legacy.fit_continual_step = sequence
    legacy.fit_joint_multitarget = joint_fit
    legacy._eval_controls = controls
    legacy._eval_heldout_map = check_g0
    legacy.run_noop_lora_merge = noop_check
    legacy.run_fresh_subprocess_audit = run.fresh
    sys.argv = [str(ROOT / "scripts/v7/run_oracle_diagnosis.py"), "--config", "configs/v6/pilot.json",
                "--out", str(run.replay), "--max-targets", "8"]
    legacy.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    protocol_path = ROOT / "configs/v8/protocol.json"
    queue_path = ROOT / "configs/v8/run_queue.json"
    protocol, queue = read(protocol_path), read(queue_path)
    if queue["version"] != protocol["version"]:
        raise RuntimeError("Protocol and queue version mismatch")
    entry, = [candidate for candidate in queue["entries"] if candidate["id"] == args.run]
    verify_inputs(protocol)
    if args.verify_only:
        return

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    os.umask(0o077)
    gpu = protocol["gpu"]
    result_root = ROOT / "results/v8-new"
    result_root.mkdir(exist_ok=True)
    if result_root.is_symlink():
        raise RuntimeError("Unexpected result root symlink")
    with (result_root / ".rebuild.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = result_root / entry["id"]
        if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu["uuid"] or gpu_usage(gpu["uuid"]):
            raise RuntimeError("Frozen GPU visibility mismatch or device already occupied")
        free_mib = command("nvidia-smi", "-i", gpu["uuid"], "--query-gpu=memory.free", "--format=csv,noheader,nounits")
        if int(free_mib) < gpu["min_start_free_mib"]:
            raise RuntimeError("Insufficient starting VRAM")
        out.mkdir()
        run = Run(out, protocol, entry)
        status = "FAILED"
        try:
            run.check_resources()
            write(out / "provenance.json", {
                "source_commit": command("git", "rev-parse", "HEAD"),
                "protocol_sha256": digest(protocol_path), "queue_sha256": digest(queue_path),
                "input_sha256": protocol["files"], "model": protocol["model"], "seed": protocol["seeds"],
                "gpu": command("nvidia-smi", "-i", gpu["uuid"], "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"),
                "dependencies": {package: importlib.metadata.version(package) for package in ("torch", "transformers", "peft", "numpy")},
            })
            (out / "command.txt").write_text(shlex.join(entry["argv"]) + "\n")
            reconstruct(run)
            summary = read(run.replay / "diagnosis_summary.json")
            status = "REPLAY_RECONSTRUCTED"
            write(out / "gate_result.json", {
                "status": status, "m0_sha256": summary["m0_sha256"],
                "original_m0_sha256": protocol["history"]["historical_m0_sha256"],
                "original_m0_hash_matches": summary["m0_sha256"] == protocol["history"]["historical_m0_sha256"],
                "historical_integrity": "INCOMPLETE_ORIGINAL_PRESERVED",
                "format_measurement": "NOT_RUN", "nonzero_merge_comparison": "NOT_RUN", "ow_confirm": "NOT_RUN",
            })
        finally:
            run.finish(status)
            with (out / "SHA256SUMS").open("x") as manifest:
                for path in sorted(out.rglob("*")):
                    if path.is_file() and path.name != "SHA256SUMS":
                        manifest.write(f"{digest(path)}  {path.relative_to(out)}\n")


if __name__ == "__main__":
    main()
