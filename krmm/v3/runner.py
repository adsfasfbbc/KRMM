"""Controlled experiment runner with resource, budget, hash, and resume guards (13.3)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch

from krmm.model import Omni
from .damage import balanced_wrong_assignments, snapshot_weights
from .evaluator import evaluate_one_unit
from .knowledge_registry import all_evaluator_units, manifest, train_retain_prompts
from .metrics import summarize_p0_results


def source_tree_sha256(root: Path) -> str:
    h = hashlib.sha256()
    v3_dir = root / "krmm" / "v3"
    for p in sorted(v3_dir.glob("*.py")):
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def git_commit_sha(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception:
        return "unknown"


def check_resources(min_free_disk_gb: float = 8.0, min_free_vram_mb: float = 1000.0) -> Dict[str, object]:
    usage = shutil.disk_usage("/root/rivermind-data" if Path("/root/rivermind-data").exists() else "/")
    free_gb = usage.free / (2**30)
    if free_gb < min_free_disk_gb:
        raise RuntimeError(f"Insufficient free disk: {free_gb:.2f} GiB < {min_free_disk_gb} GiB")

    vram_free_mb = 0.0
    gpu_name = "cpu"
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info(0)
        vram_free_mb = free_b / (2**20)
        gpu_name = torch.cuda.get_device_name(0)
        if vram_free_mb < min_free_vram_mb:
            raise RuntimeError(f"Insufficient free GPU VRAM: {vram_free_mb:.1f} MiB < {min_free_vram_mb} MiB")

    return {
        "free_disk_gib": free_gb,
        "free_vram_mib": vram_free_mb,
        "gpu_name": gpu_name,
    }


def check_budget_guard(budget_file: Optional[Path], max_launches: int) -> Dict[str, object]:
    """Enforce bounded launch budget (13.3, test_budget_guard)."""
    if budget_file is None:
        return {"launches_used": 1, "max_launches": max_launches}
    if budget_file.exists():
        state = json.loads(budget_file.read_text(encoding="utf-8"))
    else:
        state = {"launches_used": 0, "history": []}
    if state["launches_used"] >= max_launches:
        raise RuntimeError(f"Budget guard triggered: launches_used={state['launches_used']} >= max_launches={max_launches}")
    state["launches_used"] += 1
    state["history"].append({"utc": datetime.now(timezone.utc).isoformat()})
    budget_file.parent.mkdir(parents=True, exist_ok=True)
    budget_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def run_p0_experiment(
    cfg_path: Path,
    out_dir: Path,
    split: str = "dev",
    limit: int = 12,
    domain: Optional[str] = None,
    budget_file: Optional[Path] = None,
    resume: bool = False,
) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    max_launches = int(cfg.get("max_launches", 20))
    check_budget_guard(budget_file, max_launches)
    res_info = check_resources(
        min_free_disk_gb=float(cfg.get("min_free_disk_gb", 8.0)),
        min_free_vram_mb=float(cfg.get("min_free_vram_mb", 1000.0)),
    )

    if out_dir.exists() and not resume:
        raise FileExistsError(f"Run output directory already exists (unique dir required): {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(8)
    seed = int(cfg.get("seed", 107))
    torch.manual_seed(seed)
    np.random.seed(seed)

    meta = {
        "utc_start": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_sha(repo_root),
        "v3_source_sha256": source_tree_sha256(repo_root),
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "catalog_manifest": manifest(),
        "resources_precheck": res_info,
        "split": split,
        "domain": domain,
        "limit": limit,
        "config": cfg,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if split == "test" and not bool(cfg.get("allow_sealed_test", False)):
        raise PermissionError("Sealed test split cannot be queried without explicit allow_sealed_test=True after G0/G1 freeze.")

    all_split = all_evaluator_units(split=split, domain=domain)
    if domain is None:
        # Interleave domains round-robin so bounded pilots cover all domains evenly
        by_dom: Dict[str, List] = {}
        for item in all_split:
            by_dom.setdefault(item.unit.domain, []).append(item)
        interleaved = []
        dom_keys = sorted(by_dom.keys())
        max_len = max((len(v) for v in by_dom.values()), default=0)
        for idx in range(max_len):
            for dk in dom_keys:
                if idx < len(by_dom[dk]):
                    interleaved.append(by_dom[dk][idx])
        candidates = interleaved[:limit]
    else:
        candidates = all_split[:limit]
    private_dir = Path("/root/rivermind-data/setup-v03/private-assignments")
    private_dir.mkdir(parents=True, exist_ok=True)
    private_path = private_dir / f"{out_dir.name}.json"

    if resume and private_path.exists():
        assign_data = json.loads(private_path.read_text(encoding="utf-8"))
        assignment = assign_data["targets"]
    else:
        private_seed = secrets.randbits(64)
        assignment, audit = balanced_wrong_assignments(candidates, private_seed)
        private_path.write_text(json.dumps({"seed": private_seed, "targets": assignment}, indent=2), encoding="utf-8")
        os.chmod(private_path, 0o600)
        (out_dir / "damage-randomization-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    omni = Omni(str(cfg["model_path"]))
    clean_state = snapshot_weights(omni, cfg)
    retains = train_retain_prompts(include_multi_template=bool(cfg.get("multi_template_retain", True)))
    teacher_logits = omni.retain_logits(retains)

    records: List[Dict[str, object]] = []
    t_start = time.monotonic()
    timeout_sec = float(cfg.get("per_run_timeout_minutes", 90)) * 60.0

    for ev in candidates:
        unit_file = out_dir / f"{ev.unit.unit_id}.json"
        if resume and unit_file.exists():
            rec = json.loads(unit_file.read_text(encoding="utf-8"))
            records.append(rec)
            print("RESUME_SKIP", ev.unit.entity_id, flush=True)
            continue
        if time.monotonic() - t_start > timeout_sec:
            print("TIMEOUT_STOP", ev.unit.entity_id, flush=True)
            break
        check_resources(
            min_free_disk_gb=float(cfg.get("min_free_disk_gb", 8.0)),
            min_free_vram_mb=200.0,
        )
        print("START", ev.unit.domain, ev.unit.entity_id, flush=True)
        rec = evaluate_one_unit(
            omni,
            ev,
            assignment[ev.unit.unit_id],
            cfg,
            out_dir,
            clean_state,
            retains,
            teacher_logits,
            use_fresh_subprocess=bool(cfg.get("use_fresh_subprocess", True)),
        )
        unit_file.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        records.append(rec)
        print(
            "END",
            ev.unit.entity_id,
            "mastered=", rec["mastery"]["stably_mastered"],
            "op_elig=", rec["operational_eligible"],
            "strict_elig=", rec["strict_eligible"],
            "oracle_strict=", rec.get("repairs", {}).get("oracle", {}).get("strict_restoration"),
            flush=True,
        )

    summary = summarize_p0_results(records, cfg.get("gate_thresholds"))
    summary["model_resources"] = omni.state()
    summary["utc_end"] = datetime.now(timezone.utc).isoformat()
    summary["elapsed_seconds"] = time.monotonic() - t_start
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--budget-file", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run_p0_experiment(
        cfg_path=Path(args.config),
        out_dir=Path(args.out),
        split=args.split,
        limit=args.limit,
        domain=args.domain,
        budget_file=Path(args.budget_file) if args.budget_file else None,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
