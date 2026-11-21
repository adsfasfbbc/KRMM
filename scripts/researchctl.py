#!/usr/bin/env python3
"""Bounded experiment runner and auditable numerical gates for Conference AutoResearch.

This is NOT an OS sandbox, an autonomous coding agent, or a claim-validity judge.
Only Python's standard library is required (Python 3.9+).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

GATE_IDS = ["G0", "G1", "G2", "G3"]
OPERATORS = {">=", ">", "<=", "<", "==", "is_true", "is_false"}
ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"


class ResearchError(Exception):
    pass


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def token():
    return uuid.uuid4().hex[:10]


def canonical_digest(blob: bytes):
    return hashlib.sha256(blob).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{token()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def load_json(path: Path):
    def reject_nonfinite(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, parse_constant=reject_nonfinite)
    except (OSError, json.JSONDecodeError, ValueError) as err:
        raise ResearchError(f"Cannot read JSON {path}: {err}") from err


def protocol_at(ws: Path):
    path = ws / "protocol.json"
    if not path.is_file():
        raise ResearchError(f"No protocol at {path}. Run 'init' first.")
    raw = path.read_bytes()
    config = load_json(path)
    if not isinstance(config, dict):
        raise ResearchError("protocol.json must contain a JSON object")
    return config, canonical_digest(raw)


def meaningful(value):
    return isinstance(value, str) and bool(value.strip()) and not value.startswith("REPLACE")


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_protocol(config):
    errors = []
    if config.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    for name in ("project_id", "research_question", "primary_metric", "dataset_manifest", "holdout_policy", "protocol_version"):
        if not meaningful(config.get(name)):
            errors.append(f"{name}: replace placeholder with a project-specific value")
    if not meaningful(config.get("target_venue")):
        errors.append("target_venue must be a non-empty venue or 'undecided'")
    root_str = config.get("repo_root")
    if not meaningful(root_str) or not Path(root_str).is_absolute() or not Path(root_str).is_dir():
        errors.append("repo_root must be an existing absolute directory")
    prefixes = config.get("approved_command_prefixes")
    if not isinstance(prefixes, list) or not prefixes:
        errors.append("approved_command_prefixes: supply one or more specific argv prefixes")
    else:
        for idx, prefix in enumerate(prefixes):
            if not isinstance(prefix, list) or len(prefix) < 2 or not all(isinstance(x, str) and x for x in prefix):
                errors.append(f"approved_command_prefixes[{idx}] must be an argv prefix of >=2 nonempty tokens")
            elif any(x in {"-c", "-e", "--eval"} for x in prefix[:3]):
                errors.append(f"approved_command_prefixes[{idx}]: interpreter eval is disallowed")
            elif len(prefix) == 2 and prefix[1] == "-m":
                errors.append(f"approved_command_prefixes[{idx}] is too broad; include the module name")
    edits = config.get("allowed_edit_globs")
    if not isinstance(edits, list) or not edits or not all(isinstance(x, str) and x.strip() and not Path(x).is_absolute() and ".." not in Path(x).parts for x in edits):
        errors.append("allowed_edit_globs must be nonempty, safe relative path globs without ..")
    protected = config.get("protected_paths")
    if not isinstance(protected, list) or not protected or not all(isinstance(x, str) and x.strip() and not Path(x).is_absolute() and ".." not in Path(x).parts for x in protected):
        errors.append("protected_paths must be a nonempty list of relative path globs without ..")
    budget = config.get("budget")
    if not isinstance(budget, dict):
        errors.append("budget must be an object")
    else:
        for field in ("max_runs", "per_run_timeout_minutes", "max_total_run_minutes", "disk_check_interval_seconds"):
            value = budget.get(field)
            if not finite_number(value) or value <= 0:
                errors.append(f"budget.{field} must be a positive number")
        if finite_number(budget.get("max_runs")) and int(budget["max_runs"]) != budget["max_runs"]:
            errors.append("budget.max_runs must be an integer")
        for field in ("min_free_disk_gb", "min_free_gpu_memory_mb"):
            if not finite_number(budget.get(field)) or budget[field] < 0:
                errors.append(f"budget.{field} must be a nonnegative number")
        if type(budget.get("require_gpu")) is not bool:
            errors.append("budget.require_gpu must be true/false")
        gpu_ids = budget.get("gpu_ids")
        if not isinstance(gpu_ids, list) or not all(isinstance(x, str) and x.isdecimal() for x in gpu_ids):
            errors.append("budget.gpu_ids must contain decimal index strings")
        if budget.get("require_gpu") and not gpu_ids:
            errors.append("budget.require_gpu needs at least one GPU index")
    gates = config.get("gates")
    if not isinstance(gates, list) or [g.get("id") if isinstance(g, dict) else None for g in gates] != GATE_IDS:
        errors.append("gates must contain exactly G0,G1,G2,G3 in this order")
    else:
        for gate in gates:
            crits = gate.get("criteria")
            if not isinstance(crits, list) or not crits:
                errors.append(f"{gate['id']}: preregister >=1 numerical/boolean criterion")
                continue
            for idx, crit in enumerate(crits):
                loc = f"{gate['id']}.criteria[{idx}]"
                if not isinstance(crit, dict) or not meaningful(crit.get("metric")):
                    errors.append(f"{loc}: metric must be a nonempty name")
                    continue
                op = crit.get("op")
                if op not in OPERATORS:
                    errors.append(f"{loc}: unsupported operator {op!r}")
                elif op not in {"is_true", "is_false"} and not finite_number(crit.get("value")):
                    errors.append(f"{loc}: numeric operator needs a finite numeric value")
    return errors


@contextlib.contextmanager
def exclusive_workspace(ws):
    lock = ws / ".researchctl.lock"
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as err:
        raise ResearchError(f"Workspace is locked: {lock}. Inspect owner/process before recovery; do not bypass.") from err
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()} timestamp={timestamp()}\n")
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()


def git_info(repo: Path):
    def run_git(*args):
        try:
            p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10)
            return (p.stdout.strip()[:4000] if p.returncode == 0 else "unknown")
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"
    return {"head": run_git("rev-parse", "HEAD"),
            "status_short": run_git("status", "--short", "-uno"),
            "diff_stat": run_git("diff", "--stat")}


def initialized_workspace(repo: Path, ws: Path):
    if not repo.is_dir():
        raise ResearchError(f"Repository directory does not exist: {repo}")
    ws.mkdir(parents=True, exist_ok=True)
    for subdir in ("runs", "assessments", "metrics"):
        (ws / subdir).mkdir(exist_ok=True)
    protocol = ws / "protocol.json"
    if not protocol.exists():
        value = load_json(ASSET_DIR / "protocol.template.json")
        value["repo_root"] = str(repo)
        value["project_id"] = repo.name
        write_json(protocol, value)
    for asset, output in (("NOVELTY_LEDGER.template.md", "NOVELTY_LEDGER.md"),
                          ("PROJECT_BRIEF.template.md", "PROJECT_BRIEF.md"),
                          ("HYPOTHESIS.template.md", "HYPOTHESIS_H1.md")):
        dst = ws / output
        if not dst.exists():
            shutil.copyfile(ASSET_DIR / asset, dst)
    print(f"Initialized without overwriting existing files: {ws}")
    print("Edit protocol.json, preregister G0-G3, approve command prefixes and edit scope, then 'validate'.")


def latest_assessments(ws: Path, protocol_sha=None):
    latest = {}
    for p in sorted((ws / "assessments").glob("*.json")):
        try:
            value = load_json(p)
        except ResearchError:
            continue
        if protocol_sha is not None and value.get("protocol_sha256") != protocol_sha:
            continue
        hyp, gate = value.get("hypothesis"), value.get("gate")
        if isinstance(hyp, str) and gate in GATE_IDS:
            latest[(hyp, gate)] = value
    return latest


def check_gate_order(ws: Path, hyp: str, gate: str, protocol_sha: str):
    if gate not in GATE_IDS:
        raise ResearchError(f"Unknown gate: {gate}")
    idx = GATE_IDS.index(gate)
    if idx:
        prior = GATE_IDS[idx - 1]
        previous = latest_assessments(ws, protocol_sha).get((hyp, prior))
        if not previous or previous.get("status") != "PASS":
            raise ResearchError(f"{gate} blocked for {hyp}: {prior} must PASS under identical protocol hash")


def total_used_minutes(ws: Path):
    metas = []
    for path in (ws / "runs").glob("*/run.json"):
        try:
            record = load_json(path)
            metas.append(record)
        except ResearchError:
            continue
    used_seconds = 0.0
    now = dt.datetime.now(dt.timezone.utc)
    for record in metas:
        duration = record.get("duration_seconds")
        if finite_number(duration):
            used_seconds += duration
        elif record.get("status") == "RUNNING":
            try:
                begin = dt.datetime.fromisoformat(record["started_at"])
                used_seconds += max(0, (now - begin).total_seconds())
            except (ValueError, TypeError, KeyError):
                used_seconds += 86400  # fail closed for corrupt active receipts
    return len(metas), used_seconds / 60


def gpu_preflight(budget):
    if not budget["require_gpu"] and budget["min_free_gpu_memory_mb"] <= 0:
        return
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=12)
    except (OSError, subprocess.TimeoutExpired) as err:
        raise ResearchError(f"GPU preflight failed: {err}") from err
    if result.returncode:
        raise ResearchError(f"GPU preflight failed: {result.stderr.strip()[:300]}")
    free = {}
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            try:
                free[parts[0]] = int(parts[1])
            except ValueError:
                pass
    for gpu_id in budget["gpu_ids"]:
        if gpu_id not in free:
            raise ResearchError(f"Requested GPU {gpu_id} not reported by nvidia-smi")
        if free[gpu_id] < budget["min_free_gpu_memory_mb"]:
            raise ResearchError(f"GPU {gpu_id} has {free[gpu_id]} MiB free; below preflight threshold")


def available_gb(path: Path):
    return shutil.disk_usage(path).free / (1024**3)


def approved_command(command, prefixes):
    return any(command[:len(prefix)] == prefix for prefix in prefixes)


def scrub_command(command):
    clean = []
    hiding = False
    for element in command:
        if hiding:
            clean.append("[REDACTED]")
            hiding = False
        elif re.search(r"(^|[-_])(secret|password|token|api_key|apikey)(=|$|[-_])", element, re.I):
            if "=" in element:
                clean.append(element.split("=", 1)[0] + "=[REDACTED]")
            else:
                clean.append(element)
                hiding = True
        else:
            clean.append(element)
    return clean


def terminate_process(p):
    if p.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(p.pid, signal.SIGTERM)
        else:
            p.terminate()
        p.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(p.pid, signal.SIGKILL)
            else:
                p.kill()
        except OSError:
            pass
        with contextlib.suppress(subprocess.TimeoutExpired):
            p.wait(timeout=5)


def do_run(args):
    ws = Path(args.workspace).resolve()
    with exclusive_workspace(ws):
        config, sha = protocol_at(ws)
        errors = validate_protocol(config)
        if errors:
            raise ResearchError("Invalid protocol:\n - " + "\n - ".join(errors))
        if not args.approved:
            raise ResearchError("Runs require --approved AND actual authorization from the repository owner")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.hypothesis):
            raise ResearchError("hypothesis ID must be 1-64 safe ASCII characters")
        check_gate_order(ws, args.hypothesis, args.gate, sha)
        latest = latest_assessments(ws, sha).get((args.hypothesis, args.gate))
        if latest and latest.get("status") == "FAIL" and not args.repair_reason:
            raise ResearchError("Current gate already FAIL; require --repair-reason or use a new hypothesis")
        command = list(args.command)
        if command and command[0] == "--":
            command.pop(0)
        if not command:
            raise ResearchError("Specify an argv command after '--'")
        if not approved_command(command, config["approved_command_prefixes"]):
            raise ResearchError("Executable argv does not match any approved_command_prefixes")
        budget = config["budget"]
        count, spent = total_used_minutes(ws)
        if count >= budget["max_runs"]:
            raise ResearchError(f"Run budget exhausted: {count} of {budget['max_runs']} launches")
        remaining = budget["max_total_run_minutes"] - spent
        if remaining <= 0:
            raise ResearchError("Aggregate execution time budget exhausted")
        if available_gb(ws) < budget["min_free_disk_gb"]:
            raise ResearchError("Not enough free disk space before launching")
        gpu_preflight(budget)
        timeout_s = max(0, min(remaining, budget["per_run_timeout_minutes"])) * 60
        repo = Path(config["repo_root"])
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + token()
        run_dir = ws / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        record = {
            "run_id": run_id, "hypothesis": args.hypothesis, "gate": args.gate,
            "label": args.label or "unnamed", "repair_reason": args.repair_reason or None,
            "started_at": timestamp(), "status": "RUNNING", "command": scrub_command(command),
            "protocol_sha256": sha, "git": git_info(repo), "budget_seconds_allowed": timeout_s,
            "gpu_ids": budget.get("gpu_ids", []),
        }
        receipt = run_dir / "run.json"
        write_json(receipt, record)
        env = os.environ.copy()
        if budget["gpu_ids"]:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(budget["gpu_ids"])
        started = time.monotonic()
        p = None
        status = "ERROR"
        exit_code = None
        failure_reason = None
        try:
            with (run_dir / "stdout.log").open("wb") as stdout, (run_dir / "stderr.log").open("wb") as stderr:
                p = subprocess.Popen(command, cwd=repo, env=env, stdout=stdout, stderr=stderr,
                                     start_new_session=(os.name == "posix"))
                record["pid"] = p.pid
                write_json(receipt, record)
                while True:
                    exit_code = p.poll()
                    if exit_code is not None:
                        status = "SUCCEEDED" if exit_code == 0 else "FAILED"
                        break
                    if time.monotonic() - started >= timeout_s:
                        status, failure_reason = "STOP_TIMEOUT", "Per-run or aggregate wall-time budget reached"
                        terminate_process(p)
                        break
                    if available_gb(ws) < budget["min_free_disk_gb"]:
                        status, failure_reason = "STOP_DISK", "Available disk fell below threshold"
                        terminate_process(p)
                        break
                    time.sleep(min(1.0, budget["disk_check_interval_seconds"]))
        except KeyboardInterrupt:
            status, failure_reason = "INTERRUPTED", "KeyboardInterrupt"
            if p:
                terminate_process(p)
        except (OSError, ValueError) as err:
            status, failure_reason = "ERROR", str(err)
            if p:
                terminate_process(p)
        finally:
            record.update({"ended_at": timestamp(), "duration_seconds": round(time.monotonic() - started, 3),
                           "status": status, "exit_code": exit_code,
                           "error": failure_reason, "git_after": git_info(repo)})
            try:
                after_sha = canonical_digest((ws / "protocol.json").read_bytes())
            except OSError:
                after_sha = "MISSING"
            record["protocol_sha256_after"] = after_sha
            if after_sha != sha:
                record["status"] = "PROTOCOL_CHANGED"
                record["error"] = "Protocol changed during run; inspect and reapprove before using evidence"
            write_json(receipt, record)
        print(json.dumps({"run_id": run_id, "status": record["status"], "receipt": str(receipt),
                          "stdout": str(run_dir / "stdout.log"), "stderr": str(run_dir / "stderr.log")}, ensure_ascii=False))
        return 0 if record["status"] == "SUCCEEDED" else 2


def get_criterion_result(criterion, metrics):
    field, operator = criterion["metric"], criterion["op"]
    if field not in metrics:
        return "INCONCLUSIVE", None, "Metric missing"
    value = metrics[field]
    if operator in {"is_true", "is_false"}:
        if type(value) is not bool:
            return "INCONCLUSIVE", value, "Expected boolean"
        passed = value if operator == "is_true" else not value
    else:
        if not finite_number(value):
            return "INCONCLUSIVE", value, "Expected finite numeric value"
        target = criterion["value"]
        passed = {">=": lambda: value >= target, ">": lambda: value > target,
                  "<=": lambda: value <= target, "<": lambda: value < target,
                  "==": lambda: value == target}[operator]()
    return ("PASS" if passed else "FAIL"), value, ""


def file_integrity(path: Path):
    if not path.is_file():
        raise ResearchError(f"Evidence/metrics file does not exist: {path}")
    size = path.stat().st_size
    result = {"path": str(path.resolve()), "size_bytes": size}
    if size <= 64 * 1024 * 1024:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        result["sha256"] = h.hexdigest()
    else:
        result["sha256"] = None
        result["note"] = "Not hashed: evidence >64MiB"
    return result


def do_assess(args):
    ws = Path(args.workspace).resolve()
    with exclusive_workspace(ws):
        config, sha = protocol_at(ws)
        problems = validate_protocol(config)
        if problems:
            raise ResearchError("Invalid protocol:\n - " + "\n - ".join(problems))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.hypothesis):
            raise ResearchError("hypothesis ID must be safe ASCII")
        check_gate_order(ws, args.hypothesis, args.gate, sha)
        metric_path = Path(args.metrics).resolve()
        metric_info = file_integrity(metric_path)
        if metric_info["size_bytes"] > 2 * 1024 * 1024:
            raise ResearchError("Metrics JSON too large (>2MiB); use raw outputs as separate evidence")
        metrics = load_json(metric_path)
        if not isinstance(metrics, dict):
            raise ResearchError("Metrics file must contain an object of named scalar values")
        if not args.evidence:
            raise ResearchError("Provide --evidence path to original metric/run evidence")
        evidence = [file_integrity(Path(item).resolve()) for item in args.evidence]
        gate = config["gates"][GATE_IDS.index(args.gate)]
        checks = []
        for c in gate["criteria"]:
            result, observed, reason = get_criterion_result(c, metrics)
            checks.append({"metric": c["metric"], "op": c["op"], "threshold": c.get("value"),
                           "observed": observed, "result": result, "reason": reason})
        if any(c["result"] == "FAIL" for c in checks):
            status = "FAIL"
        elif any(c["result"] == "INCONCLUSIVE" for c in checks):
            status = "INCONCLUSIVE"
        else:
            status = "PASS"
        record = {"assessment_id": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + token(),
                  "created_at": timestamp(), "hypothesis": args.hypothesis, "gate": args.gate,
                  "status": status, "note": args.notes or "", "checks": checks,
                  "metrics_file": metric_info, "evidence": evidence, "protocol_sha256": sha,
                  "git": git_info(Path(config["repo_root"])),
                  "caveat": "Numerical preregistered checks only; independent scientific/causal review required"}
        write_json(ws / "assessments" / f"{record['assessment_id']}.json", record)
        print(json.dumps({"status": status, "gate": args.gate, "hypothesis": args.hypothesis,
                          "checks": checks, "scientific_review": "REQUIRED"}, ensure_ascii=False))
        return 0 if status == "PASS" else (3 if status == "INCONCLUSIVE" else 2)


def escaped(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def do_report(args):
    ws = Path(args.workspace).resolve()
    config, sha = protocol_at(ws)
    runs = []
    for path in sorted((ws / "runs").glob("*/run.json")):
        try:
            runs.append(load_json(path))
        except ResearchError:
            pass
    assessments = []
    for path in sorted((ws / "assessments").glob("*.json")):
        try:
            assessments.append(load_json(path))
        except ResearchError:
            pass
    active = latest_assessments(ws, sha)
    count, used = total_used_minutes(ws)
    output = Path(args.output).resolve() if args.output else ws / "EVIDENCE_PACK.md"
    budget = config.get("budget", {})
    lines = ["# AutoResearch Evidence Pack", "", f"Generated UTC: {timestamp()}",
             f"Project: {config.get('project_id','UNKNOWN')}",
             f"Target venue: {config.get('target_venue','UNKNOWN')}",
             f"Question: {config.get('research_question','UNKNOWN')}",
             f"Primary metric: {config.get('primary_metric','UNKNOWN')}",
             f"Protocol version: {config.get('protocol_version','UNKNOWN')}",
             f"Current protocol SHA256: `{sha}`", "",
             "**Important:** PASS below means mechanical threshold PASS, not causal proof, independent replication, publication readiness, or venue acceptance.",
             "", "## Budget", "", f"Recorded launches: {count} / {budget.get('max_runs','?')}",
             f"Aggregate recorded execution minutes: {used:.2f} / {budget.get('max_total_run_minutes','?')}",
             "", "## Run ledger", "", "| ID | Hypothesis | Gate | Label | Status | Duration (min) | Commit |", "|---|---|---|---|---|---:|---|"]
    for r in runs:
        mins = r.get("duration_seconds", 0) / 60 if finite_number(r.get("duration_seconds")) else 0
        lines.append("| " + " | ".join(escaped(x) for x in [r.get("run_id"), r.get("hypothesis"), r.get("gate"),
                    r.get("label"), r.get("status"), f"{mins:.2f}", str(r.get("git", {}).get("head", "?"))[:12]]) + " |")
    lines.extend(["", "## Gate history (includes failed and exploratory trials)", "",
                  "| Assessment | Hypothesis | Gate | Mechanical result | Protocol SHA |", "|---|---|---|---|---|"])
    for a in assessments:
        lines.append("| " + " | ".join(escaped(x) for x in [a.get("assessment_id"), a.get("hypothesis"),
                     a.get("gate"), a.get("status"), str(a.get("protocol_sha256", ""))[:12]]) + " |")
    lines.extend(["", "## Latest gates for current protocol", "",
                  "| Hypothesis | Gate | Status | Criterion observations |", "|---|---|---|---|"])
    for (hyp, gate), a in sorted(active.items()):
        check_text = "; ".join(f"{c.get('metric')}={c.get('observed')} ({c.get('result')})" for c in a.get("checks", []))
        lines.append("| " + " | ".join(escaped(x) for x in [hyp, gate, a["status"], check_text]) + " |")
    lines.extend(["", "## Research interpretation (human verification required)", "",
                  "- Claim status: NOT VERIFIED BY THIS SCRIPT; audit evidence against literature and strongest simple alternative.",
                  "- Determine independent sampling unit, sample sizes, CIs, paired seeds, negative controls and heldout split integrity.",
                  "- Document source versions, full commands, model releases, preprocessing, runtime and resource use.",
                  "- Distinguish phenomenon / mechanism / causal evidence / transfer; preserve failures and conflicting evidence.",
                  "- Recheck current official venue rules before submission.", "",
                  "## Linked artifacts", "", "- `protocol.json`: preregistration; history is keyed by protocol SHA.",
                  "- `runs/<id>/run.json`: command receipt, Git head, Git status, stdout/stderr paths.",
                  "- `assessments/*.json`: threshold checks, metric file hash and evidence hashes.",
                  "- `NOVELTY_LEDGER.md`: research overlap and unresolved originality checks.",
                  "- `PROJECT_BRIEF.md` and `HYPOTHESIS_H1.md`: project and hypothesis provenance.", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Evidence Pack: {output}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("init", help="Create templates without overwriting existing files")
    p.add_argument("--repo", required=True, help="Existing repository path")
    p.add_argument("--workspace", required=True)
    p = sub.add_parser("validate", help="Validate protocol for executing real experiments")
    p.add_argument("--workspace", required=True)
    p = sub.add_parser("run", help="Launch an allowlisted experiment, bounded and logged")
    p.add_argument("--workspace", required=True)
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--gate", choices=GATE_IDS, required=True)
    p.add_argument("--label", default="")
    p.add_argument("--repair-reason", default="")
    p.add_argument("--approved", action="store_true", help="Acknowledge actual owner authorization; not proof")
    p.add_argument("command", nargs=argparse.REMAINDER, help="argv after --")
    p = sub.add_parser("assess", help="Record gate result from metrics JSON and evidence")
    p.add_argument("--workspace", required=True)
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--gate", choices=GATE_IDS, required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--evidence", action="append", required=True)
    p.add_argument("--notes", default="")
    p = sub.add_parser("report", help="Create a Markdown Evidence Pack")
    p.add_argument("--workspace", required=True)
    p.add_argument("--output")
    args = parser.parse_args()
    try:
        if args.action == "init":
            initialized_workspace(Path(args.repo).resolve(), Path(args.workspace).resolve())
            return 0
        if args.action == "validate":
            config, sha = protocol_at(Path(args.workspace).resolve())
            errors = validate_protocol(config)
            if errors:
                raise ResearchError("Invalid protocol:\n - " + "\n - ".join(errors))
            print(f"VALID protocol SHA256: {sha}")
            return 0
        if args.action == "run":
            return do_run(args)
        if args.action == "assess":
            return do_assess(args)
        if args.action == "report":
            return do_report(args)
        raise ResearchError("Unsupported action")
    except ResearchError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
