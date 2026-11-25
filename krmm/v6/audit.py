"""Fresh-subprocess cumulative adapter reload audit (Section 3.1 F & 5.4)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def run_fresh_subprocess_audit(
    job_path: Path,
    python_exe: str = sys.executable,
    timeout_seconds: int = 900,
) -> Dict[str, object]:
    """Launch a brand-new isolated Python subprocess to reload base + damaged_state + cumulative adapters."""
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "v6" / "verify_fresh_subprocess.py"
    cmd = [python_exe, str(script_path), "--job", str(job_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    if proc.returncode != 0:
        return {
            "fresh_subprocess_ok": False,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-1500:],
            "stderr_tail": proc.stderr[-1500:],
        }
    job = json.loads(job_path.read_text(encoding="utf-8"))
    out_p = Path(job["output"])
    if not out_p.exists():
        return {"fresh_subprocess_ok": False, "error": f"Missing output file {out_p}"}
    res = json.loads(out_p.read_text(encoding="utf-8"))
    res["fresh_subprocess_ok"] = True
    return res
