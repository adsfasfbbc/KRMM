# KRMM v0.7 Master Evidence Pack (`EVIDENCE_PACK.md`)

- **Branch:** `codex/krmm-v07-continual-safe-repair` (branched from `bdb628a27feb5a41f8fa1676e5f9eeef9ed81f2b`, v0.6 bundle backed up locally at `/root/rivermind-data/KRMM_v06_backup_bdb628a.bundle`)
- **Hardware:** 1× NVIDIA GeForce RTX 3090 (24 GB VRAM)
- **Batch C1 GPU Budget Used:** `~0.90h` (`3220.5s` = `753.6s` `OW-replay` + `1344.3s` `OW-writer-matrix` + `1122.6s` `OW-dev-v7`) across `3` top-level GPU launches (`+ 13` isolated fresh-subprocess reload checks), well inside `12h / 24 launches` ceiling; `/root/rivermind-data` free disk `>= 20 GiB` (`>= 15 GiB` limit).
- **Test Suite:** `106/106` PASS (`6` v7 + `7` v6 + `6` v5 + `23` v4 + `46` v3_cpu + `13` v3 + `5` contracts); `0` modifications to protected v0.1–v0.6 paths.

## Exact Section 14 Status Block
```text
baseline_integrity: PASS
diagnosis_G0: PASS
OW_PILOT: PASS (replay: W1 Y=0.889, W4/T4/W5 Y=0.778 vs W0 Y=0.000)
OW_CONFIRM: FAIL (out-of-sample DEV-V7: W0/W1/W2/W4 all restore 15/15 yet incur 1-3 same-entity regressions -> Y_fresh=0.000)
TX_PUBLIC_MONITOR: PASS_PILOT_ONLY (T4 rolled back 2/8 steps on replay with Y=0.778, lower norm 12.75; full blind C2 NOT_RUN)
UK_REAL_UNKNOWN: NOT_RUN (Batch C2)
E2E_CONTINUAL: NOT_RUN (Batch C2 gated behind OW_CONFIRM)
TRANSFER_OPEN_RELATION: NOT_RUN (Batch C3)
TRANSFER_D3: NOT_RUN (Batch C3)
novelty_check: VERIFIED_SCOPE
arxiv: HUMAN_REVIEW_READY (Diagnostic/Feasibility edition: paper/v7/main.tex)
icml_decision: PIVOT (human decision required — pivot to diagnostic/limits paper or authorize hard-constraint null-space/parameter-isolated continual writer)
remote_push: NOT_AUTHORIZED (local commit only; v0.6 bundle backed up locally)
```
