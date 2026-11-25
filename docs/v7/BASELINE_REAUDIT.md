# KRMM v0.7 P0 Baseline Re-Audit (`BASELINE_REAUDIT.md`)

- **Verified Base Commit:** `bdb628a27feb5a41f8fa1676e5f9eeef9ed81f2b` (`codex/krmm-v06-preprint-icml`)
- **Verified Frozen v0.5 Parent:** `7a41d02a0b7cfa226d8ef7bc47e13d5402d44c1f` (`v5_frozen_parent_verified = True`)
- **Local Backup Bundle:** `/root/rivermind-data/KRMM_v06_backup_bdb628a.bundle` (`308 MB`)
- **Audit Artifact:** `reports/v7/integrity.json` (`scripts/v7/audit_v6.py --base bdb628a --out reports/v7`)

## 1. Historical Integrity & Protection Check
- All protected v0.1–v0.6 code, tests, data manifests, configs, reports, and `paper/v6/` remain untouched (read-only).
- `pilot_manifest.json` SHA256 verified; `D = 19` (`Qwen2.5-Omni-3B`) and `D = 10` (`Llama-3.2-3B-Instruct`) verified directly from v0.6 JSON artifacts.
- All 100 historical regression tests (`7` v6 + `6` v5 + `23` v4 + `46` v3_cpu + `13` v3 + `5` contracts) + `6` new v0.7 unit tests = **`106/106` PASSED**.

## 2. Exact Reconciliation of v0.6 Failure Counts
- `qwen_Oracle_upper_bound` (`8` writes): `3` same-entity regressions (`c2w=3`, masked by `w2c=11` in group mean accuracy `0.819 -> 0.931`), `token_agreement = 0.806`, `general_commonsense drop = 0.000`.
- `qwen_M_evidence_gated` (`8` writes): `11` same-entity regressions (`4` on written entities, `7` on unwritten sibling entities).
- `qwen_M_full` (`8` writes): `10` same-entity regressions (`3` on written entities, `7` on unwritten sibling entities).
- `qwen_B1_generative` (`8` writes): `16` same-entity regressions (`4` on written entities, `12` on unwritten sibling entities).
- `llama_Oracle_upper_bound` & `llama_M_evidence_gated` (`8` writes): **`0` regressions on written entities**, **`1` regression on unwritten entity** (`erhu`: `"How does a musician play the Chinese erhu?"`).
