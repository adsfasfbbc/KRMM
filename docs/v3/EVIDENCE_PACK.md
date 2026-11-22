# KRMM v0.3 Evidence Pack & Verification Manifest

## 1. Environment & Model Integrity
- **Base Repository / Commit:** `https://github.com/adsfasfbbc/KRMM.git`, branch `codex/krmm-v03` (branched from `origin/codex/krmm-v02` at `bbf4ab0`).
- **Base Model:** `Qwen/Qwen2.5-Omni-3B` (revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`), stored at `/root/rivermind-data/models/Qwen2.5-Omni-3B`; all 3 `.safetensors` shards matched official SHA256 digests in `docs/v2/model-integrity.json`.
- **Key Software Stack:** `transformers==4.57.6`, `peft==0.15.2`, `accelerate==1.6.0`, `huggingface_hub==0.36.2`, `scikit-learn`, `scipy`, `pytest==9.2.0`, Python `3.12.13` on NVIDIA RTX 3090 (24 GiB VRAM). Peak run VRAM: `11.16 GiB`.

## 2. Boundary & Contract Unit Tests
- `python3 -m pytest tests/v3/ -v`: **13 / 13 PASSED** (`12.11s`)
  - `test_label_isolation`, `test_no_inverse_shortcut`, `test_mastery_filter`, `test_split_integrity`, `test_detector_blindness`, `test_evidence_provenance`, `test_abstention`, `test_independent_locality`, `test_counterfactual_attempts`, `test_fresh_process_reload`, `test_rollback`, `test_budget_guard`, `test_result_integrity`.
- `python3 -m unittest tests/test_contracts.py`: **5 / 5 PASSED** (`0.011s`, backward compatibility with v0.2 contract checks).

## 3. Core Code & Config Inventory (`krmm/v3/`, `configs/v3/`, `scripts/v3/`)
- **Core Modules (`krmm/v3/`):** `schemas.py`, `knowledge_registry.py`, `mastery.py`, `damage.py`, `damage_audit.py`, `knowledge_graph.py`, `evidence.py`, `reconstruction.py`, `discovery.py`, `repairability.py`, `repair_policy.py`, `parameter_repair.py`, `memory_recovery.py`, `verify_fresh.py`, `evaluator.py`, `metrics.py`, `runner.py`, `report.py`, `pipeline_p1_p3.py`.
- **Frozen Configs (`configs/v3/`):**
  - `p0_baseline_d1.json`, `p0_baseline_d2.json`, `p0_d1_steps64.json`, `p0_d2_steps64.json`, `p2_dev12_orth_e6.json`, `p2_dev36_full.json`, `p2_cal36_full.json`, `p2_test48_frozen.json`.
- **Entrypoint Scripts (`scripts/v3/`):** `run_mastery_survey.py`, `run_p1_p3.py`, `analyze_e0_e7.py`, `export_manifest.py`.

## 4. Artifact Checksums (SHA256)
| Artifact | SHA256 |
|---|---|
| `docs/v3/DATA_MANIFEST.jsonl` | `b4a1fc522492868c7360e921b7e02e787f94b91821c9bc7c3768e8b47e382b4d` |
| `results/v3-new/e0_e7_full_final.json` | `0a4b6c596747e0a540e2573a7fe209d7d9b9100d1a66883d1636fb97b19fbab3` |
| `results/v3-new/exp05c-m2-discovery-dev36/m2_discovery_metrics.json` | `b5e7c31c2628fc4932b3d0e103a74d067bf7a8f7d5cbc784f6442000ac554aff` |
| `results/v3-new/exp10-m2-discovery-cal36/m2_discovery_metrics.json` | `39313c97f92dacdb1caac3ebc9b425ab2d0e460ee742a51034632e89c8ea21f8` |
| `results/v3-new/exp11-m2-discovery-test48/m2_discovery_metrics.json` | `b23fd7955b8fb1952c54ef65699a95abf56b6c7c73340f2a6e8d38311a626a0a` |
| `results/v3-new/exp06-e5-e6-dev12/metrics.json` | `b8132faea0281e65d594181c993dd9e8420a28907ccdc577fcc7dc2ea2a0916a` |
| `results/v3-new/exp07-p2-dev36/metrics.json` | `843da19f1bac6f9e41a603b33b2c80b9305e0c2240955e133c46455a558792e7` |
| `results/v3-new/exp08-p2-cal36/metrics.json` | `52c9e991dcdc70e05776cfac21e4dd255a0e044d557ef216d947459129acc00d` |
| `results/v3-new/exp09-p2-test48/metrics.json` | `b721f1b1296c0aee971b49ade67fcd764b2772e63c61048adaa09f76cf55746a` |
| `results/v3-new/exp09-p2-test48/damage-randomization-audit.json` | `3ba897a74d6b70f7060ff45e9698599a1417e2290941389bc4b4e3a02ba18c50` |

## 5. Experiment Run Ledger (`results/v3-new/`)
1. `exp01-mastery-survey-dev` — Clean mastery + unknown control survey (`dev=36`, `unknown_control=8`)
2. `exp02-p0-d1-dev12` — Initial `D1` `damage_steps=24` pilot (`0/12` effective -> escalated steps)
3. `exp03-p0-d1-s64-dev12` — `D1` `damage_steps=64` Gate G0 pass (`10/10` oracle strict)
4. `exp04-p0-d2-s64-dev12` — `D2_response_suppression` pilot (`5/12` effective, `3/12` strict-eligible, `3/3` oracle)
5. `exp05a-m3-probe-check-dev` — Refined `knowledge_graph.py` / `evidence.py` survey (`63.9% -> 77.8%`)
6. `exp05b-m2-discovery-dev36` / `exp05c-m2-discovery-dev36` — M2 Discovery on `dev` (`AUROC 0.8125`, `Unknown FPR 0.000`)
7. `exp06-e5-e6-dev12` — E5 (`4/9`) vs E6 `orthogonal_retain_proj` (`9/9`) ablation
8. `exp07-p2-dev36` — Full `dev` (`33/36` mastered, `21` strict-eligible, `structured 14/21` vs `oracle 21/21`)
9. `exp08-p2-cal36` — Full `calibration` (`32/36` mastered, `16` strict-eligible, `structured 12/16` vs `generative 8/16`, `+25.0 pp [+6.25, +50.0]`)
10. `exp09-p2-test48` — Sealed `test48` (`43/48` mastered, `29` strict-eligible, `structured 20/29` vs `generative 13/29`, `+24.14 pp [+6.90, +41.38]`, `E7 AUROC 0.9327 / 0.8125`)
11. `exp10-m2-discovery-cal36` — Out-of-sample M2 Discovery on `calibration` (`AUROC 0.9902`, `Unknown FPR 0.000`)
12. `exp11-m2-discovery-test48` — Confirmatory M2 Discovery on `test48` (`AUROC 0.9177`, `Unknown FPR 0.000`)
13. `e0_e7_full_final.json` — Consolidated E0–E7 statistics and cross-split / confirmatory E7 comparisons.

## 6. Clean Separation & Repository Hygiene Checks
- No proxy credentials or `/opt/proxy/*` files exist inside the repository tree.
- No large binary checkpoints (`.safetensors`, `.pt`, adapter weights) are committed in git; heavy per-unit adapter directories under `results/v3-new/*/*/` are excluded via `.gitignore` while compact metric/audit JSONs and documentation are preserved.
- `git` repository bundle archived at `docs/v3/source-history.bundle` alongside commits on branch `codex/krmm-v03`.
