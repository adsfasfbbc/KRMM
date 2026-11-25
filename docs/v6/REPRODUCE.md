# KRMM v0.6 Reproducibility Guide (`REPRODUCE.md`)

- **Branch:** `codex/krmm-v06-preprint-icml` (branched from `7a41d02`)
- **Python Environment:** `/root/rivermind-data/krmm-env/bin/python` (`torch 2.12.1+cu130`, `transformers 4.57.6`, `peft 0.15.2`, `pytest 9.2.0`)
- **Hardware:** Single NVIDIA GeForce RTX 3090 (24 GB VRAM)

## Exact Command Allowlist (Verified Executed)
```bash
# 1. P0 Read-Only Audit of Historical v0.5 Artifacts
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python scripts/v6/audit_v5.py \
  --baseline-ref 7a41d02 --out reports/v6/legacy_recompute.json

# 2. Build & Freeze Brand-New 50-Entity v0.6 Pool (0 Overlap with 192 Historical Entities)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python scripts/v6/build_pool.py \
  --config configs/v6/pilot.json --out data/v6/pilot_manifest.json

# 3. Full Regression & Isolation Test Suite (100/100 PASS including real subprocess smoke)
CUDA_VISIBLE_DEVICES="" python -m pytest tests/v6 tests/v5 tests/v4 tests/v3_cpu tests/v3
CUDA_VISIBLE_DEVICES="" python -m unittest tests/test_contracts.py

# 4. Qwen2.5-Omni-3B Multi-Target Discovery (G1) + True Sequential E2E + Fresh Subprocess Audit
PYTHONPATH=. python scripts/v6/run_discovery.py --config configs/v6/pilot.json --run-id qwen_pilot_g1_r2
PYTHONPATH=. python scripts/v6/run_episode.py --config configs/v6/pilot.json --discovery-run-id qwen_pilot_g1_r2 --run-id qwen_pilot_e2e_r2
PYTHONPATH=. python scripts/v6/run_selectivity.py --config configs/v6/pilot.json --discovery-run-id qwen_pilot_g1_r2 --run-id qwen_pilot_g2a_r3

# 5. Llama-3.2-3B-Instruct Cross-Architecture True Sequential E2E + Fresh Subprocess Audit
PYTHONPATH=. python scripts/v6/run_discovery.py --config configs/v6/transfer_llama32.json --run-id llama32_pilot_g1
PYTHONPATH=. python scripts/v6/run_episode.py --config configs/v6/transfer_llama32.json --discovery-run-id llama32_pilot_g1 --run-id llama32_pilot_e2e --strategies B0_no_repair,B1_generative,B2_ungated_structured,M_evidence_gated,M_full,Oracle_upper_bound

# 6. Programmatic Generation of LaTeX Tables & Matplotlib Figures
PYTHONPATH=. python scripts/v6/generate_paper_assets.py
```
