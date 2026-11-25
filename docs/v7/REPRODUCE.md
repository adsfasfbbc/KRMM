# KRMM v0.7 Reproducibility Guide (`REPRODUCE.md`)

```bash
# 1. P0 Read-Only Forensics on v0.6 (bdb628a)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python scripts/v7/audit_v6.py --base bdb628a --out reports/v7

# 2. Full 106-Test Regression Suite (v7 + v6 + v5 + v4 + v3_cpu + v3 + contracts)
CUDA_VISIBLE_DEVICES="" python -m pytest tests/v7/ tests/v6/ tests/v5/ tests/v4/ tests/v3/ -q
CUDA_VISIBLE_DEVICES="" python -m pytest tests/v3_cpu/ -q
CUDA_VISIBLE_DEVICES="" python -m unittest tests/test_contracts.py

# 3. G0-Diagnosis (D0-D4 + Isolated Fresh Subprocess Reload)
PYTHONPATH=. python scripts/v7/run_oracle_diagnosis.py --config configs/v6/pilot.json --out results/v7-new/OW-replay

# 4. P1 Matched Oracle Writer Matrix (W0-W5 + T4 + 7 Fresh Subprocess Reloads)
PYTHONPATH=. python scripts/v7/run_writer_matrix.py --config configs/v6/pilot.json --diag-dir results/v7-new/OW-replay --out results/v7-new/OW-writer-matrix

# 5. Out-of-Sample DEV-V7 Confirmation (36 brand-new entities, 0 overlap with 281 historical names/aliases)
PYTHONPATH=. python scripts/v7/run_dev_writer_check.py --config configs/v6/pilot.json --out results/v7-new/OW-dev-v7 --seed 3407
```
