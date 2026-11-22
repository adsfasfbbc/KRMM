# KRMM v0.3 Reproduction Guide

1. Verify environment and tests:
```bash
pytest tests/v3/ -v
python3 -m unittest tests/test_contracts.py
```

2. Verify clean model weights (`Qwen/Qwen2.5-Omni-3B`, revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`):
```bash
python3 -u scripts/v2/download_model.py
```

3. Run P0 / G0 dev pilot (D1 wrong association):
```bash
python3 -u -m krmm.v3.runner --config configs/v3/p0_baseline_d1.json --out results/v3-new/p0-d1-dev --split dev --limit 12
```

4. Run P0 / G0 dev pilot (D2 suppression):
```bash
python3 -u -m krmm.v3.runner --config configs/v3/p0_baseline_d2.json --out results/v3-new/p0-d2-dev --split dev --limit 12
```
