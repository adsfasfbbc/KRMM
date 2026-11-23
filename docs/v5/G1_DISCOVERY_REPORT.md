# G1 Report: Equal-Budget Blind Damage Discovery on Brand-New Entities

- **Artifact:** `results/v5-new/g1_discovery_results.json`
- **Model:** `/root/rivermind-data/models/Qwen2.5-Omni-3B`
- **Runtime:** `575.6s`
- **Pool (`N = 45` evaluated):** `29/36` cleanly mastered brand-new entities + `12` originally unknown controls + `4` ambiguous controls.
- **Post-Injection Behavioral Truth:** `17 damaged_mastered`, `3 collateral_damaged` (uninjected mastered entities degraded by D1 injection), `9 healthy_mastered`, `12 originally_unknown`, `4 ambiguous_or_unverifiable`.

## 1. Equal-Budget Comparison at `5N` (`B = 225` Queries)
| Detector | Queries Used | Binary AUROC | Binary AUPRC | P@B / R@B | Healthy FPR | Unknown FPR | Collateral Recall | E2E Coverage | 3-Class Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct_confidence` | 45 | 0.166 | 0.305 | 0.250 | 0.111 | 0.667 | 0.000 | 0.000 | 0.178 |
| `entropy` | 45 | 0.148 | 0.300 | 0.200 | 0.222 | 0.917 | 0.667 | 0.118 | 0.244 |
| `self_consistency` | 90 | 0.435 | 0.423 | 0.550 | 0.111 | 0.417 | 0.333 | 0.118 | 0.244 |
| `direct_paraphrases` | 135 | 0.300 | 0.408 | 0.300 | 0.556 | 0.667 | 0.333 | 0.118 | 0.156 |
| `constraint_only` | 140 | 0.345 | 0.427 | 0.450 | 0.222 | 0.417 | 0.000 | 0.059 | 0.178 |
| **`m2_reference_free`** | 185 | **0.789** | **0.732** | **0.800** | 0.222 | **0.167** | 0.333 | **0.882** | **0.667** |
| **`full_m2`** | 220 | **0.860** | **0.762** | **0.900** | 0.333 | 0.583 | **1.000 (3/3)** | **0.941** | 0.644 |

## 2. Key Findings & Honest Trade-Offs
1. **Why Confidence/Entropy Collapse (`AUROC < 0.20`):** Wrong-fact association (`D1`) drives the model to become *highly confident* in the injected wrong answer; thus low-confidence/high-entropy detectors rank unknown controls as "damaged" (`unknown_fpr = 0.667–0.917`) and miss confident corrupted facts.
2. **Constraint + Access-Path Discrepancy Succeeds on Unseen Entities:** Both `m2_reference_free` (`AUROC 0.789`) and `full_m2` (`AUROC 0.860`) dramatically outperform all 5 baselines under strict budget tracking, and `full_m2` detects all `3/3` uninjected collateral damages.
3. **Unknown-Control FPR Trade-Off:** `m2_reference_free` achieves much lower `unknown_fpr` (`0.167` vs `0.583`) and slightly higher 3-class accuracy (`0.667` vs `0.644`), whereas `full_m2` maximizes binary AUROC (`0.860`) and collateral recall (`1.000`). Downstream gating (G2) is therefore essential to prevent false positives on unknown entities from turning into wrong parameter writes.
