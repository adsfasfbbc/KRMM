# G2-B & G2-C Report: Paired Selective Repair & 5-Gate Locality Graph

- **Artifact:** `results/v5-new/g2_repair_results.json`
- **Model:** `/root/rivermind-data/models/Qwen2.5-Omni-3B`
- **Runtime:** `1350.5s` (`22.5 min`)
- **Funnel:** `36` brand-new known entities -> `29` cleanly mastered -> `22` effectively D1-damaged -> **`13` strict-eligible** (passing independent damage locality audit).
- **Writeback Method:** Fixed LoRA `standard_kl` (`rank=8`, `last_layers=4`, `repair_steps=32`, `retain_weight=50.0`) + serialized PEFT disk reload verification from clean `damaged_state`.

## 1. Paired 5-Strategy Comparison (`n = 13` Strict-Eligible Targets)
| Strategy | Attempted | Strict Restored | Eligible Rate | Attempted Precision | Wrong Writes | Abstention Rate | Same-Entity Preservation | Expected Ripple Consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct` | 13/13 | 0/13 | 0.0% | 0.0% | **13 (100.0%)** | 0.0% | 100.0% | 84.6% |
| `generative` | 13/13 | 8/13 | 61.5% | 61.5% | **5 (38.5%)** | 0.0% | 100.0% | 76.9% |
| **`structured`** | 12/13 | **11/13** | **84.6%** | 91.7% | 1 (7.7%) | 7.7% | 100.0% | 83.3% |
| **`evidence_gated`** | 9/13 | **9/13** | **69.2%** | **100.0% (9/9)** | **0 (0.0%)** | 30.8% | 100.0% | 77.8% |
| `oracle` (upper bound) | 13/13 | 13/13 | 100.0% | 100.0% | 0 (0.0%) | 0.0% | 100.0% | 84.6% |

## 2. Paired Cluster Bootstrap 95% CIs (`source_group` clusters, `n=13`, `seed=107`)
- `evidence_gated vs direct`: **`+69.2 pp`** (`95% CI [+46.2, +92.3] pp`) — strictly positive.
- `structured vs generative`: **`+23.1 pp`** (`95% CI [-7.7, +53.8] pp`) — closely matches v0.3's `+24.1 pp` on `test48`, though wider CI at `n=13`.
- `evidence_gated vs generative`: **`+7.7 pp`** (`95% CI [-15.4, +30.8] pp`) — reflects the explicit **safety-vs-coverage trade-off**: `evidence_gated` eliminates **all** wrong writes (`0/13` vs `5/13` for `generative` and `1/13` for `structured`), abstaining on `4` lower-support targets (`electric eel`, `cast iron skillet`, `granite mortar`, `quartz countertop`), of which `structured` correctly restored `2` and wrongly wrote `1`.

## 3. G2-C Locality Graph & Reload Verification
- Every passing adapter across all strategies passed serialized PEFT disk reload verification from `damaged_state`.
- `same_entity_relation_preservation` = **`100.0%`** across all attempted repairs; `expected_ripple_consistency` = **`76.9%–84.6%`**, cleanly separating expected logical ripple from collateral damage.
