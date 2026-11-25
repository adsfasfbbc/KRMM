# KRMM v0.6 Cross-Architecture Persistence Report (`GENERALIZATION_REPORT.md`)

- **Model:** `Llama-3.2-3B-Instruct` (Llama architecture, 3.2B, distinct from `Qwen2.5-Omni-3B`)
- **Discovery Artifact:** `results/v6-new/llama32_pilot_g1/metrics.json` (`378.7s`)
- **True E2E + Fresh-Subprocess Artifact:** `results/v6-new/llama32_pilot_e2e/metrics.json` (`918.5s`)
- **Pool:** Same frozen 50-entity v0.6 pilot pool (`pool_sha256 = 95b845901b2f`).

## 1. G0-SCIENCE Gate & Funnel (`Llama-3.2-3B-Instruct`)
- Clean mastery (screen + held-out = 1.0): `23/36 (63.9%)` -> **below `75%` G0 mastery threshold (`g0_science_pass = False`, honestly preserved)**, leaving `22` unknown/unmastered/unverifiable units in the 50-entity pool!
- Paired degradation denominator: **`D = 10`** (`8` effective planned D1 + `2` collateral damaged), `18` healthy.

## 2. True Sequential E2E Results with Isolated Fresh-Subprocess Reload (`D = 10`, `B_w = 8`)
| Strategy | Committed | Raw Restored (`/10`) | Known Wrong Writes | Unknown Write Attempts | Harmed Healthy | Fresh Reload `max_prob_diff` | Strict Locality Pass | Strict Yield `Y` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0_no_repair` | 0 | 0/10 | 0 | 0 | 6 | **`0.00e+00`** | **True** | 0.0 |
| `B1_generative` | 8 | 7/10 (`1` active + `6` incidental) | 0 | 4 | 0 | **`0.00e+00`** | False | 0.0 |
| `B2_ungated_structured` | 8 | 5/10 | 0 | 5 | 1 | **`0.00e+00`** | False | 0.0 |
| `M_evidence_gated` | 8 | 3/10 | 0 | 5 | 0 | **`0.00e+00`** | False | 0.0 |
| `M_full` | 8 | 3/10 | 0 | 5 | 0 | **`0.00e+00`** | False | 0.0 |
| `Oracle_upper_bound` | 8 | **10/10** | 0 | 0 | 0 | **`0.00e+00`** | False | 0.0 |

## 3. Cross-Architecture Synthesis (Replicating Two Core Failure Modes)
1. **Exact Fresh-Subprocess Persistence Verified (`max_prob_diff = 0.00e+00`):** Across all 6 cumulative strategies on `Llama-3.2-3B-Instruct`, isolated fresh-subprocess reload from base weights + `damaged_state.pt` + 8 merged LoRA adapters matches in-process probabilities to `0.00e+00`, resolving v0.5's Llama reload gap.
2. **Replication of Cumulative Same-Entity Interference:** Just as on `Qwen2.5-Omni-3B`, even `Oracle_upper_bound` (`10/10` raw restored, `0` harmed healthy) fails strict per-fact locality (`loc = False`, `Y = 0.0`) after 8 cumulative LoRA merges.
3. **Unknown-Entity Hallucination Bottleneck:** Because `Llama-3.2-3B-Instruct` leaves `22/50` units unmastered/unknown, `4–5` of the `8` blind write slots are consumed by unknown entities (`unk_att = 4–5`), reducing `M_evidence_gated` active recovery and allowing `B1_generative` to achieve higher incidental recovery (`7/10`, of which `6/7` are incidental).
4. **Open-Relation (`CounterFact/zsRE`) & Second Effective Damage (`D3`):** Explicitly `NOT_RUN` (reserved for Batch B).
