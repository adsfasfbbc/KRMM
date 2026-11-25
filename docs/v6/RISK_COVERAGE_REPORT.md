# KRMM v0.6 Risk-Coverage & Full-Policy Intervention Report (`RISK_COVERAGE_REPORT.md`)

- **Artifact:** `results/v6-new/qwen_pilot_g2a_r3/metrics.json`
- **Evaluated Cohort:** `D = 19` paired degraded brand-new v0.6 targets on `Qwen2.5-Omni-3B` (same damaged checkpoint `M_d` as `qwen_pilot_g1_r2`).

## 1. Candidate Generation Separation (Shared Damaged State)
- `generative_reconstruction`: **`21.1%` (4/19)**
- `structured_evidence_graph`: **`73.7%` (14/19)** (`+52.6 pp` over generative), covering `16/19 (84.2%)` with **`87.5%` (14/16)** precision among covered targets.

## 2. Fixed-Candidate Risk-Coverage Curves (Same 16 Covered Structured Candidates)
| Selector | Max Reachable Coverage | AURC (Reachable) | Risk @ 25% | Risk @ 50% | Risk @ 75% |
|---|---:|---:|---:|---:|---:|
| `score_random` | 84.2% | 0.0984 | 0.200 | 0.100 | 0.143 |
| `score_direct_confidence` | 84.2% | **0.0718** | **0.000** | **0.100** | 0.143 |
| `score_candidate_change` | 84.2% | 0.1569 | 0.200 | 0.100 | **0.071** |
| `score_fact_count` | 84.2% | 0.0755 | **0.000** | **0.100** | 0.143 |
| `score_minset_count` | 84.2% | 0.0837 | **0.000** | 0.200 | 0.143 |
| `score_evidence_gate` | 84.2% | 0.0837 | **0.000** | 0.200 | 0.143 |

**Honest Interpretation (Section 6.3):** At `25%` coverage, `score_evidence_gate` achieves `0.0%` risk (vs `20.0%` random), matching `direct_confidence` and `fact_count`. Across the full reachable prefix (`14/16` correct, only `2` errors), `score_evidence_gate` (`AURC=0.0837`) does **not** outperform `direct_confidence` (`0.0718`) or `fact_count` (`0.0755`). Thus, on this pilot cohort, the heuristic support score should **not** be presented as a superior within-coverage ranker; the demonstrated mechanism value lies in **structured reconstruction (`+52.6 pp`)** and **structural abstention on unidentifiable/conflicting/correlated-duplicate cases**.

## 3. Full-Policy Controlled Interventions A–F (Acting on Complete Gating Decision)
| Intervention | Metric | Observed Value |
|---|---|---:|
| D vs A (Sham attribute removal) | `sham_action_invariance_rate` | **100.0% (19/19)** |
| B vs A (Hitting-set removal across all minimal support sets) | `hitting_set_removal_abstain_rate` (given A accept) | **100.0% (15/15)** |
| C vs A (Counterfactual key replacement) | `cf_action_or_cand_change_rate` | **94.7% (18/19)** |
| E (Correlated wrong duplicates) | `correlated_wrong_fooled_gate_rate` | **0.0% (0/19)** |
| F (Twin-world unidentifiable negative control) | `twin_world_negative_control_pass_rate` | **100.0% (2/2 `REQUEST_EVIDENCE`)** |
