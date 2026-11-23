# G2-A Report: Evidence Structure, Strong Baselines & Model-Level Counterfactual Interventions

- **Artifact:** `results/v5-new/g2_mechanism_results.json`
- **Model:** `/root/rivermind-data/models/Qwen2.5-Omni-3B`
- **Runtime:** `865.4s`
- **Evaluated Cohort:** `n = 22` cleanly mastered, effectively D1-damaged brand-new v0.5 entities (`assignment_audit.no_deterministic_inverse = True`).

## 1. Strong Baseline Comparison (Paired on Exact Same `n = 22` Targets)
| Method | Overall Accuracy | Coverage / Accepted | Precision When Accepted |
|---|---:|---:|---:|
| `rule_only` (zero model facts) | 0.0% (0/22) | 0.0% | N/A |
| `entity_name_only` | 4.5% (1/22) | 100.0% | 4.5% |
| `direct_probing` | 4.5% (1/22) | 100.0% | 4.5% |
| `self_consistency` (3 direct probes) | 4.5% (1/22) | 100.0% | 4.5% |
| `generic_cot` | 9.1% (2/22) | 100.0% | 9.1% |
| `generative_reconstruction` | 50.0% (11/22) | 100.0% | 50.0% |
| `existing_evidence_graph` (un-gated) | **77.3%** (17/22) | 90.9% (20/22) | 85.0% (17/20) |
| **`evidence_gated` (v0.5)** | 68.2% (15/22) | 72.7% (16/22) | **93.75% (15/16)** |

## 2. Support-Class Stratification
- `single_decisive`: `6/6` (**100.0%**), `6/6` accepted
- `redundant_distinct_attributes`: `9/10` (**90.0%**), `10/10` accepted (`giant anteater` is the single error)
- `single_chain`: `2/4` (`50.0%`), `0/4` accepted (correctly gated out)
- `conflicting_support`: `0/2` (`0.0%`), `0/2` accepted (correctly gated out)

## 3. Real Model-Level Causal Interventions (A–E)
Evaluated directly via forward passes on the damaged `Qwen2.5-Omni-3B` model:
- **Sham attribute removal invariance (`D vs A`):** **`100.0%` (22/22)** argmax invariance; mean `|ΔP_gold| = 0.045`.
- **Key support removal (`B vs A`):** mean `ΔP_gold = -0.145`; argmax flip rate `13.6%`.
- **Counterfactual key flip (`C vs A`):** mean `ΔP_gold = -0.256`; argmax flip rate `27.3%` (e.g., `bodhran drum`: `P_gold` drops `0.975 -> 0.293` on key removal and `0.975 -> 0.027` on counterfactual flip, while sham removal shifts `P_gold` by `0.004`).

**Conclusion:** Direct probing, self-consistency, and generic CoT fail catastrophically (`4.5%–9.1%`) on D1-corrupted facts. Structured extraction + support-structure gating achieves `93.75%` precision (`15/16`) and demonstrates genuine model-level causal sensitivity to supporting attributes with `100%` sham invariance.
