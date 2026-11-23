# KRMM v0.3 EvidenceGraph & Compositional Reconstruction Audit (P4)

**Evidence Source:** `krmm/v3/knowledge_graph.py`, `krmm/v3/reconstruction.py`, `reports/v3-cpu-audit/recomputed_metrics.json`

---

## 1. Original vs Corrected Minimal Support-Set Analysis (Zero Rules Added)

Without modifying `SINGLETON_CONSTRAINTS` or using any `test48` error analysis to add rules, we re-ran the corrected minimal support-set classifier (`EvidenceGraph.find_minimal_support_sets`) on all archived `atomic_facts` across `dev`, `calibration`, and `test48`:

- **Candidate Answer Differences (Original vs Corrected):** **`0` / 21** (`dev`), **`0` / 16** (`cal`), **`0` / 29** (`test`) — 100% identical candidate answers.
- **`true_composition` Count (Original vs Corrected):** `dev: 0 vs 0`, `cal: 3 vs 3`, `test: 5 vs 5` — exact match.
- **Fine-Grained Minimal Support-Set Classification:**

| Split (`n_strict_eligible`) | `single_hop` (correct / total) | `redundant_support` (correct / total) | `minimal_multi_premise` (correct / total) | `conflicting_evidence` | `insufficient_evidence` |
|---|---:|---:|---:|---:|---:|
| `dev` (21) | 7 / 8 (87.5%) | **7 / 7 (100.0%)** | 0 / 0 | 6 | 0 |
| `calibration` (16) | 5 / 5 (100.0%) | **8 / 8 (100.0%)** | 1 / 2 (50.0%) | 1 | 0 |
| `test48` (29, post-hoc) | 8 / 10 (80.0%) | **12 / 12 (100.0%)** | 0 / 2 (0.0%) | 5 | 0 |
| **Combined (66)** | **20 / 23 (87.0%)** | **27 / 27 (100.0%)** | **1 / 4 (25.0%)** | **12** | **0** |

**Key Audit Insight:**
- Whenever **multiple independent minimal support sets** agreed (`redundant_support`), candidate accuracy was **27 / 27 (100.0%)** across all three splits (note: 4 of the 8 `true_composition` units—1 in `cal`, 3 in `test`—actually had multiple distinct size-2 minimal sets, hence fell into `redundant_support` with `4/4 = 100%` accuracy).
- Conversely, when a unit relied on a **single uncorroborated minimal multi-premise chain** (`minimal_multi_premise`, no backup minimal set), accuracy was only **1 / 4 (25.0%)** (`1/2` cal, `0/2` test).

---

## 2. Three-Level Separation of Causal Evidence

To prevent overclaiming from rule-engine ablations, future reports must strictly separate:
1. **Rule-level sufficiency (`key_intervention_result`, `sham_intervention_result`):** Verified offline (`tests/v3_cpu/test_counterfactual_rule_controls.py`). Dropping a premise from a minimal support set changes the rule-engine output; adding an irrelevant `sham_control` fact leaves it unchanged. This proves *logical sensitivity of `EvidenceGraph`*, **not** internal neural computation.
2. **Candidate-level correctness:** Verified offline against evaluator gold labels on archived probes (`20/24 = 83.3%` non-null precision on `test48`; `27/27 = 100%` on `redundant_support`).
3. **Model-level causal dependence:** Proving whether the neural model internally relies on the same latent attributes requires direct model-level intervention/ablation during forward passes — explicitly marked **`PENDING_GPU_VALIDATION`**.
