# KRMM v0.4 CPU Mechanism Validation Report (Synthetic Evidence Graphs)

> **Scope Warning:** All results below come from **deterministic CPU synthetic fact graphs** (`scripts/v4/run_cpu_mechanism.py`, `results/v4-new/cpu_mechanism_results.json`) and CPU mock pipeline dry-runs (`scripts/v4/run_blind_discovery.py`, `scripts/v4/run_repair_matrix.py`). They verify algorithm logic, minimal support-set math, provenance dependence discounting, and policy gating — **NOT** real neural model repair performance.

---

## 1. Stratified 8-Condition Synthetic Evidence Benchmark (Section 5.2)

| Synthetic Condition | v0.4 Category | Identifiable | Candidate | `n_ms` | Overlap | Est. Reliability | Policy Action |
|---|---|---|---:|---:|---:|---:|---|
| 1. No decisive evidence | `insufficient_support` | False | `None` | 0 | 0.00 | 0.00 | `REQUEST_EXTERNAL_EVIDENCE` |
| 2. Single decisive fact | `single_decisive` | True | `B` | 1 | 0.00 | 0.72 | `ATTEMPT_PARAMETRIC_REPAIR` |
| 3. Single multi-premise chain | `single_chain` | True | `C` | 1 | 0.00 | 0.40 | `REQUEST_EXTERNAL_EVIDENCE` |
| 4. Same-source spurious redundancy | `redundant_same_source` | True | `B` | 2 | 0.00 | 0.50 | `REQUEST_EXTERNAL_EVIDENCE` |
| 5. Distinct semantic paths (same model) | `redundant_distinct_paths` | True | `B` | 2 | 0.00 | 0.85 | `ATTEMPT_PARAMETRIC_REPAIR` |
| 6. Independent external sources | `independent_source_support` | True | `B` | 2 | 0.00 | 0.90 | `ATTEMPT_PARAMETRIC_REPAIR` |
| 7. Correlated wrong consensus (same source) | `redundant_same_source` | True | `A` (wrong vs `D`) | 2 | 0.00 | 0.50 | **`REQUEST_EXTERNAL_EVIDENCE` (blocked!)** |
| 8. Conflicting premises | `conflicting_support` | False | `None` | 0 | 0.00 | 0.00 | **`ABSTAIN`** |

---

## 2. Five Causal Interventions on Evidence Support (Section 5.3)

Starting from a corroborated target (`feathers` + `warm_blooded` + `hard_shelled_eggs` + sham `container_color=brown`):

| Intervention | Category | Identifiable | Conflict | Est. Reliability | Policy Action | Expected Behavior Verified? |
|---|---|---|---:|---:|---|---|
| **Complete Support** | `redundant_distinct_paths` | True | False | 0.85 | `ATTEMPT_PARAMETRIC_REPAIR` | Yes |
| **Key Removal** (drop `feathers`) | `single_chain` | True | False | **0.40** | **`REQUEST_EXTERNAL_EVIDENCE`** | Yes (reliability drops, blocks blind write) |
| **Counterfactual Replacement** (`feathers -> scales`) | `conflicting_support` | False | **True** | **0.00** | **`ABSTAIN`** | Yes (triggers conflict abstention) |
| **Sham Removal** (drop `container_color`) | `redundant_distinct_paths` | True | False | 0.85 | `ATTEMPT_PARAMETRIC_REPAIR` | Yes (strictly invariant) |
| **Correlated Wrong Support** (2× same wrong attr) | `redundant_same_source` | True | False | **0.50** | **`REQUEST_EXTERNAL_EVIDENCE`** | Yes (discounted, prevents blind wrong write) |

---

## 3. End-to-End CPU Mock Pipeline & Unit Tests
- `scripts/v4/run_blind_discovery.py --mode cpu_mock`: all 5 comparators (`direct_confidence`, `self_consistency`, `direct_paraphrases`, `constraint_conflict_only`, `full_m2`) stayed strictly within equal hard budget (`<= 18` queries over 6 units).
- `scripts/v4/run_repair_matrix.py --mode cpu_mock`: verified full mixed-pool blind execution (`damaged_mastered`, `healthy_mastered`, `originally_unknown`) with isolated post-hoc evaluator scoring (`End-to-End Yield`, `Wrong-Write Rate`, `Healthy-Unit Harm Rate`).
- `pytest tests/v4/ -v`: **23 / 23 PASSED**; `pytest tests/v3_cpu/ -q`: **46 / 46 PASSED**; `pytest tests/v3/ -q`: **13 / 13 PASSED**; `unittest tests/test_contracts.py`: **5 / 5 PASSED** (**87 / 87 total**).
