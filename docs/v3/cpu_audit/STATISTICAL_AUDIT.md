# KRMM v0.3 Statistical & Offline Re-Computation Audit

**Evidence Source:** `reports/v3-cpu-audit/recomputed_metrics.json`, `reports/v3-cpu-audit/baseline_comparisons.json`  
**Note on `test48`:** Because `test48` has already been opened in v0.3 (`exp09-p2-test48`), all re-computations below on `test48` are explicitly classified as **`post_hoc_audit`** (transparent error-correction audit), **NOT** new confirmatory experiments.

---

## 1. Unified Denominator Ledger (Exact Integer Numerators / Denominators)

| Metric / Stage | `dev` (`exp07`) | `calibration` (`exp08`) | `test48` (`exp09`, post-hoc audit) |
|---|---:|---:|---:|
| `n_total_catalog` | 36 | 36 | 48 |
| `n_stably_mastered` | 33 / 36 (91.67%) | 32 / 36 (88.89%) | 43 / 48 (89.58%) |
| `n_damage_effective` | 25 / 33 (75.76%) | 24 / 32 (75.00%) | 33 / 43 (76.74%) |
| `n_strict_eligible` | 21 / 33 (63.64%) | 16 / 32 (50.00%) | 29 / 43 (67.44%) |
| `n_unique_source_groups_eligible` | 21 / 21 (no dup) | 16 / 16 (no dup) | 29 / 29 (no dup) |
| `direct` strict restored / eligible | 0 / 21 (0.00%) | 1 / 16 (6.25%) | 0 / 29 (0.00%) |
| `generative` strict restored / eligible | 12 / 21 (57.14%) | 8 / 16 (50.00%) | 13 / 29 (44.83%) |
| `structured` candidate produced / eligible | 15 / 21 (71.43%) | 15 / 16 (93.75%) | 24 / 29 (82.76%) |
| `structured` repair attempted / eligible | 15 / 21 (71.43%) | 15 / 16 (93.75%) | 24 / 29 (82.76%) |
| `structured` strict restored / attempted | 14 / 15 (93.33%) | 12 / 15 (80.00%) | 20 / 24 (83.33%) |
| `structured` strict restored / eligible | 14 / 21 (66.67%) | 12 / 16 (75.00%) | 20 / 29 (68.97%) |
| `oracle` (`standard_kl`) strict restored / eligible | 21 / 21 (100.00%) | 14 / 16 (87.50%) | 27 / 29 (93.10%) |

---

## 2. E7 Selector Denominator & Coverage Reconciliation (`test48`, `structured` branch)

In historical `e0_e7_full_final.json`, `strict_success_all_eligible` for `structured` on `test48` divided by `n_attempted = 24` (`12/24 = 50.0%`), conflating attempted units with all `29` strict-eligible units. Corrected unified metrics at `budget_fraction = 0.5` (`selected_count = 12`):

- **Candidate Coverage:** `24 / 29` (**82.76%**)
- **Abstention Rate (all eligible):** `5 / 29` (**17.24%**)
- **Selected Coverage among Attempted:** `12 / 24` (**50.00%**)
- **Selected Coverage among All Strict-Eligible:** `12 / 29` (**41.38%**)
- **Success among Selected (`direct_success_model` / `fixed_decomposed_ere`):** `12 / 12` (**100.00%**, `0` wrong writes)
- **Success among All Strict-Eligible (end-to-end under 50% cap):** **`12 / 29` (41.38%)** — NOT `50.0%`!
- **Comparison (`candidate_change` baseline):** `9 / 12` (**75.00%**) among selected (`3` wrong writes), **`9 / 29` (31.03%)** among all strict-eligible.
- **Comparison (`direct_confidence` baseline):** `10 / 12` (**83.33%**) among selected (`2` wrong writes), **`10 / 29` (34.48%)** among all strict-eligible.

---

## 3. Fair Probability / Brier Comparison Against Baselines

Ranking/binary baselines (`candidate_change`, `direct_confidence`, `entropy`, `self_consistency`, `support_completeness`) are **not** calibrated probabilities (`is_calibrated_probability = False`). When mapped via train-only 1D logistic regression on `dev+cal` and evaluated post-hoc on `test48`:
- **`generative` branch (`n=29`):**
  - Raw `candidate_change` Brier `0.4483` -> train-prob-mapped Brier **`0.2510`** (AUROC `0.5865`).
  - Raw `direct_confidence` Brier `0.4752` -> train-prob-mapped Brier **`0.2555`** (AUROC `0.7019`).
  - `direct_success_model` (`fitted_uncalibrated`): Brier **`0.1334`** (AUROC `0.9327`).
  - `direct_success_independently_calibrated` (fit `dev`, Platt-calibrated on `cal`, `status="calibrated"`): Brier **`0.1153`** (AUROC `0.9327`).
- **`structured` branch (`n_att=24`):**
  - Raw `candidate_change` Brier `0.1667` -> train-prob-mapped Brier **`0.1404`** (AUROC `0.5000`).
  - Raw `direct_confidence` Brier `0.1286` -> train-prob-mapped Brier **`0.1371`** (AUROC `0.7125`).
  - `direct_success_model` (`fitted_uncalibrated`): Brier **`0.1173`** (AUROC `0.8125`).
  - `fixed_decomposed_ere_conditional` (`fitted_uncalibrated`): Brier **`0.1222`** (AUROC `0.8000`).

---

## 4. Bootstrap Robustness & Seed Stability (`structured` vs `generative`, Strict Restoration)

- **Statistical Unit:** Independent `source_group` (verified `0` duplicate clusters in `dev=21`, `cal=16`, `test=29`).
- **5-Seed Paired Bootstrap Stability (`seeds = 107, 202, 303, 404, 505`, `n_boot = 2000`):**
  - `dev` (`n=21`, exploratory): point `+9.52 pp`, 95% CIs across 5 seeds range `[-19.05 pp, +38.10 pp]` (all cross 0).
  - `cal` (`n=16`, exploratory): point `+25.00 pp`, 95% CIs across 5 seeds range `[+6.25 pp, +43.75 pp]` to `[+6.25 pp, +50.00 pp]` (all strictly > 0).
  - `test` (`n=29`, post-hoc audit): point `+24.14 pp`, 95% CIs across all 5 seeds are identical `[+6.90 pp, +41.38 pp]` (strictly > 0).
- **Multiple-Testing Caution:** Multiple branches, predictors, and metrics were inspected across `dev`/`cal`/`test`; post-hoc `test48` comparisons must be treated as diagnostic audits, not fresh confirmatory tests.
