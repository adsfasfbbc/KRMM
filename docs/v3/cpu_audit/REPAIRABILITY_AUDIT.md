# KRMM v0.3 M4 Repairability Predictor Audit (P1)

**Evidence Source:** `krmm/v3/repairability.py`, `reports/v3-cpu-audit/baseline_comparisons.json`

---

## 1. Probability Factorization Bug Fixed
- **Historical Bug:** `DecomposedPredictor` fitted `rel_model` on `y_cand_correct` (`P(cand_correct | x)`) and `write_model` on unconditional `y_strict_success` (`P(strict_success | x)`), then multiplied them (`P(cand_correct | x) * P(strict_success | x)`), double-counting candidate correctness because `strict_success = 1` already implies `cand_correct = 1`.
- **Corrected Implementation (`krmm/v3/repairability.py`):**
  - `DirectSuccessPredictor`: directly models `P(strict_success | x)`.
  - `CandidateReliabilityPredictor`: models `P(candidate_correct | x)`.
  - `ConditionalWriteFeasibilityPredictor`: models `P(strict_success | candidate_correct = 1, x)` strictly on the `y_cand_correct == 1` subset (or explicit `oracle_writeback` records), returning `calibration_status = "insufficient_data"` whenever `n < 6`, single-class, or `min(n_pos, n_neg) < 2`.
  - `DecomposedPredictor`: computes `P(strict_success | x) = P(candidate_correct | x) * P(strict_success | candidate_correct = 1, x)` and strictly distinguishes `untrained`, `fitted_uncalibrated`, `calibrated` (only after independent calibration split fitting), and `insufficient_data`.

---

## 2. Severe Data Sparsity in Conditional Write Feasibility

Offline audit of `exp07` (`dev36`), `exp08` (`cal36`), and `exp09` (`test48`) reveals why the two-stage write-risk component cannot be reliably validated on the current dataset:

| Training Pool | Branch | `n_attempted` | `cond_pos` (`cand=1, strict=1`) | `cond_neg` (`cand=1, strict=0`) | `oracle_neg` | Conditional Write Status |
|---|---|---:|---:|---:|---:|---|
| `dev` only | `structured` | 15 | 14 | **0** | **0** | **`insufficient_data`** |
| `dev` only | `generative` | 21 | 12 | **0** | **0** | **`insufficient_data`** |
| `cal` only | `structured` | 15 | 12 | **2** | **2** | `fitted_uncalibrated` (`near_separation_warning=True`; `rel_model` = `insufficient_data` with 1 neg) |
| `cal` only | `generative` | 16 | 8 | **2** | **2** | `fitted_uncalibrated` (`near_separation_warning=True`) |
| `dev + cal` | `structured` | 30 | 26 | **2** | **2** | `fitted_uncalibrated` (`near_separation_warning=True`) |
| `dev + cal` | `generative` | 37 | 20 | **2** | **2** | `fitted_uncalibrated` (`near_separation_warning=True`) |

**Key Finding:** On `dev` alone there are **0** conditional writeback failures; across the entire `dev + cal` pool there are only **2** writeback failures (`concert bass drum` and `timpani` in `cal36`). Under `standard_kl` (`test48` oracle strict success = `27/29 = 93.1%`), strict success is almost completely dominated by candidate correctness!

---

## 3. Offline Post-Hoc Re-Computation (`dev+cal -> test48`, Post-Hoc Audit Only)

| Method | Status | Structured AUROC (`n=24`) | Structured Brier | Generative AUROC (`n=29`) | Generative Brier | Generative Selected@50% |
|---|---|---:|---:|---:|---:|---:|
| `legacy_buggy_ere` (double-counted) | `legacy_mislabeled` | 0.8125 | 0.1120 | 0.9327 | 0.1224 | 78.6% |
| **`direct_success_model`** | `fitted_uncalibrated` | **0.8125** | **0.1173** | **0.9327** | **0.1334** | **78.6%** |
| **`candidate_reliability_model`** | `fitted_uncalibrated` | **0.8250** | 0.1411 | 0.8942 | 0.1603 | 78.6% |
| `fixed_decomposed_ere_conditional` | `fitted_uncalibrated` | 0.8000 | 0.1222 | 0.9327 | 0.1444 | 78.6% |
| `fixed_decomposed_ere_oracle_write` | `fitted_uncalibrated` | 0.8250 | 0.1245 | 0.9135 | 0.1523 | 78.6% |
| `direct_success_independently_calibrated` (fit `dev`, cal `cal`) | **`calibrated`** | `insufficient_data` (`dev` has 1 neg) | — | **0.9327** | **0.1153** | **78.6%** |
| `direct_confidence` (train prob mapped) | `fitted_uncalibrated` | 0.7125 | 0.1371 | 0.7019 | 0.2555 | 64.3% |
| `candidate_change` (train prob mapped) | `fitted_uncalibrated` | 0.5000 | 0.1404 | 0.5865 | 0.2510 | 35.7% |

---

## 4. Scientific Assessment of H2
- **Is there verifiable incremental value of decomposition over `DirectSuccessPredictor` in current data?** **No.** `DirectSuccessPredictor` (`0.8125` / `0.9327`) and `CandidateReliabilityPredictor` (`0.8250` / `0.8942`) match or slightly exceed `fixed_decomposed_ere_conditional` (`0.8000` / `0.9327`), because there are only `2` writeback failures in `dev+cal`.
- **What IS supported:** Diagnostic features strongly outperform simple heuristics (`candidate_change`, `direct_confidence`, `entropy`, `self_consistency`), and independent Platt calibration (`dev -> cal -> test` on generative) genuinely reduces Brier (`0.1675 -> 0.1153`).
- **Verdict on H2:** **需要收窄** — diagnostic predictability is supported, but *separating write-risk from candidate reliability* lacks sufficient write-failure samples in the current `standard_kl` regime and remains **`PENDING_GPU_VALIDATION`** under a harder writeback regime (`NEXT_GPU_PROTOCOL.md`, Experiment G2).
