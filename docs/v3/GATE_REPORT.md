# KRMM v0.3 Gate & Scientific Validity Report

**Branch:** `codex/krmm-v03` (from `codex/krmm-v02`)  
**Model:** `Qwen/Qwen2.5-Omni-3B` (revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`, verified SHA256)  
**Benchmark Catalog:** `docs/v3/DATA_MANIFEST.jsonl` (140 entity-disjoint units: `dev=36`, `calibration=36`, `test=48`, `anchor=12`, `unknown_control=8`; SHA256 `b4a1fc522492868c7360e921b7e02e787f94b91821c9bc7c3768e8b47e382b4d`)

---

## 1. Gate Summary

| Gate | Scope | Status | Primary Evidence |
|---|---|---|---|
| **Boundary Unit Tests** | Evaluator/Method isolation, inverse check, reload, rollback, budget, abstention | **PASS (13/13 + 5/5 v0.2)** | `tests/v3/test_v3_boundaries.py`, `tests/test_contracts.py` |
| **G0 (P0)** | Clean mastery, local effective D1 damage, supervised LoRA upper bound (`orthogonal_retain_proj`) | **PASS (dev36, cal36, test48)** | `exp07` (21/21), `exp08` (14/16 = 87.5%), `exp09` (27/29 = 93.1%) |
| **G0-D2 (Pilot)** | Unlikelihood response suppression (`D2_response_suppression`, `dev12`) | **PARTIAL / FRAGILE** | `exp04`: 5/12 effective, 2 failed `same_entity_other_relation` locality, 3/3 oracle strict |
| **G1 (P1: H1 + H3 Candidates)** | Reference-free M2 discovery under fixed query budget + M3 structured reconstruction | **PASS (dev36, cal36, test48)** | `exp05c`, `exp10`, `exp11`; `e0_e7_full_final.json` |
| **G2 (P2–P3: H2 + H3 Strict Repair)** | Out-of-sample M4 Decomposed ERE prediction + constrained writeback vs baselines on sealed `test48` | **PASS (test48 confirmatory)** | `exp09-p2-test48`, `e0_e7_full_final.json` |
| **G3-P4 (Multimodal)** | Real image + audio complementary evidence paths | **NOT RUN (DEFERRED)** | Text-domain P0–P3 completed first per staged priority |
| **G3-P5 (Second Model / Multi-seed)** | Cross-architecture replication & multi-seed sweep | **NOT RUN (DEFERRED)** | Single-model P0–P3 completed; `orthogonal_retain_proj` single-seed |

---

## 2. Detailed Gate Verification

### 2.1 Gate G0 (P0 Baseline, Damage Validity, and Supervised Writeback Ceiling)
1. **Clean Mastery (`M0.3`):** Across 4 phrasings × 2 option permutations (8 checks), clean mastery was **33/36 (91.67%)** on `dev`, **32/36 (88.89%)** on `calibration`, and **43/48 (89.58%)** on sealed `test48`, vs **0/8 (0.0%)** on `unknown_control`. Unmastered items were strictly excluded prior to damage.
2. **Damage Effectiveness & Non-Invertibility (`M1.2`):** With `D1_wrong_association` (`damage_steps=64`), heldout accuracy on mastered items collapsed to `33.3%` across all three splits. On sealed `test48`, assigned wrong options were balanced (`A:11, B:12, C:12, D:13`), `best_in_sample_inverse_accuracy = 0.4583`, `cyclic_inverse_accuracy = 0.375`, and `no_deterministic_inverse = true`.
3. **Independent 4-Category Locality Audit (`M1.3`):** Damage passed train retention (`min = 0.9821`) and independent 4-category locality on **21/25** effective `dev` units, **16/24** `cal` units, and **29/33** `test48` units. Non-local damages (`4+8+4`) were excluded before downstream evaluation.
4. **Supervised Writeback Upper Bound (`M1.4` / E5 vs E6):** Unconstrained LoRA (`exp06`, `retain_weight=0`) achieved only **4/9 (44.4%)** strict restorations due to `mean_kl_shift` up to `2.86–7.64`, whereas `orthogonal_retain_proj` achieved **9/9 (100%)** on `exp06`, **21/21 (100%)** on `dev36`, **14/16 (87.5%)** on `cal36`, and **27/29 (93.10%)** on sealed `test48`. Every single strict restoration across all splits (**142/142 = 100%**) passed isolated fresh-process reload verification.

### 2.2 Gate G1 / H1 (M2 Damage Discovery Under Fixed Budget)
Across `dev36` (`exp05c`), `calibration` (`exp10`), and sealed `test48` (`exp11`), `m2_reference_free` consistently outperformed `simple_direct_baseline` while separating damaged mastered knowledge from never-known `unknown_control` items:

| Split | Method | AUROC | AUPRC | Recall@Budget | Unknown FPR | Mean Query Cost |
|---|---|---:|---:|---:|---:|---:|
| `dev36` (`n=41`) | `simple_direct_baseline` | 0.6287 | 0.7588 | 0.6250 | 0.8750 | 3.00 |
| `dev36` (`n=41`) | **`m2_reference_free`** | **0.8125** | **0.8844** | **0.8125** | **0.0000** | 4.10 |
| `cal36` (`n=40`) | `simple_direct_baseline` | 0.7656 | 0.8431 | 0.6875 | 0.6250 | 3.00 |
| `cal36` (`n=40`) | **`m2_reference_free`** | **0.9902** | **0.9846** | **0.9375** | **0.0000** | 4.43 |
| `test48` (`n=51`) | `simple_direct_baseline` | 0.8182 | 0.7410 | 0.8182 | 0.8750 | 3.00 |
| `test48` (`n=51`) | **`m2_reference_free`** | **0.9177** | **0.8645** | **0.9091** | **0.0000** | 4.57 |

- *Caveat recorded honestly:* On `test48`, `m2_reference_free` incurred `healthy_false_positive_rate = 0.1905` vs `0.0952` for `simple_direct_baseline`, and `m2_reference_based` had `unknown_false_positive_rate = 0.75–0.875` because its simple reference threshold did not gate on clean-state mastery.

### 2.3 Gate G2 / H2 & H3 (Reconstruction, Writeback, and Decomposed Repairability)
1. **H3 Confirmatory Test (`test48`, n=29 strict-eligible):**
   - `direct`: **0 / 29 (0.0%)** strict restorations (`28` wrong pseudolabels written)
   - `generative`: **13 / 29 (44.83%)** strict restorations (`16` wrong pseudolabels written)
   - `structured` (`EvidenceGraph`): **20 / 29 (68.97%)** strict restorations (`82.76%` coverage `24/29`, `83.33%` precision when non-null, `91.67%` locality pass, `4` wrong pseudolabels, `5` clean abstentions)
   - **Paired 95% Cluster-Bootstrap CI (`structured` vs `generative`):** **+24.14 pp**, 95% CI **`[+6.90 pp, +41.38 pp]`** (strictly > 0; replicates `cal36` `+25.00 pp [+6.25, +50.00]`).
2. **H2 Confirmatory Test (Fit on `dev+cal`, Eval on Sealed `test48`):**
   - On `structured` (`n=24`): `decomposed_ere` AUROC **0.8125**, AUPRC **0.9592**, Brier **0.1120**, Selected@50% **100.0% (12/12)** vs `candidate_change` (AUROC `0.5000`, Brier `0.1667`, Selected@50% `75.00%`) and `direct_confidence` (AUROC `0.7125`, Brier `0.1286`, Selected@50% `83.33%`).
   - On `generative` (`n=29`): `decomposed_ere` AUROC **0.9327**, AUPRC **0.9320**, Brier **0.1224**, Selected@50% **78.57%** vs `candidate_change` (AUROC `0.5865`, Selected@50% `35.71%`); paired Selected@50% diff **+20.69 pp**, 95% CI **`[+6.90 pp, +37.93 pp]`**.

---

## 3. Honest Negative & Null Findings
1. **D1 step count:** `damage_steps=24` (`exp02`) was completely ineffective (`0/12` effective) on mastered commonsense facts; `damage_steps=64` was required and frozen before `cal`/`test`.
2. **D2 fragility:** `D2_response_suppression` without KL regularization (`exp04`) yielded only `5/12` effective and `3/12` strict-eligible units (2 failed `same_entity_other_relation` locality due to representation drift `l2 ≈ 49.7`).
3. **True multi-premise composition sample size:** Multi-premise (`structured_composition`) chains only fired when singleton rules did not cover the unit: `0` on `dev`, `3` on `cal` (`2/3`), `5` on `test` (`3/5`), totaling **`5/8 (62.5%)`** across `dev+cal+test`, whereas `single_hop` rules accounted for **`43/46 (93.48%)`** precision.
4. **Why independent locality matters:** On both `timpani` (`cal`) and `crystal wine glass` (`test`), oracle LoRA repair hit `100%` target accuracy and kept all ~125/161 other locality probes clean, yet broke `same_entity_other_relation` (`1.0 -> 0.0`, KL `2.59` / `0.19`) and was rightly rejected from strict restoration.
