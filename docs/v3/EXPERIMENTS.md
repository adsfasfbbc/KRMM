# KRMM v0.3 Experiment Journal

This journal records the experimental design, exact commands/configs, quantitative outcomes, failure diagnoses, and self-iteration decisions for every KRMM v0.3 run. Historical v0.1 (`EXPERIMENTS.md`) and v0.2 (`docs/v2/REPORT.md`) remain frozen and unmodified.

---

## Pre-Experiment Setup & Verification
- **Branch:** `codex/krmm-v03` (branched from `codex/krmm-v02` at `bbf4ab0`)
- **Hardware:** NVIDIA RTX 3090 (24576 MiB VRAM), Python 3.12.13, PyTorch 2.12.1+cu130, Transformers 4.57.6, PEFT 0.15.2
- **Model:** `Qwen/Qwen2.5-Omni-3B` pinned at revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e` (all shards SHA256-verified against `docs/v2/model-integrity.json`)
- **Unit & Boundary Tests:** `python3 -m pytest tests/v3/ -v` (13/13 passed) and `python3 -m unittest tests/test_contracts.py` (5/5 passed).

---

## Experiment V3-01: Clean Mastery, Control Baseline & Attribute Probe Survey (`dev`)

### 1. Design
- **Goal:** Answer Acceptance Question A ("Does the clean model genuinely master the new high-frequency commonsense benchmark across diverse phrasings and option permutations?") and establish clean baselines for independent retention controls (`same_entity_other_relation`, `audit_general`, `unknown_control`).
- **Command:** `python3 -u scripts/v3/run_mastery_survey.py --out results/v3-new/exp01-mastery-survey-dev`
- **Protocol:** 36 `dev` units (12 `vertebrate_class`, 12 `everyday_material`, 12 `sound_source_mechanism`) evaluated on 4 phrasing templates `(0, 1, 2, 3)` + 2 option permutations `(0, D/C/B/A)` and `(1, B/D/A/C)`, plus heldout templates `(4, 5)` + heldout permutation `(4, C/A/D/B)`, `min_choice_mass >= 0.25`. Also evaluated 12 `anchor` units, 8 `unknown_control` units, 8 `train_general` controls, and 20 `audit_general` controls.

### 2. Quantitative Results (`results/v3-new/exp01-mastery-survey-dev/summary.json`)
- **Overall Dev Clean Mastery:** **33 / 36 (91.67%)** (exceeds G0 threshold of 75%).
- **Heldout Unseen Templates + Permutation Among Mastered:** **33 / 33 (100.0% of mastered; 91.67% overall)**.
- **Per-Domain Breakdown:**
  - `everyday_material`: **12 / 12 (100.0%)**, `same_entity_ctrl_acc = 91.67%`, `clean_structured_acc = 66.67%`
  - `vertebrate_class`: **11 / 12 (91.67%)** (excluded: `great white shark`), `same_entity_ctrl_acc = 100.0%`, `clean_structured_acc = 41.67%`
  - `sound_source_mechanism`: **10 / 12 (83.33%)** (excluded: `orchestral triangle`, `crash cymbal`), `same_entity_ctrl_acc = 91.67%`, `clean_structured_acc = 83.33%`
- **Anchor Mastery (Train Retain Pool):** **11 / 12 (91.67%)**
- **Unknown Control Mastery (M2 Negative Control):** **0 / 8 (0.00%)** (8/8 `failed_screen_accuracy`)
- **General Controls:** `train_general_acc = 87.5%` (7/8), `audit_general_acc = 80.0%` (16/20)
- **Peak VRAM:** 11.16 GiB

### 3. Diagnosis & Self-Iteration Insights
1. **Question A Verified:** Shifting from obscure museum instruments (31.25% mastery in v0.2) to high-frequency commonsense domains raised clean mastery to **91.67%** even under strict 6-check phrasing + permutation screening.
2. **Attribute Probe Bias Identified:** Raw output inspection of `clean_structured_acc` showed two actionable issues for M3 (to address after P0/G0):
   - Binary probe `"Is a {entity} warm-blooded or cold-blooded?"` induced first-option bias (`"Warm-blooded"`) on ectotherms (`nile crocodile`, `king cobra`, `galapagos tortoise`, `atlantic salmon`, `bluefin tuna`).
   - `silicone baking spatula` rightly answered `"remains rigid and does not melt"` at 250C, revealing that `heat_behavior` alone does not separate refractory ceramics from heat-resistant elastomers.
3. **Next Step:** Execute Experiment V3-02 (`p0_baseline_d1` on `dev`, `limit=12`, balanced 4 per domain) to test Questions B and C (damage locality and oracle writeback feasibility under the v0.2 reference LoRA setup + expanded anchor retention).

---

## Experiment V3-02: P0 Baseline D1 Wrong-Association (`damage_steps=24`, `retain_weight=50.0`)

### 1. Design
- **Goal:** Evaluate the v0.2 reference LoRA parameters (`rank=8, alpha=16, last_layers=4, damage_steps=24, lr=5e-4, retain_weight=50.0, l2=1e-4`) on the new high-frequency benchmark with multi-template anchor retention (`retain_batch_size=24`) and permuted wrong-target injection (`D1_wrong_association`).
- **Config:** `configs/v3/p0_baseline_d1.json`
- **Command:** `python3 -u -m krmm.v3.runner --config configs/v3/p0_baseline_d1.json --out results/v3-new/exp02-p0-d1-dev12 --split dev --limit 12`

### 2. Quantitative Results (`results/v3-new/exp02-p0-d1-dev12/metrics.json`)
- **Total Candidates (balanced 4/domain):** 12 (`stably_mastered = 12/12 = 100.0%`, `heldout_clean_mastered = 12/12 = 100.0%`)
- **Damage Effective:** **0 / 12 (0.0%)** (`damage_not_effective_or_train_retain_failed: 12`)
- **Independent Damage Locality Pass (among evaluated):** **12 / 12** (`train_ret = 0.982 - 1.000`, `loc_pass = True`)
- **Strict Eligible:** 0 / 12
- **Gate G0:** **FAIL** (`A_clean_mastery_ok: true`, `B_damage_local_and_effective_ok: false`, `C_oracle_strict_writeback_ok: false`)
- **Elapsed Time:** 236.2s, Peak VRAM: 11.16 GiB

### 3. Diagnosis & Self-Iteration Decision
- Inspecting `damage_logs` showed `target_loss` descending smoothly (`6.0 - 8.4` at step 0 -> `0.44 - 2.75` at step 23) while `retain_kl` stayed tiny (`0.002 - 0.064`).
- High-frequency commonsense knowledge is substantially more entrenched than v0.2's obscure museum instruments, and with `damage_use_permuted_target=True` across 2 variants x 2 permutations, 24 total steps gave only 6 updates per exact prompt.
- **Single-Factor Iteration (V3-03):** Keep all other parameters fixed (including the v0.2 reference repairer at 32 steps, lr `5e-4`, retain_weight `50.0`) and increase ONLY `damage_steps` from `24` to `64` (`configs/v3/p0_d1_steps64.json`) to test whether stronger D1 convergence achieves effective damage while preserving independent locality.

---

## Experiment V3-03: P0 G0 Positive Control — D1 Wrong Association (`damage_steps=64`)

### 1. Design
- **Goal:** Re-test Questions A, B, and C for Gate G0 under `D1_wrong_association` with `damage_steps=64`, keeping the v0.2 reference repairer (`rank=8, alpha=16, last_layers=4, repair_steps=32, lr=5e-4, retain_weight=50.0`) intact.
- **Config:** `configs/v3/p0_d1_steps64.json`
- **Command:** `python3 -u -m krmm.v3.runner --config configs/v3/p0_d1_steps64.json --out results/v3-new/exp03-p0-d1-s64-dev12 --split dev --limit 12`

### 2. Quantitative Results (`results/v3-new/exp03-p0-d1-s64-dev12/metrics.json`)
- **A. Clean Mastery:** **12 / 12 (100.0%)** (`heldout_clean_mastered = 12/12`) -> **PASS** (`>= 75%`)
- **B. Damage Validity & Independent Locality:**
  - `damage_effective = 10 / 12 (83.33%)` (excluded: `acoustic guitar`, `trumpet` where `dmg_heldout >= 0.67`)
  - `damage_independent_locality_pass = 10 / 10 (100.0%)` across all 126 heldout controls (`same_entity_other_relation`, `other_entity_unseen_template`, `near_neighbor_invariant`, `general_commonsense`)
  - `min_damage_train_retention = 0.9821`
  - `strict_eligible = 10 / 12 (83.33%)` -> **PASS** (`>= 8 units` and `>= 50%`)
- **C. Oracle Strict Writeback (Fresh-Process Reload Verified):**
  - `oracle_strict_restorations = 10 / 10 (100.0%)` (probability reload error `<= 1e-4`, unseen phrasings + unseen option permutation recovered to 100%, 4-category independent locality pass `10/10`) -> **PASS** (`>= 75%`)
- **Gate G0 (D1):** **PASS** (`A=true, B=true, C=true`)
- **Non-Oracle Branch Comparison (Descriptive, All 10 Strict-Eligible):**
  - `direct`: **1 / 10 (10.0%)**
  - `generative`: **8 / 10 (80.0%)** (2 wrong pseudolabels written: `iron carpentry nail`, `copper electrical wire`)
  - `structured`: **7 / 10 (70.0% all-eligible; 7 / 7 = 100.0% precision when candidate produced, 0 wrong pseudolabels)**
- **Elapsed Time:** 1659.0s (27.6 min), Peak VRAM: 11.16 GiB

### 3. Diagnosis & Next Step
1. **G0 Breakthrough:** Replacing obscure museum instruments with high-frequency commonsense domains and adding multi-template anchor retention solved the catastrophic locality collapse seen in v0.2 (0/10 -> **10/10 independent locality pass** on both damage and oracle repair).
2. **Safety Advantage of EvidenceGraph:** On the 3 abstained units (`stainless steel frying pan`, `copper electrical wire`, `bottlenose dolphin`), `EvidenceGraph` detected internal attribute conflicts and abstained rather than writing wrong pseudolabels, yielding **0% wrong-pseudolabel writebacks** vs 20% for unconstrained generative reconstruction.
3. **Next Step (Experiment V3-04):** Verify G0 on the second damage mechanism **`D2_suppression`** (`configs/v3/p0_d2_steps64.json`, `limit=12` on `dev`).

---

## Experiment V3-04: P0 Second Damage Mechanism — D2 Targeted Suppression (`damage_steps=64`)

### 1. Design
- **Goal:** Evaluate whether unbounded unlikelihood response suppression (`D2_suppression`, `-log(1 - P(gold))`) satisfies G0 (`B` and `C`) under the exact same hyperparameters as V3-03 (`damage_steps=64`, `retain_weight=50.0`).
- **Config:** `configs/v3/p0_d2_steps64.json`
- **Command:** `python3 -u -m krmm.v3.runner --config configs/v3/p0_d2_steps64.json --out results/v3-new/exp04-p0-d2-s64-dev12 --split dev --limit 12`

### 2. Quantitative Results (`results/v3-new/exp04-p0-d2-s64-dev12/metrics.json`)
- **A. Clean Mastery:** **12 / 12 (100.0%)**
- **B. Damage Validity & Independent Locality:**
  - `damage_effective = 5 / 12 (41.67%)` (`7` units did not drop sufficiently)
  - `damage_independent_locality_pass = 3 / 5` (`2` failed `same_entity_other_relation` when unlikelihood pushed `l2` up to ~49.7 and suppressed the same letter on an unrelated relation of the same entity)
  - `strict_eligible = 3 / 12 (25.0%)` (below 50% threshold for standalone D2 at `damage_steps=64`)
- **C. Oracle Writeback:**
  - `oracle_write_feasible_count = 5 / 5 (100.0%)` among all operationally eligible damaged units
  - `oracle_strict_restorations = 3 / 3 (100.0%)` among strict-eligible units
- **Elapsed Time:** 1109.3s (18.5 min), Peak VRAM: 11.16 GiB

### 3. Diagnosis & Progression Decision
- **Gate G0 Status:** **PASSED on D1 (`V3-03`: 12/12 mastery, 10/12 strict-eligible, 10/10 oracle strict restoration).** Writeback feasibility `C` is **100%** across both D1 (`10/10`) and D2 (`5/5`).
- Unbounded D2 unlikelihood requires bounded/clamped suppression for G2 transfer (later phase), whereas D1 satisfies all G0 requirements cleanly.
- **Proceed to P1–P3:**
  1. Refine M3 attribute probe wording (`evidence.py`) and silicone heat-resistance rule (`knowledge_graph.py`) based strictly on V3-01 / V3-03 `dev` findings, and verify on `dev` (`V3-05a`).
  2. Run M2 Damage Discovery benchmark (`H1`) on `dev` (`V3-05b`).
  3. Run full `dev` (36 units) and `calibration` (36 units) for P2/P3 (`E0-E7` matrix, E5 unconstrained & E6 `orthogonal_retain_proj` ablations, and independent `DecomposedPredictor` calibration).

---

## Experiment V3-05a: M3 Probe Refinement Verification (`dev`)

### 1. Design
- **Goal:** Verify on `dev` that replacing the first-option-biased thermoregulation prompt, adding `smooth_skin` covering, fixing `origin` prompt parroting, and updating silicone `heat_behavior` improves structured candidate accuracy without sacrificing zero-false-label safety.
- **Command:** `python3 -u scripts/v3/run_mastery_survey.py --out results/v3-new/exp05a-m3-probe-check-dev`

### 2. Quantitative Results (`results/v3-new/exp05a-m3-probe-check-dev/summary.json`)
- **Clean Mastery:** **33 / 36 (91.67%)** (unchanged)
- **Clean Structured Reconstruction Accuracy:** **77.78%** (up from **63.89%** in V3-01, **+13.89 pp**)
  - `everyday_material`: **83.33%** (up from 66.67%)
  - `vertebrate_class`: **66.67%** (up from 41.67%)
  - `sound_source_mechanism`: **83.33%** (unchanged)
- **Unknown Control Mastery:** **0 / 8 (0.00%)**

---

## Experiment V3-05b: M2 Damage Discovery Benchmark (`dev36` + 8 unknown controls, Initial Run)

### 1. Design
- **Goal:** Evaluate H1 (knowledge-constraint discovery vs simple direct anomaly screening) across 33 mastered `dev` units (16 sequentially damaged with D1, 17 kept healthy) + 8 `unknown_control` units, at `probe_budget = 5 * 41 = 205`.
- **Command:** `python3 -u scripts/v3/run_p1_p3.py --mode m2_discovery --config configs/v3/p0_d1_steps64.json --out results/v3-new/exp05b-m2-discovery-dev36 --split dev --limit 36`

### 2. Quantitative Results (`results/v3-new/exp05b-m2-discovery-dev36/m2_discovery_metrics.json`)
- **`simple_direct_baseline`:** AUROC = `0.599`, AUPRC = `0.756`, Recall@Budget = `0.625`, Healthy FPR = `0.000`, Unknown FPR = `1.000`
- **`m2_reference_free`:** AUROC = `0.586`, AUPRC = `0.703`, Recall@Budget = `0.625`, Healthy FPR = `0.118`, Unknown FPR = `1.000`
- **`m2_reference_based`:** AUROC = **`1.000`**, AUPRC = **`1.000`**, Recall@Budget = **`1.000`**, Healthy FPR = `0.000`, Unknown FPR = `1.000`

### 3. Diagnosis & Self-Iteration Fix
1. **Why Reference-Free Failed to Beat Simple Baseline:** `constraint_probe_fraction=0.5` sorted units by Stage 1 instability (`entropy / disagree`), which excluded *confidently wrong* D1-damaged units (low Stage 1 entropy!) from Stage 2 `EvidenceGraph` probing while wasting Stage 2 budget on high-entropy `unknown_control` units.
2. **Why Unknown FPR = 1.0:** Stage 2 constraint mismatch unconditionally promoted `status = "suspected"` even when `reference_based` showed zero historical shift (`flipped=False`) or `reference_free` showed high-entropy / low-mass unknown uncertainty.
3. **Fix Applied (`V3-05c`):** Update `discovery.py` so Stage 2 skips low-mass/high-entropy unknown candidates, allocates Stage 2 `EvidenceGraph` budget (`constraint_probe_fraction=0.9`, budget `6*N`) across known-candidate units including low-entropy ones, and requires actual historical shift before promoting `reference_based` conflicts.

---

## Experiment V3-05c: M2 Damage Discovery Benchmark (`dev36` + 8 unknown controls, Fixed)

### 1. Design
- **Goal:** Re-test H1 under identical fixed `probe_budget = 6 * 41 = 246` with refined `discovery.py` (16 D1-damaged units, 17 healthy mastered units, 8 `unknown_control` units).
- **Command:** `python3 -u scripts/v3/run_p1_p3.py --mode m2_discovery --config configs/v3/p0_d1_steps64.json --out results/v3-new/exp05c-m2-discovery-dev36 --split dev --limit 36`

### 2. Quantitative Results (`results/v3-new/exp05c-m2-discovery-dev36/m2_discovery_metrics.json`)
| Method | AUROC | AUPRC | Recall@Budget | Healthy FPR | Unknown FPR | Mean Cost |
|---|---:|---:|---:|---:|---:|---:|
| `simple_direct_baseline` | 0.629 | 0.759 | 0.625 (10/16) | **0.000** | 0.875 | 3.00 |
| `m2_reference_free` (`EvidenceGraph`) | **0.813** | **0.884** | **0.813 (13/16)** | 0.059 | **0.000** | 4.10 |
| `m2_reference_based` | **1.000** | **1.000** | **1.000 (16/16)** | **0.000** | 0.875* | 4.10 |

*\*Note on `m2_reference_based` Unknown FPR:* Sequential injection of 16 D1 units mildly perturbed random tie-breaking on originally unmastered `unknown_control` units relative to clean baseline (`flipped=True`), whereas `m2_reference_free` properly classified all 8/8 `unknown_control` units as `"unknown"` (`0.000` FPR).

### 3. Diagnosis & Next Step
- **H1 Supported:** On `dev`, `m2_reference_free` improves AUROC by **+18.4 pp** (`0.629 -> 0.813`) and Recall@Budget by **+18.8 pp** (`0.625 -> 0.813`) over the fixed-budget simple anomaly baseline while eliminating false alarms on unknown controls (`0.875 -> 0.000`).
- **Next Step (`V3-06`):** Run `configs/v3/p2_dev12_orth_e6.json` on `dev12` to test **E5** (unconstrained LoRA ablation) and **E6** (`orthogonal_retain_proj` alternative writeback) alongside refined M3 probes.

---

## Experiment V3-06: E5 Unconstrained Ablation & E6 Orthogonal Gradient Projection (`dev12`)

### 1. Design
- **Goal:** Directly compare E6 (`orthogonal_retain_proj` + retention KL), E5 (`oracle_unconstrained`, no retention KL), E3 (`structured` with refined M3 probes), and E1 (`direct`, `generative`) on the same D1-damaged `dev12` cohort.
- **Config:** `configs/v3/p2_dev12_orth_e6.json`
- **Command:** `python3 -u -m krmm.v3.runner --config configs/v3/p2_dev12_orth_e6.json --out results/v3-new/exp06-e5-e6-dev12 --split dev --limit 12`

### 2. Quantitative Results (`results/v3-new/exp06-e5-e6-dev12/metrics.json`)
- **Clean Mastery:** **12 / 12 (100.0%)**; **Strict-Eligible:** **9 / 12 (75.0%)** (`2` ineffective damage, `1` damage locality fail)
- **Branch Comparison Over 9 Strict-Eligible Units:**
  | Branch / Condition | Strict Restorations | Independent Locality Pass | Wrong Pseudolabels |
  |---|---:|---:|---:|
  | **E4/E6: Oracle (`orthogonal_retain_proj`)** | **9 / 9 (100.0%)** | **9 / 9 (100.0%)** | 0 |
  | **E5: Oracle Unconstrained (no KL)** | 4 / 9 (44.4%) | 4 / 9 (44.4%) | 0 |
  | **E3: Structured (`EvidenceGraph`)** | **6 / 9 (66.7%; 6/6=100% when non-null)** | **6 / 6 (100.0%)** | **0** |
  | **E1: Generative** | 4 / 9 (44.4%) | 7 / 8 | 5 |
  | **E1: Direct Residual** | 2 / 9 (22.2%) | 9 / 9 | 7 |
- **Elapsed Time:** 1808.6s (30.1 min), Peak VRAM: 11.16 GiB

### 3. Diagnosis & Next Step
- **E5 Confirms H3 Retention Necessity:** Removing retention constraints cut strict restoration from **100% -> 44.4%**, causing non-target mean KL drift up to **2.86–7.64** (vs `0.0004–0.0090` with constrained writeback).
- **E3 Outperforms E1:** `structured` achieved **66.7% all-eligible** (**100% precision**, 0 wrong pseudolabels) vs `44.4%` (`generative`) and `22.2%` (`direct`).
- **Next Step (`V3-07` & `V3-08`):** Run full `dev` (36 units, `configs/v3/p2_dev36_full.json`) and full disjoint `calibration` (36 units, `configs/v3/p2_cal36_full.json`) to test M4 `DecomposedPredictor` calibration & E7 selector comparison against `candidate_change`.

---

## Experiment V3-07: Full `dev36` E0–E4 Matrix (`D1_wrong_association`, `damage_steps=64`)

### 1. Design
- **Goal:** Evaluate P0/G0 and E0–E4 across all 36 entity-disjoint `dev` units (12 per domain), using frozen `standard_kl` constrained LoRA writeback (`rank=8, alpha=16, last_layers=4, 32 steps, lr=5e-4, retain_weight=50.0`) and fresh-process reload verification.
- **Config:** `configs/v3/p2_dev36_full.json`
- **Command:** `python3 -u -m krmm.v3.runner --config configs/v3/p2_dev36_full.json --out results/v3-new/exp07-p2-dev36 --split dev --limit 36`

### 2. Quantitative Results (`results/v3-new/exp07-p2-dev36/e0_e7_analysis.json`)
- **Clean Mastery (A):** **33 / 36 (91.67%)** (`3` excluded on screening: `great white shark`, `orchestral triangle`, `crash cymbal`)
- **Damage Validity & Independent Locality (B):**
  - `damage_effective = 25 / 33 (75.76%)`
  - `damage_independent_locality_pass = 21 / 25` (`min_damage_train_retention = 0.9821`)
  - `strict_eligible = 21 / 33 (63.64%)`
- **G0 Gate Status:** **PASS** (`A=true, B=true, C=true`)
- **E0–E4 Branch Comparison Over 21 Strict-Eligible Units:**
  | Condition | Strict Restorations | Coverage | Locality Pass (Attempted) | Wrong Pseudolabels |
  |---|---:|---:|---:|---:|
  | **E0: Damaged Baseline (no writeback)** | 0 / 21 (heldout acc 33.3%) | — | — | — |
  | **E1: Direct Residual + Writeback** | 0 / 21 (0.0%) | 100.0% | 20 / 21 (95.2%) | 21 |
  | **E1: Generative + Writeback** | 12 / 21 (57.1%) | 100.0% | 19 / 21 (90.5%) | 9 |
  | **E3: Structured (`EvidenceGraph`) + Writeback** | **14 / 21 (66.7%)** | 71.4% (15/21) | **15 / 15 (100.0%)** | **1** |
  | **E4: Oracle Gold + Writeback** | **21 / 21 (100.0%)** | 100.0% | **21 / 21 (100.0%)** | 0 |
- **Structured Precision (when candidate produced):** **14 / 15 (93.33%)** (`15/15` tagged `structured_single_hop`, `0` tagged `true_composition` because singleton rules already fired on valid facts).
- **Entity-Clustered Paired Bootstrap 95% CIs (Strict Restoration):**
  - `structured` vs `direct`: **+66.7 pp**, 95% CI `[+47.6 pp, +85.7 pp]`
  - `structured` vs `generative`: **+9.5 pp**, 95% CI `[-14.3 pp, +38.1 pp]` (9x fewer wrong pseudolabels: 1 vs 9)
- **Elapsed Time:** 3979.1s (66.3 min), Peak VRAM: 11.16 GiB

### 3. Next Step (`V3-08`)
- Run identical frozen protocol on the 36 entity-disjoint `calibration` units (`results/v3-new/exp08-p2-cal36`) to independently fit and cross-evaluate M4 `DecomposedPredictor` vs `candidate_change` (E7).

---

## Experiment V3-08: Independent `calibration36` Run & E7 Cross-Split Repairability Evaluation

### 1. Design
- **Goal:** Run the exact frozen protocol on the 36 entity-disjoint `calibration` units (`configs/v3/p2_cal36_full.json`) and evaluate independent `DecomposedPredictor` calibration (`dev -> calibration` and `calibration -> dev`) against `candidate_change` and all 5 other baselines (E7).
- **Command:** `python3 -u -m krmm.v3.runner --config configs/v3/p2_cal36_full.json --out results/v3-new/exp08-p2-cal36 --split calibration --limit 36`

### 2. Quantitative Results (`results/v3-new/e0_e7_dev_cal.json`)
- **Clean Mastery (A):** **32 / 36 (88.89%)**
- **Damage Validity & Independent Locality (B):** `damage_effective = 24/32`, `strict_eligible = 16/32 (50.0%)`, `min_damage_train_retention = 0.9821`
- **Oracle Writeback (E4):** **14 / 16 (87.50%)** (`gate_g0.pass = true`)
- **E0–E4 Branch Comparison Over 16 Strict-Eligible `calibration` Units:**
  | Condition | Strict Restorations | Coverage | Locality Pass (Attempted) | Wrong Pseudolabels |
  |---|---:|---:|---:|---:|
  | **E0: Damaged Baseline** | 0 / 16 (heldout acc 33.3%) | — | — | — |
  | **E1: Direct Residual + Writeback** | 1 / 16 (6.25%) | 100.0% | 16 / 16 (100.0%) | 15 |
  | **E1: Generative + Writeback** | 8 / 16 (50.0%) | 100.0% | 15 / 16 (93.8%) | 6 |
  | **E3: Structured (`EvidenceGraph`) + Writeback** | **12 / 16 (75.0%)** | 93.8% (15/16) | 14 / 15 (93.3%) | **1** |
  | **E4: Oracle Gold + Writeback** | **14 / 16 (87.5%)** | 100.0% | 15 / 16 (93.8%) | 0 |
- **True Multi-Premise Composition:** 3 units (`2 / 3` correct); Single-hop: 12 units (`12 / 12` correct).
- **Entity-Clustered Paired 95% CIs on `calibration`:**
  - `structured` vs `generative` (strict restoration): **+25.0 pp**, 95% CI **[+6.25 pp, +50.0 pp]** (statistically > 0)
  - `structured` vs `direct` (strict restoration): **+68.75 pp**, 95% CI **[+43.75 pp, +87.5 pp]**
- **E7 Cross-Split Repairability Comparison (Disjoint Train -> Eval, No LOOCV):**
  - `fit dev -> eval calibration` (`structured`, n=15): `decomposed_ere` AUROC **`0.889`** / Brier `0.149` vs `candidate_change` AUROC `0.625` / Brier `0.200` (both 100% precision at 50% coverage)
  - `fit calibration -> eval dev` (`structured`, n=15): `decomposed_ere` AUROC **`1.000`** / Brier `0.084` vs `candidate_change` AUROC `0.500` / Brier `0.067`
  - `fit dev -> eval calibration` (`generative`, n=16): `decomposed_ere` AUROC **`0.875`** / Brier **`0.147`** / Selected@50% **`87.5%`** vs `candidate_change` AUROC `0.563` / Brier `0.438` / Selected@50% `75.0%`
  - `fit calibration -> eval dev` (`generative`, n=21): `decomposed_ere` AUROC **`0.769`** / Brier **`0.274`** / Selected@50% **`90.0%`** vs `candidate_change` AUROC `0.542` / Brier `0.429` / Selected@50% `60.0%` (**+30.0 pp** among selected)
- **Elapsed Time:** 3535.9s (58.9 min), Peak VRAM: 11.16 GiB

### 3. Progression to Sealed `test48` (`V3-09`)
- With `dev` + `calibration` completed and all model/probe/predictor hyperparameters frozen, unlock `configs/v3/p2_test48_frozen.json` (`allow_sealed_test=true`) for confirmatory evaluation on the 48 unseen entity-disjoint `test` units.

---

## Experiment V3-09: Confirmatory Sealed `test48` E0–E4 & E7 Evaluation

### 1. Design
- **Goal:** Run the exact frozen KRMM v0.3 protocol (`configs/v3/p2_test48_frozen.json`, zero threshold/probe changes after `calibration`) on all 48 unseen entity-disjoint `test` units (16 per domain: `vertebrate_class`, `everyday_material`, `sound_source_mechanism`), then fit `M4 DecomposedPredictor` on `dev+calibration` and evaluate out-of-sample on `test48`.
- **Command:** `python3 -u -m krmm.v3.runner --config configs/v3/p2_test48_frozen.json --out results/v3-new/exp09-p2-test48 --split test --limit 48`
- **Post-hoc E0–E7 Analysis:** `python3 scripts/v3/analyze_e0_e7.py --dev-dir results/v3-new/exp07-p2-dev36 --cal-dir results/v3-new/exp08-p2-cal36 --test-dir results/v3-new/exp09-p2-test48 --out results/v3-new/e0_e7_full_final.json`

### 2. Quantitative Results (`results/v3-new/e0_e7_full_final.json`)
- **Clean Mastery (A):** **43 / 48 (89.58%)** (`5` excluded during 4-phrasing × 2-permutation mastery screening, e.g. `castanets`, `tam-tam gong`).
- **Damage Validity & Randomization Audit (`damage-randomization-audit.json`):**
  - Assigned wrong options balanced (`A:11, B:12, C:12, D:13`), `best_in_sample_inverse_accuracy = 0.4583`, `cyclic_inverse_accuracy = 0.375`, `no_deterministic_inverse = true`.
  - `damage_effective = 33 / 43 (76.74%)` (`10` excluded for insufficient damage effect or train-retain failure).
  - `damage_independent_locality_pass = 29 / 33` (`4` excluded on independent 4-category locality audit after damage); **`strict_eligible = 29 / 43 (67.44%)`**, `min_damage_train_retention = 0.9821`.
- **G0 Gate Status on Sealed `test48`:** **PASS** (`A=true, B=true, C=true`; oracle `27/29 = 93.10%` strict restorations, `28/29 = 96.55%` operational success).
- **E0–E4 Branch Comparison Over 29 Strict-Eligible `test48` Units:**
  | Condition | Strict Restorations | Coverage | Locality Pass (Attempted) | Wrong Pseudolabels Written |
  |---|---:|---:|---:|---:|
  | **E0: Damaged Baseline** | 0 / 29 (heldout acc 33.33%) | — | — | — |
  | **E1: Direct Residual + Writeback** | 0 / 29 (0.0%) | 100.0% (29/29) | 24 / 29 (82.76%) | 28 |
  | **E1: Generative + Writeback** | 13 / 29 (44.83%) | 100.0% (29/29) | 26 / 29 (89.66%) | 16 |
  | **E3: Structured (`EvidenceGraph`) + Writeback** | **20 / 29 (68.97%)** | 82.76% (24/29) | 22 / 24 (91.67%) | **4** |
  | **E4: Oracle Gold + Writeback** | **27 / 29 (93.10%)** | 100.0% (29/29) | 28 / 29 (96.55%) | 0 |
- **Structured Breakdown:** Precision when non-null = **20 / 24 (83.33%)**; `structured_single_hop = 17 / 19 (89.47%)`; `structured_composition = 3 / 5 (60.00%)`. Across all 3 splits (`dev+cal+test`), true multi-premise composition fired on `8` strict-eligible units (`5/8 = 62.5%` correct) and single-hop fired on `46` (`43/46 = 93.48%` precision), with `12` clean abstentions (never writing wrong pseudolabels).
- **Entity-Clustered Paired Bootstrap 95% CIs on Sealed `test48` (`n=29`):**
  - `structured` vs `generative` (strict restoration & candidate accuracy): **+24.14 pp**, 95% CI **`[+6.90 pp, +41.38 pp]`** (strictly > 0, replicating H3 out-of-sample).
  - `structured` vs `direct` (strict restoration): **+68.97 pp**, 95% CI **`[+51.72 pp, +86.21 pp]`**.
- **Fresh-Process Persistence Verification:** **142 / 142 (100%)** strict restorations across `dev36` (47/47), `cal36` (35/35), and `test48` (60/60) passed fresh subprocess reload from disk.
- **Confirmatory E7 Repairability Comparison (Fit on `dev+cal`, Eval on Sealed `test48`):**
  - **Structured Branch (`train_n=30`, `test n=24`, positives=20):**
    | Method | AUROC | AUPRC | Brier ↓ | Selected@50% (12/24) |
    |---|---:|---:|---:|---:|
    | `random` | 0.4875 | 0.7967 | 0.3879 | 83.33% |
    | `entropy` | 0.2750 | 0.7458 | 0.6983 | 83.33% |
    | `self_consistency` | 0.3750 | 0.8333 | 0.8438 | 66.67% |
    | `candidate_change` | 0.5000 | 0.8333 | 0.1667 | 75.00% |
    | `support_completeness` | 0.5000 | 0.8333 | 0.1667 | 75.00% |
    | `direct_confidence` | 0.7125 | 0.9329 | 0.1286 | 83.33% |
    | **`decomposed_ere` (Ours)** | **0.8125** | **0.9592** | **0.1120** | **100.00% (12/12)** |
  - **Generative Branch (`train_n=37`, `test n=29`, positives=13):**
    | Method | AUROC | AUPRC | Brier ↓ | Selected@50% (14/29) |
    |---|---:|---:|---:|---:|
    | `random` | 0.5288 | 0.4460 | 0.3519 | 50.00% |
    | `entropy` | 0.2981 | 0.3910 | 0.4356 | 21.43% |
    | `self_consistency` | 0.4375 | 0.4483 | 0.4655 | 7.14% |
    | `candidate_change` | 0.5865 | 0.4960 | 0.4483 | 35.71% |
    | `support_completeness` | 0.5865 | 0.4960 | 0.4483 | 35.71% |
    | `direct_confidence` | 0.7019 | 0.7182 | 0.4752 | 64.29% |
    | **`decomposed_ere` (Ours)** | **0.9327** | **0.9320** | **0.1224** | **78.57%** |
    - Paired Selected@50% CI (`decomposed_ere` vs `candidate_change` on generative): **+20.69 pp**, 95% CI **`[+6.90 pp, +37.93 pp]`** (strictly > 0).
- **Failure Taxonomy on `dev+cal+test`:**
  - *Oracle writeback failures (`2` on `cal36`, `2` on `test48`):* Two (`concert bass drum`, `bodhran`) stopped at 66.7% heldout accuracy within 24 LoRA steps (`under-written`); two (`timpani`, `crystal wine glass`) hit 100% target heldout accuracy (`operational_success=true`) yet were explicitly caught and rejected by the independent `same_entity_other_relation` audit (`mean_kl=2.59` / `0.19`, accuracy drop `1.0 -> 0.0`) while all 125/161 other locality controls stayed clean — empirically validating why single-pool KL retain checks are insufficient.
  - *Structured reconstruction failures (`1` wrong on `dev`, `3` on `cal`, `4` on `test` + `1` oracle locality fail):* Wrong pseudolabels stemmed from damaged-model internal premise errors (e.g., classifying `xylophone` oscillator as `stretched_membrane` instead of `solid_body`, or `brass doorknob` / `titanium orthopedic screw` conductivity as `insulator`). Clean abstention prevented writes on `12` units (`6` dev, `1` cal, `5` test).
- **Elapsed Time:** 5446.4s (90.8 min), Peak VRAM: 11.16 GiB, Decoder Tokens: 16.07M.

---

## Experiment V3-10: Out-of-Sample M2 Damage Discovery on `calibration` (+ `unknown_control`)

### 1. Design
- **Goal:** Verify H1 (`m2_reference_free` vs `simple_direct_baseline` under bounded query budget and unknown-control separation) out-of-sample on `calibration` (`36` units + `8` `unknown_control`).
- **Command:** `python3 -u scripts/v3/run_p1_p3.py --mode m2_discovery --config configs/v3/p2_test48_frozen.json --out results/v3-new/exp10-m2-discovery-cal36 --split calibration --limit 36`

### 2. Quantitative Results (`results/v3-new/exp10-m2-discovery-cal36/m2_discovery_metrics.json`)
- **Sample Partition:** `n_mastered=32` (`n_damaged=16`, `n_healthy=16`), `n_unknown_ctrl=8` (`n_total=40`).
- **Comparison:**
  | Detector | AUROC | AUPRC | Recall@Budget | Precision@Budget | Healthy FPR | Unknown FPR | Mean Query Cost |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | `simple_direct_baseline` | 0.7656 | 0.8431 | 0.6875 | 0.6875 | 0.0000 | 0.6250 | 3.000 |
  | **`m2_reference_free`** | **0.9902** | **0.9846** | **0.9375** | **0.9375** | 0.0625 | **0.0000** | 4.425 |
  | `m2_reference_based` (ref upper bound) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.7500* | 4.425 |
  - \*Note: `m2_reference_based` compares current accuracy against a threshold without checking whether clean accuracy already started near chance on `unknown_control`, whereas `m2_reference_free` explicitly tests constraint inconsistency vs ignorance and suppresses `unknown_control` false alarms (`Unknown FPR = 0.000`).
- **Resources:** Peak VRAM 11.16 GiB, `query_calls = 842`, `tokens = 64,225`.

---

## Experiment V3-11: Confirmatory Sealed `test48` M2 Damage Discovery

### 1. Design
- **Goal:** Verify H1 out-of-sample on the 48 unseen entity-disjoint `test` units (`43` mastered -> `22` damaged, `21` healthy) mixed with `8` never-known `unknown_control` units (`51` total).
- **Command:** `python3 -u scripts/v3/run_p1_p3.py --mode m2_discovery --config configs/v3/p2_test48_frozen.json --out results/v3-new/exp11-m2-discovery-test48 --split test --limit 48`

### 2. Quantitative Results (`results/v3-new/exp11-m2-discovery-test48/m2_discovery_metrics.json`)
- **Sample Partition:** `n_mastered=43` (`n_damaged=22`, `n_healthy=21`), `n_unknown_ctrl=8` (`n_total=51`).
- **Comparison:**
  | Detector | AUROC | AUPRC | Recall@Budget | Precision@Budget | Healthy FPR | Unknown FPR | Mean Query Cost |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | `simple_direct_baseline` | 0.8182 | 0.7410 | 0.8182 | 0.8182 | 0.0952 | 0.8750 | 3.000 |
  | **`m2_reference_free`** | **0.9177** | **0.8645** | **0.9091** | **0.9091** | 0.1905 | **0.0000** | 4.569 |
  | `m2_reference_based` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0952 | 0.8750 | 4.569 |
- **Resources:** Peak VRAM 11.16 GiB, `query_calls = 1,107`, `tokens = 84,229`.

---

## Final Synthesis Across KRMM v0.3 (`V3-01` – `V3-11`)

1. **P0 / Gate G0 (Valid Mastery, Effective Local Damage, Supervised Writeback Upper Bound):** **PASS** across `dev36` (`91.7%` mastery, `21/33` strict-eligible, `21/21 = 100%` oracle strict), `cal36` (`88.9%` mastery, `16/32` strict-eligible, `14/16 = 87.5%` oracle strict), and sealed `test48` (`89.6%` mastery, `29/43` strict-eligible, `27/29 = 93.1%` oracle strict). Zero inverse-mapping shortcut (`best_in_sample = 0.458` vs chance `0.25`, `no_deterministic_inverse = true`). D2 (`D2_response_suppression`) without explicit KL constraint remains fragile (`exp04`: `5/12` effective, `3/12` strict-eligible, `3/3` oracle strict).
2. **H1 (`M2` Discovery under Fixed Query Budget):** **SUPPORTED** consistently across all 3 splits (`dev`: AUROC `0.8125 vs 0.6287`; `cal`: AUROC `0.9902 vs 0.7656`; sealed `test`: AUROC `0.9177 vs 0.8182`), while completely eliminating false alarms on never-known `unknown_control` units (`Unknown FPR = 0.000` vs `0.625–0.875` for `simple_direct_baseline`).
3. **H2 (`M4` Decomposed ERE Repairability Prediction):** **SUPPORTED** out-of-sample across cross-split (`dev <-> cal`) and confirmatory (`dev+cal -> test48`) evaluations: `decomposed_ere` outperforms `candidate_change`, `entropy`, `self_consistency`, and `direct_confidence` (`test48` structured AUROC `0.8125`, AUPRC `0.9592`, Brier `0.1120`, Selected@50% **`100.0%`**; `test48` generative AUROC `0.9327`, Brier `0.1224`, Selected@50% **`78.57%` vs `35.71%`**, paired 95% CI **`+20.69 pp [+6.90, +37.93]`**).
4. **H3 (`M3` Structured Evidence Reconstruction + `M5` Constrained Writeback):** **SUPPORTED** on both `cal36` (`+25.0 pp [+6.25, +50.0]`) and sealed `test48` (`structured` `20/29 = 68.97%` vs `generative` `13/29 = 44.83%`, paired 95% CI **`+24.14 pp [+6.90, +41.38]`**; vs `direct` `0/29 = 0.0%`, **`+68.97 pp [+51.72, +86.21]`**), cutting wrong pseudolabel writes on `test48` from `28` (`direct`) and `16` (`generative`) down to `4` (`structured`), with `142/142 (100%)` strict restorations passing isolated fresh-process reload verification.











