# KRMM v0.3 Next-Stage GPU Experimental Protocol (`PENDING_GPU_VALIDATION`)

**Status:** Protocol design only — **NO GPU runs authorized or executed in this CPU audit cycle.**  
**Code Baseline:** Branch `codex/krmm-v03-cpu-audit` (all P0–P4 fixes verified by 64/64 CPU tests).

---

## 1. Highest-Information-Gain GPU Experiments (Strictly Limited to 3 Runs)

### Experiment G1: Fair Equal-Budget & Full-Pool M2 Discovery (Tests & Can Falsify H1)
- **Research Hypothesis:** Under **strictly equal query budgets** (`B ∈ {2N, 3N, 4N, 5N}`), `m2_reference_free` outperforms `simple_direct_baseline` on **full-pool AUROC / AUPRC / Recall@K** (ranking damaged, healthy, and unseen unknown controls together) evaluated against **post-injection behavioral degradation truth**.
- **Falsification Criterion:** If `m2_reference_free` fails to beat `simple_direct_baseline` on full-pool AUROC or Recall@K at matched query cost (`3N` and `4N`), or if `unknown_in_topk_rate > 0.25` on fresh unknown controls, **H1 is falsified**.
- **Data & Splits:** New entity-disjoint split (`n_known >= 48` across 3 domains) + **fresh, non-overlapping** `unknown_control` pools (`>= 12` per split, never reused across dev/cal/test).
- **Randomization:** `select_stratified_damage_targets(..., damage_fraction=0.5, seed=frozen_seed)` with SHA256 manifest logged before injection.
- **Ground Truth:** `build_post_injection_truth_table` on final post-injection model (`behaviorally_degraded`, reporting `collateral_damage` rate).
- **Logging:** Export complete `per_unit_records` and `BudgetedQueryProxy` query logs for exact offline recomputation.

### Experiment G2: Hard Write-Risk Regime for True H2 Decomposition Validation (Tests & Can Falsify H2)
- **Research Hypothesis:** In a writeback regime with non-trivial conditional writeback failure rate (`15%–40%` failure among correct candidates, e.g., reduced `repair_steps ∈ {8, 12}` or multi-unit sequential writeback or comparing `standard_kl` vs `orthogonal_retain_proj`), the corrected `DecomposedPredictor` (`P(cand|x) * P(strict|cand=1, x)`) beats `DirectSuccessPredictor` (`P(strict|x)`) and `CandidateReliabilityPredictor` (`P(cand|x)`) on out-of-sample AUROC, Brier (after independent calibration on disjoint `cal`), and Selected@50% strict restoration.
- **Falsification Criterion:** If `DecomposedPredictor` fails to outperform `DirectSuccessPredictor` (paired 95% CI includes 0 or negative) once conditional writeback negatives reach `>= 15` in `dev+cal`, **H2's decomposition claim is falsified** (direct single-stage prediction suffices).
- **Data & Calibration Protocol:** Strictly 3-way disjoint split: `train` (fit) -> `cal` (independent Platt calibration ONLY) -> `fresh_test` (sealed confirmatory evaluation, opened once).

### Experiment G3: Writeback Strategy (`orthogonal_retain_proj` vs `standard_kl`) & Bounded D2 Generalization (Tests & Can Falsify H3 Mechanism Scope)
- **Research Hypothesis:** (a) `orthogonal_retain_proj` matches or improves strict restoration and `same_entity_other_relation` locality over `standard_kl` on a full `n >= 48` entity-disjoint cohort; (b) bounded/margin `D2_suppression` (`clamp(target_loss, max=tau)`) achieves `strict_eligibility_rate >= 0.50` while preserving `oracle_strict >= 0.75` and `structured > generative`.
- **Falsification Criterion:** If `orthogonal_retain_proj` shows no locality or restoration gain over `standard_kl` at `n >= 48`, or if bounded D2 still fails G0 (`strict_eligible < 50%` or `structured <= generative`), **narrow H3 strictly to `D1_wrong_association` + `standard_kl`**.

---

## 2. Resource Caps & Stop Rules
- **Max GPU Launches:** `<= 6` total (2 per experiment: 1 pilot/cal + 1 sealed confirmatory).
- **Stop Rule 1 (G0 Gate):** Stop immediately if clean mastery `< 75%` or strict-eligible `< 50%` on pilot before opening any sealed test split.
- **Stop Rule 2 (Collateral Drift):** In G1 multi-unit sequential injection, stop and reduce per-unit steps if `collateral_damage > 15%` of healthy units.
- **Stop Rule 3 (Sparsity Gate):** In G2, do not claim H2 decomposition validation if `train_conditional_write_neg < 10` (report `insufficient_data` honestly).
- **No Post-Hoc Tuning:** Zero threshold, rule, or hyperparameter edits after opening any sealed test split.
