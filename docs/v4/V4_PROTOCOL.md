# KRMM v0.4 Pre-Registered Experimental Protocol & Bounded GPU Plan

**Status:** `PLAN + CPU VALIDATION COMPLETE` — **Awaiting explicit GPU authorization before launching any model run.**

---

## 1. Hypotheses & Conditions
- **H1–H4:** Defined in `docs/v4/HYPOTHESES.md`.
- **Two Strictly Separated Study Conditions:**
  1. `internal_only`: Damaged model + public constraints only (zero entity-specific external facts).
  2. `evidence_augmented`: Adds external verified text/image/audio facts (never directly giving gold class labels), evaluated and reported separately.

## 2. Data Construction & Strict Splits
- **Historical `test48`:** Sealed for v0.3 historical audit ONLY — **forbidden** as confirmatory test for v0.4 modifications.
- **New Disjoint Mixed Pool (`n >= 60` per split: `dev_v4`, `cal_v4`, `test_v4`):**
  - Entity, `source_group`, and relation-disjoint across all splits.
  - Five behavioral ground-truth pool categories audited post-injection: `damaged_mastered`, `healthy_mastered`, `originally_unknown` (**fresh disjoint unknown controls per split, `>= 12` each**), `collateral_damaged`, `ambiguous_or_unverifiable`.
- **Damage Injection:** Domain- and class-stratified random assignment (`stratified_damage_split`) with fixed seeds `[2026, 2027, 2028]`, randomized injection order, SHA256 manifest, and post-injection behavioral re-audit of all units.

## 3. Methods, Baselines & Primary Metrics
- **M2 Equal-Budget Discovery Comparators (`B ∈ {2N, 3N, 4N, 5N}`):** `direct_confidence`, `self_consistency`, `direct_paraphrases`, `constraint_conflict_only`, `full_m2`.
  - Metrics: Full-pool AUPRC/AUROC, Recall@K at equal actual query budget, `healthy_fpr`, `originally_unknown_fpr`, `cost_per_true_damage_found`, generated token count.
- **Shortcut & Ablation Controls (P2):** `rule_only`, `entity_only`, `generic_decomposition`, `fixed_budget_self_consistency`, `shuffled_facts`, `full_evidence_gated` (+ anonymized-entity synthetic control).
- **Writeback Strategy (P3):** Primary fixed at `standard_kl` (`repair_steps=32, retain_weight=50.0`), paired ablations against `orthogonal_retain_proj`, `unconstrained_lora`, `oracle + standard_kl`, and `nonparametric_memory`.
- **Primary End-to-End Metrics:** `End-to-End Strict Safe Repair Yield`, `Wrong-Write Rate`, `Healthy-Unit Harm Rate`, `Candidate Coverage`, `Repair Attempt Rate`, `Abstention Rate`, `Strict Success Among Attempted`, `Opportunity Cost (Missed Repairable)`, 5-gate `SideEffectReport`, paired cluster-bootstrap 95% CIs (`n_boot=2000`, `source_group` clusters).

## 4. Gates, Falsification & Pivot Rules
- **G0 (Mechanism Verifiability — CPU DONE):** 27/27 & 1/4 re-audited; minimal support sets + correlated-wrong-support + boundary tests **23/23 PASSED**.
- **G1 (Blind Discovery):** On fresh mixed pool at equal budget `3N`/`4N`, `full_m2` must beat all 4 baselines on full-pool AUPRC and keep `originally_unknown_fpr <= 0.15`. *Pivot:* If G1 fails, freeze H1 claim and proceed to G2 with the strongest simple detector.
- **G2 (Evidence Structure & Selective Repair):** On fresh `test_v4`, `EvidenceReliabilityModel` (calibrated independently on `cal_v4`) must beat `direct_confidence` and `entity_only`/`shuffled_facts` shortcuts on AUROC/Brier, and `selective_policy` must improve `End-to-End Yield` or cut `Wrong-Write Rate` at matched yield with `Healthy-Unit Harm Rate <= 0.05`.
- **G3 (Generalization):** Replicate core matrix on bounded `D2_suppression` and a second 3B–4B model family (e.g., Gemma) passing G0 first; optionally run P5 ACM MM multimodal extension (audio/image + `shuffled_image`/`unrelated_audio` controls) if targeting ACM MM.

---

## 5. Bounded GPU Authorization Request Block (Section 11 — Awaiting Explicit Approval)

```text
approved_commit: <populated after local v0.4 plan commit on codex/krmm-v04-plan-cpu>
allowed_commands:
  - python3 scripts/v4/run_blind_discovery.py --mode gpu --authorized-gpu-plan configs/v4/g1_g2_gpu_plan.json
  - python3 scripts/v4/run_repair_matrix.py --mode gpu --authorized-gpu-plan configs/v4/g1_g2_gpu_plan.json
allowed_edit_paths:
  - configs/v4/
  - results/v4-new/
  - docs/v4/
gpu_device: cuda:0 (NVIDIA GeForce RTX 3090 24GB)
max_launches: 6
max_total_runtime: 180 minutes
max_disk_usage: 10 GiB (current free: 14 GiB on /root/rivermind-data)
stop_conditions:
  - clean_mastery_rate < 0.75 on pilot split
  - strict_eligibility_rate < 0.50 on pilot split
  - collateral_damage_rate > 0.15 in multi-unit sequential injection
  - any protected v0.3 artifact SHA256 mismatch
```
