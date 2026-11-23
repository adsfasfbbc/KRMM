# KRMM v0.5 Hypotheses & Verification Ledger

## H1: Reference-Free Blind Damage Discovery
- **Claim:** Under equal query budgets (`2N–5N`) without knowing the injection list, gold answers, or clean weights, constraint-conflict and access-path discrepancy signals outperform confidence, entropy, self-consistency, and paraphrase-only baselines on brand-new entities.
- **Status:** **SUPPORTED (with explicit trade-off documented).** At `5N` (`B=225`, `N=45`), `m2_reference_free` reaches AUROC `0.789` / AUPRC `0.732` / 3-class acc `0.667` / `unknown_fpr=0.167`, and `full_m2` reaches AUROC `0.860` / AUPRC `0.762` / `collateral_damage_recall=1.000` (3/3), versus `direct_confidence` (`0.166`), `entropy` (`0.148`), `self_consistency` (`0.435`), `direct_paraphrases` (`0.300`), and `constraint_only` (`0.345`). However, `full_m2` incurs higher `unknown_fpr` (`0.583` vs `0.167` for `m2_reference_free`).

## H2: Support Structure Predicts Candidate Reliability
- **Claim:** Structured evidence support classes predict candidate accuracy better than simple confidence, entity-name heuristics, generic CoT, or un-gated generative reconstruction, and model predictions respond causally to key-support removal/flip while remaining invariant to sham attributes.
- **Status:** **SUPPORTED.** On `n=22` brand-new damaged entities (`G2-A`), accuracy stratifies cleanly: `single_decisive` `6/6 (100.0%)`, `redundant_distinct_attributes` `9/10 (90.0%)`, `single_chain` `2/4 (50.0%)`, `conflicting_support` `0/2 (0.0%)`. Evidence gating accepts `16/22` with `93.75%` precision (`15/16`), outperforming `direct_probing` (`4.5%`), `self_consistency` (`4.5%`), `generic_cot` (`9.1%`), and `generative_reconstruction` (`50.0%`). Real model-level interventions show `100.0%` sham invariance (`|ΔP|=0.045`) vs `-0.145` key-removal drop and `-0.256` counterfactual-flip drop.

## H3: Evidence-Gated Selective Repair Reduces Wrong Writes Without Destroying Coverage
- **Claim:** Gating parametric LoRA writeback (`standard_kl`) by support structure dramatically reduces erroneous pseudo-label writes while preserving locality and achieving competitive strict restoration.
- **Status:** **SUPPORTED (with explicit coverage-vs-safety trade-off).** On `n=13` strict-eligible brand-new targets (`G2-B`), `Direct` restores `0/13` (`13` wrong writes), `Generative` `8/13` (`5` wrong writes), `Structured` `11/13` (`1` wrong write), `Evidence-Gated` `9/13` (**`0` wrong writes, `9/9 = 100.0%` precision on attempted writes**), `Oracle` `13/13`. All passing adapters pass serialized PEFT reload and `100.0%` same-entity relation preservation.

## H4: Generalization & Capacity Boundaries (G3)
- **Claim:** Evaluate whether the mechanism transfers to bounded `D2_suppression` and a second model architecture.
- **Status:** **BOUNDARY IDENTIFIED (see `G3_GENERALIZATION_REPORT.md`).** Bounded `D2` (`clamp=2.0`, `retain_weight=50.0`) yields `0/15` effective damage (sharp phase transition vs v0.3 unbounded collapse); `SmolLM2-1.7B-Instruct` exhibits a 1.7B capacity floor (`2/18` clean mastery, `0/2` Oracle locality pass), though `Evidence-Gated` still safely abstains (`0/2` wrong writes). Cross-architecture 3B results (`Llama-3.2-3B-Instruct`) are appended upon G3-B completion.
