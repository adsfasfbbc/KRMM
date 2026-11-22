# KRMM v0.3 Falsifiable Hypotheses

- **H1 (Damage Discovery):** Under a fixed query budget, detection combining knowledge constraint graphs and alternative probing paths outperforms simple direct anomaly screening (higher AUROC/AUPRC and Recall@Budget without inflating false positives on originally unknown units).
- **H2 (Decomposed Repairability):** Separating candidate reliability from parameter write-back risk predicts strict repair success more accurately (higher AUROC/AUPRC, lower Brier score, and higher strict restoration rate at matched coverage) than `candidate_change` and single-signal baselines.
- **H3 (Constrained Local Repair):** For damaged knowledge with sufficient retained evidence and feasible write-back conditions, constrained local parameter updates achieve persistent functional restoration on unseen phrasings and permuted options while keeping non-target independent audit degradation within frozen tolerances.

Any hypothesis failing confirmatory evaluation on the sealed test split must be retained as a negative result without retroactive threshold tuning or test-set overfitting.
