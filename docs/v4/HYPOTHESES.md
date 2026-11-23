# KRMM v0.4 Research Hypotheses (`H1`–`H4`)

- **H1 (Reference-Free Blind Damage Discovery):** Without access to injection lists, gold answers, or clean weights, constraint conflict + path inconsistency improves detection of real behavioral degradation over equal-budget `direct_confidence`, `self_consistency`, `direct_paraphrases`, and `constraint_conflict_only` on a full mixed pool (including unseen unknown controls).
  - *Falsification:* All gain disappears at equal query cost, under domain-stratified randomization, or under full-pool ranking with unseen unknown controls.
  - *Current Status:* **PENDING_GPU_VALIDATION** (G1).

- **H2 (Evidence Structure Governs Candidate Reliability):** Minimal support-set structure (`single_decisive`, `single_chain`, `redundant_same_source`, `redundant_distinct_paths`, `independent_source_support`, `conflicting_support`, `insufficient_support`) predicts candidate correctness and reduces wrong-writes beyond simple confidence, fact count, or rule-only shortcuts.
  - *Falsification:* Raw model confidence, fact count, entity-only, or rule-only baselines explain all reliability gains, or correlated wrong consensus is over-trusted.
  - *Current Status:* **EXPLORATORY_SIGNAL_ONLY** (`27/27` vs `1/4` retrospective on same-model v0.3 archive; fresh confirmatory G2 is **PENDING_GPU_VALIDATION**).

- **H3 (Evidence-Gated Selective Safe Repair):** Under fixed `standard_kl` writeback and equal budget, evidence-gated selective repair achieves higher `End-to-End Strict Safe Repair Yield` (or strictly lower `Wrong-Write Rate` and `Healthy-Unit Harm Rate` at matched yield) than blind writeback or naive confidence selection, passing all 5 side-effect gates (`Target Restored`, `Same-Entity Preserved`, `Neighbor Preserved`, `General Preserved`, `Fresh Reload Passed`).
  - *Falsification:* Gain comes solely from excessive abstention (high opportunity cost) or fails on unseen paraphrases / same-entity locality.
  - *Current Status:* **PARTIALLY_SUPPORTED_OFFLINE_ONLY** (v0.3 `D1 + standard_kl` non-blind `20/29` vs `13/29`; full end-to-end blind pipeline is **PENDING_GPU_VALIDATION**).

- **H4 (Mechanism Transferability):** Evidence-structure reliability signals and selective repair trade-offs generalize beyond a single D1 corruption, a single 3B model (`Qwen2.5-Omni-3B`), and 4-way classification templates to bounded `D2_suppression`, a second model family (e.g. Gemma 3B/4B), and optionally real multimodal evidence (ACM MM extension).
  - *Falsification:* Patterns vanish when changing damage mechanism, model family, or task structure.
  - *Current Status:* **NOT_STARTED** (G3 / P4–P5).
