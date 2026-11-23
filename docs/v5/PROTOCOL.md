# KRMM v0.5 Experimental Protocol (`Blind Knowledge Repair`)

## 1. Brand-New Disjoint Entity Pool (`results/v5-new/v5_blind_pool_manifest.json`)
To prevent any test-set overfitting or leakage from v0.3 (`140` entities), v0.5 constructs a **100% disjoint pool of 52 brand-new entities** (`v3_overlap_count = 0`, verified by `test_v5_catalog_100_percent_disjoint_from_v3`):
- **36 known entities** across 3 balanced domains (`12 vertebrate_class`, `12 everyday_material`, `12 sound_source_mechanism`), each with 2 same-entity invariant controls and 1 expected-ripple consistency probe.
- **12 originally-unknown controls** (`4` fictional/unseen entities per domain).
- **4 ambiguous / unverifiable controls**.

## 2. Information Isolation (Blindness Contract)
- **Discovery (`G1`)** receives ONLY public `KnowledgeUnit` objects (`entity_id`, `domain`, `options`) and a `CostTrackedProxy` enforcing exact query budgets `B ∈ {2N, 3N, 4N, 5N}`. It has zero access to the injection list, gold answers, or clean weights.
- **Reconstruction & Gating (`G2-A`, `G2-B`)** receives ONLY the damaged model and public unit schema; gold answers (`gold_evaluator_only`) are used exclusively by the post-hoc evaluator.
- **Balanced Wrong-Answer Injection (`D1`)** enforces `no_deterministic_inverse = True` (audited in both G1 and G2 JSON artifacts: `cyclic_inverse_accuracy = 0.333`).

## 3. Three-Stage Independent & End-to-End Evaluation
1. **Batch 1 — G1 (Equal-Budget Blind Discovery) + G2-A (Support Structure, Strong Baselines, and Real Model-Level Causal Interventions A–E).**
2. **Batch 2 — G2-B (Paired 5-Strategy Selective Repair Matrix: `Direct`, `Generative`, `Structured`, `Evidence-Gated`, `Oracle`) + G2-C (5-Gate Locality Graph + Serialized PEFT Adapter Reload).**
3. **Batch 3 — G3 (Generalization & Stress Boundaries: G3-A Bounded `D2_suppression` + G3-B Cross-Architecture Transfer on `SmolLM2-1.7B-Instruct` and `Llama-3.2-3B-Instruct`).**
