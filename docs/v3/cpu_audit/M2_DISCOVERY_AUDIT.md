# KRMM v0.3 M2 Damage Discovery Audit (P2)

**Historical Artifact Check (`reports/v3-cpu-audit/recomputed_metrics.json`):**
- `results/v3-new/exp05c-m2-discovery-dev36/`: contains only `["m2_discovery_metrics.json"]`
- `results/v3-new/exp10-m2-discovery-cal36/`: contains only `["m2_discovery_metrics.json"]`
- `results/v3-new/exp11-m2-discovery-test48/`: contains only `["m2_discovery_metrics.json"]`
- **Recomputability Verdict:** **`NOT_RECOMPUTABLE_FROM_AVAILABLE_ARTIFACTS`** for full-pool AUROC/Recall@K, behavioral post-injection ground truth, and equal-budget cost curves. Historical conditional AUROC numbers (`0.8125`, `0.9902`, `0.9177`) are preserved untouched with explicit caveats below; zero synthetic historical scores were fabricated.

---

## 1. Four Methodological Flaws Identified & Fixed in Code

1. **Prefix-Slice Target Assignment (`krmm/v3/pipeline_p1_p3.py`):**
   - *Historical bug:* `dmg_targets = mastered_evs[:n_dmg]` sliced the first half of mastered units in catalog order, risking domain/category order bias.
   - *Fix:* Implemented `select_stratified_damage_targets(mastered_evs, damage_fraction, seed, stratify_by_gold)` which randomly samples balanced damaged/healthy targets within each domain, shuffles injection order, enforces strict entity/unit disjointness, and emits a SHA256-hashed `split_manifest`. Verified by `tests/v3_cpu/test_damage_randomization.py`.
2. **Planned Injection Conflated with Actual Behavioral Degradation (`krmm/v3/pipeline_p1_p3.py`):**
   - *Historical bug:* Sequential multi-unit LoRA injection used the planned target list as detection ground truth without re-auditing the final post-injection model for failed injections or collateral damage to healthy units.
   - *Fix:* Implemented `classify_post_injection_truth` and `build_post_injection_truth_table` distinguishing `planned_damaged`, `injection_completed`, `behaviorally_degraded`, `strict_local_damage`, and `collateral_damage`, and updated `run_m2_discovery_eval` to re-audit all mastered units post-injection and export `per_unit_records`. Verified by `tests/v3_cpu/test_post_injection_truth_schema.py`.
3. **Unequal Query Cost Comparison (`krmm/v3/discovery.py`):**
   - *Historical bug:* `simple_direct_baseline` spent `3.00` queries/unit while `m2_reference_free` spent `4.10` (`dev`), `4.43` (`cal`), and `4.57` (`test`) queries/unit; sharing a ceiling budget did not equalize actual query cost.
   - *Fix:* Implemented `BudgetedQueryProxy` and `evaluate_detection_at_budget(detector, query_budget, units, cached_responses=None)` enforcing immediate stop upon budget exhaustion and tracking `total_queries` and `per_unit_queries`. Verified by `tests/v3_cpu/test_detection_budget_accounting.py`.
4. **Excluded Unknown Controls in Ranking + Reused Unknown Control Pool (`krmm/v3/metrics.py`):**
   - *Historical bug:* `detection_metrics` excluded `is_unknown_ctrl == 1` when computing AUROC, AUPRC, and Recall@K, and all three runs (`dev`, `cal`, `test`) reused the exact same 8 `unknown_control` units—meaning `Unknown FPR = 0.000` across 3 splits was **repeated observation of the same 8 units**, not three independent unknown-control replications.
   - *Fix:* Updated `detection_metrics` to report both `conditional_known_only` and `full_pool` (where healthy + damaged + unknown_control are ranked together without hiding unknown labels, plus `unknown_in_topk_count`). Verified by `tests/v3_cpu/test_unknown_pool_metrics.py`.

## 2. Impact on H1 Claim
- **Verdict:** **需要收窄（Narrowed; full equal-budget & full-pool confirmation is `PENDING_GPU_VALIDATION`）.**
- Existing evidence shows `m2_reference_free` detects direct-vs-constraint contradictions on known units with lower unknown false alarms than simple direct screening on the shared 8-item unknown control set, but equal-cost comparison, behavioral post-injection ground truth, and disjoint unknown-control validation require the next GPU run (`NEXT_GPU_PROTOCOL.md`, Experiment G1).
