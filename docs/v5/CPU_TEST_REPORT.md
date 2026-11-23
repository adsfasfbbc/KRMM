# KRMM v0.5 CPU Verification & Regression Test Report

## 1. Test Suites Executed (`CUDA_VISIBLE_DEVICES=""`)
- `tests/v5/test_v5_suite.py`: **6/6 PASSED**
  1. `test_v5_catalog_100_percent_disjoint_from_v3`: verifies `36` known + `12` unknown + `4` ambiguous = `52` unique entities with `0` overlap against all `140` v0.3 entities.
  2. `test_all_eight_v5_support_classes`: validates all 8 support-structure classifications.
  3. `test_v5_rule_level_interventions_and_scope_tag`: validates rule-level counterfactual interventions and scope tagging.
  4. `test_v5_seven_detectors_equal_budget_and_three_class_metrics`: validates `CostTrackedProxy`, all 7 comparators, and 3-class + full-pool binary metrics.
  5. `test_v5_locality_graph_separates_ripple_from_collateral`: validates 5-gate locality graph separation.
  6. `test_v5_reliability_sparsity_guard_and_end_to_end_mock`: validates sparsity guard and end-to-end mock pipeline.
- `tests/v4/`: **23/23 PASSED** (unchanged)
- `tests/v3_cpu/`: **46/46 PASSED** (unchanged)
- `tests/v3/`: **13/13 PASSED** (unchanged)
- `tests/test_contracts.py`: **5/5 PASSED** (unchanged)

**Total:** **93/93 PASSED** (0 failures, 0 regressions, 0 modifications to protected historical v0.3/v0.4 directories).
