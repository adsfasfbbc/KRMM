# KRMM v0.3 CPU Audit Corrections Ledger

**Branch:** `codex/krmm-v03-cpu-audit` (from `42043bd1c719d995f4d3046895a1b1dbe459c254`)  
**Execution Mode:** CPU-Only (`CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`)

## Summary Table of Confirmed Issues

| 问题 | 严重程度 | 原实现 | 修复方案 | 验证结果 |
|---|---|---|---|---|
| **1. `write_strategy` 归属错误** (`GATE_REPORT.md`) | 高 | 将 `exp07`/`exp08`/`exp09` 的 `21/21`、`14/16`、`27/29` 归功于 `orthogonal_retain_proj` | 保持原报告只读；在 `01_EVIDENCE_CONSISTENCY.md` 建立完整实验-配置-SHA256 映射表，明确全量结果属于 `standard_kl`，正交投影仅在 `exp06 (dev12)` 验证 | `test_report_config_consistency.py` 通过；正交投影全量评测标为 `PENDING_GPU_VALIDATION` |
| **2. `D2` 正则表述不准确** (`GATE_REPORT.md`, `EXPERIMENTS.md`) | 中-高 | 将 `exp04` 简写为“无 KL 正则” | 核查 `p0_d2_steps64.json` 与 `damage.py:146-161`，明确 KL (`retain_weight=50.0`) 与 L2 (`1e-4`) 均已启用，脆弱性源于无界 unlikelihood 目标损失 | `test_report_config_consistency.py` 通过；有界 D2 标为 `PENDING_GPU_VALIDATION` |
| **3. M4 概率分解重复计数** (`repairability.py`, `analyze_e0_e7.py`) | 极高 | `p_rel = P(cand\|x)` 与 `p_write = P(strict\|x)` 直接相乘，重复计入候选正确性 | 拆分为 `DirectSuccessPredictor`、`CandidateReliabilityPredictor`、`ConditionalWriteFeasibilityPredictor`（仅在 `cand_correct=1` 或 oracle 写回上拟合），修正 `DecomposedPredictor` 分解公式 | `test_probability_factorization.py`、`test_conditional_write_model.py` 通过；离线复算揭示 `dev` 无写回负样本、`dev+cal` 仅 2 例负样本 |
| **4. M4 拟合即宣称 `"calibrated"`** (`repairability.py`) | 高 | `LogisticRegression.fit()` 后直接设 `calibration_status = "calibrated"` | 严格区分 `untrained` / `fitted_uncalibrated` / `calibrated` / `insufficient_data`；仅在独立校准集通过 Platt 校准后才允许标记 `calibrated`；小样本/单类强制返回 `insufficient_data` | `test_calibration_status.py`、`test_small_sample_fallback.py` 通过 |
| **5. M2 损伤目标切片偏差** (`pipeline_p1_p3.py`) | 高 | `mastered_evs[:n_dmg]` 直接按目录顺序截取前一半 | 新增 `select_stratified_damage_targets`：域内随机平衡采样、可选按答案分层、随机化注入顺序、导出分配 manifest 与 SHA256 | `test_damage_randomization.py` 通过；真实模型重跑标为 `PENDING_GPU_VALIDATION` |
| **6. M2 损伤真值未区分计划与行为退化** (`pipeline_p1_p3.py`) | 高 | 直接以“计划注入名单”作检测真值 | 新增 `classify_post_injection_truth` 与 `build_post_injection_truth_table`，区分 `planned_damaged` / `injection_completed` / `behaviorally_degraded` / `strict_local_damage` / `collateral_damage`，并在注入后重审最终模型 | `test_post_injection_truth_schema.py` 通过；历史 M2 标为 `NOT_RECOMPUTABLE_FROM_AVAILABLE_ARTIFACTS` |
| **7. M2 查询预算不公平与缺少逐单元日志** (`discovery.py`, `pipeline_p1_p3.py`) | 高 | `simple_direct_baseline` (`3.00` calls/unit) 与 `m2_reference_free` (`4.57` calls/unit) 成本不等；未导出逐单元原始记录 | 新增 `BudgetedQueryProxy` 与 `evaluate_detection_at_budget` 强制硬停止与逐单元记账；`run_m2_discovery_eval` 导出 `per_unit_records` | `test_detection_budget_accounting.py` 通过；历史 M2 等成本曲线标为 `NOT_RECOMPUTABLE_FROM_AVAILABLE_ARTIFACTS` |
| **8. M2 排序指标排除 `unknown_control` 且三组复用同 8 题** (`metrics.py`) | 高 | `detection_metrics` 计算 AUROC / Recall@K 时剔除了 `unknown_control` | 保留 `conditional_known_only` 同时新增 `full_pool` 全池盲排指标与 `unknown_in_topk_count`；审计说明 8 个未知对照重复使用 | `test_unknown_pool_metrics.py` 通过；不重叠新未知集写入 `NEXT_GPU_PROTOCOL.md` |
| **9. E7 选择统计混淆分母与基线概率映射** (`pipeline_p1_p3.py`, `analyze_e0_e7.py`) | 中-高 | 仅在 `24` 个尝试单元上汇报 `strict_success_all_eligible`，未显式区分 `29` 个严格合格单元与未校准排序基线 | 统一汇报 8 级分母、`candidate_coverage`、`selected_coverage_among_attempted` / `all_eligible`、`wrong_write_rate`、`abstention_rate` 及 `is_calibrated_probability` | `test_selector_coverage.py`、`test_paired_bootstrap.py` 通过 |
| **10. EvidenceGraph 最小支持集未精确枚举** (`knowledge_graph.py`, `reconstruction.py`) | 中 | 仅检查单事实是否均模糊，未区分 `redundant_support` 与 `minimal_multi_premise` | 实现 `find_minimal_support_sets`，严格划分 5 类解释并输出 `rule_trace` 与最小子集；区分规则级、候选级与模型级因果依赖 | `test_minimal_support_set.py`、`test_counterfactual_rule_controls.py` 通过；离线复算候选零变化 |
| **11. `pipeline_p1_p3.py` 顶层导入 `Omni`** (`pipeline_p1_p3.py`) | 中 | 顶层 `from krmm.model import Omni` 妨碍纯 CPU 审计 | 改为仅在 GPU 入口按需懒导入，并新增 `scripts/v3/offline_audit.py` 纯 CPU 入口 | `test_no_gpu_execution.py` 通过 |
| **12. `EXPERIMENTS.md:321` `repair_steps` 笔误** (`EXPERIMENTS.md`) | 低 | 将 `repair_steps` 误写为 24（实际配置为 32） | 记录于 `01_EVIDENCE_CONSISTENCY.md`，不修改只读历史文件 | `test_report_config_consistency.py` 通过 |
