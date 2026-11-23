# KRMM v0.3 P0 Evidence & Code Consistency Audit

**Audit Branch:** `codex/krmm-v03-cpu-audit`  
**Baseline Commit:** `42043bd1c719d995f4d3046895a1b1dbe459c254` (clean working tree prior to audit)  
**Runtime Environment (CPU-Only):** Python `3.12.13`, `numpy==2.4.4`, `scipy==1.18.1`, `scikit-learn==1.9.1`, `pytest==9.2.0` (`CUDA_VISIBLE_DEVICES=""`)  
**Baseline Regression Tests (Pre-Edit):** `pytest tests/v3/ -v` (**13/13 PASSED**, 14.43s); `unittest tests/test_contracts.py` (**5/5 PASSED**, 0.011s)

---

## 1. Protected Historical Artifact SHA256 Baseline

| File | SHA256 (Pre-Audit) |
|---|---|
| `docs/v3/EXPERIMENTS.md` | `c34d89ecafff6889f6e676eb34f639854e68d5321ccb8c31849f1011bb0a4b63` |
| `docs/v3/GATE_REPORT.md` | `f7d6f9f51e2cc9a19888cad81e02bc4cb077ff848cb8635dbcfc131e00401961` |
| `docs/v3/EVIDENCE_PACK.md` | `a33e9ecb5a2cbe2fc57f61d775975d89f2750776a36e86882c404c285d0516c7` |
| `docs/v3/PROTOCOL.md` | `46ccd1419799a6f2430b8af41b55c0c4bf53f07e8e94ca3ad022f08cd1fe2abd` |
| `docs/v3/DATA_MANIFEST.jsonl` | `b4a1fc522492868c7360e921b7e02e787f94b91821c9bc7c3768e8b47e382b4d` |
| `results/v3-new/e0_e7_full_final.json` | `0a4b6c596747e0a540e2573a7fe209d7d9b9100d1a66883d1636fb97b19fbab3` |
| `configs/v3/p0_d2_steps64.json` | `b4695d61f96efa8411e1fe360483456069a16a4497ffa735fd98838ea54a9681` |
| `configs/v3/p2_dev12_orth_e6.json` | `74d29687a82a82671d20b24e21030ed64475642ba2f82e6f145e9c33cdd7f9c0` |
| `configs/v3/p2_dev36_full.json` | `d89fe1c6a8b867edbc00a42563c91bc88e063d0f1d0dd268e6aa4e9b05062fa9` |
| `configs/v3/p2_cal36_full.json` | `ea354cf94be5337c4c43a803e0f5cf5ab84a35f6228c397e49771ef9852b4749` |
| `configs/v3/p2_test48_frozen.json` | `08b5831a55329bd71bf1e33f509b18a0be84d88ead17d2ca7190de7a3ba6013f` |

---

## 2. Experiment-to-Config-to-Code Mapping Table

| Exp ID | Run Dir (`results/v3-new/`) | Git SHA at Run | Config File & SHA256 (prefix) | `damage_type` (`steps`, `retain_w`) | `write_strategy` (`steps`, `retain_w`) | Split / `n_total` | `strict_eligible` | `oracle_strict` |
|---|---|---|---|---|---|---|---|---|
| V3-01 | `exp01-mastery-survey-dev` | `bbf4ab0` | survey script (no LoRA) | — | — | `dev` (36) + `unk` (8) | — | — |
| V3-02 | `exp02-p0-d1-dev12` | `bbf4ab0` | `p0_baseline_d1.json` (`aee7b796`) | `D1_wrong_association` (24, 50.0) | `standard_kl` (32, 50.0) | `dev` / 12 | 0 | 0 |
| V3-03 | `exp03-p0-d1-s64-dev12` | `bbf4ab0` | `p0_d1_steps64.json` (`5cea3c05`) | `D1_wrong_association` (64, 50.0) | `standard_kl` (32, 50.0) | `dev` / 12 | 10 | 10 |
| V3-04 | `exp04-p0-d2-s64-dev12` | `bbf4ab0` | `p0_d2_steps64.json` (`b4695d61`) | `D2_suppression` (64, 50.0) | `standard_kl` (32, 50.0) | `dev` / 12 | 3 | 3 |
| V3-05a | `exp05a-m3-probe-check-dev` | `bbf4ab0` | survey script | — | — | `dev` / 36 | — | — |
| V3-05c | `exp05c-m2-discovery-dev36` | `bbf4ab0` | `p0_d1_steps64.json` (M2 script) | `D1_wrong_association` (64, 50.0) | — (detection only) | `dev` (33) + `unk` (8) | — | — |
| V3-06 | `exp06-e5-e6-dev12` | `bbf4ab0` | `p2_dev12_orth_e6.json` (`74d29687`) | `D1_wrong_association` (64, 50.0) | **`orthogonal_retain_proj`** (32, 50.0) | `dev` / 12 | 9 | 9 |
| V3-07 | `exp07-p2-dev36` | `bbf4ab0` | `p2_dev36_full.json` (`d89fe1c6`) | `D1_wrong_association` (64, 50.0) | **`standard_kl`** (32, 50.0) | `dev` / 36 | 21 | 21 |
| V3-08 | `exp08-p2-cal36` | `bbf4ab0` | `p2_cal36_full.json` (`ea354cf9`) | `D1_wrong_association` (64, 50.0) | **`standard_kl`** (32, 50.0) | `calibration` / 36 | 16 | 14 |
| V3-09 | `exp09-p2-test48` | `bbf4ab0` | `p2_test48_frozen.json` (`08b5831a`) | `D1_wrong_association` (64, 50.0) | **`standard_kl`** (32, 50.0) | `test` / 48 | 29 | 27 |
| V3-10 | `exp10-m2-discovery-cal36` | `bbf4ab0` | `p2_test48_frozen.json` (M2 script) | `D1_wrong_association` (64, 50.0) | — (detection only) | `calibration` (32) + `unk` (8) | — | — |
| V3-11 | `exp11-m2-discovery-test48` | `bbf4ab0` | `p2_test48_frozen.json` (M2 script) | `D1_wrong_association` (64, 50.0) | — (detection only) | `test` (43) + `unk` (8) | — | — |

---

## 3. Identified Discrepancies (Read-Only Historical Docs vs Actual Code/Configs)

### Discrepancy 1: Writeback Strategy Misattribution (`orthogonal_retain_proj` vs `standard_kl`)
- **原始陈述:**
  - `docs/v3/GATE_REPORT.md` line 14: `G0 (P0) | Clean mastery, local effective D1 damage, supervised LoRA upper bound (orthogonal_retain_proj) | PASS (dev36, cal36, test48)`
  - `docs/v3/GATE_REPORT.md` line 25: `whereas orthogonal_retain_proj achieved 9/9 (100%) on exp06, 21/21 (100%) on dev36, 14/16 (87.5%) on cal36, and 27/29 (93.10%) on sealed test48.`
- **源码或配置证据:**
  - `configs/v3/p2_dev12_orth_e6.json` (`run_meta.json` in `exp06-e5-e6-dev12`): `"write_strategy": "orthogonal_retain_proj"`, `"repair_retain_weight": 50.0`.
  - `configs/v3/p2_dev36_full.json`, `configs/v3/p2_cal36_full.json`, `configs/v3/p2_test48_frozen.json` (and `run_meta.json` in `exp07`, `exp08`, `exp09`): all specify **`"write_strategy": "standard_kl"`**, `"repair_retain_weight": 50.0`.
- **问题类型:** 报告与实际运行配置归属不一致（高严重性，Method Misattribution）。
- **修正后的建议表述:**
  - `exp06` (`dev12`) 验证了 `orthogonal_retain_proj` + KL (`9/9` oracle strict) 优于无约束 LoRA (`4/9`)；而全量 `dev36` (`21/21`)、`cal36` (`14/16`) 与封存 `test48` (`27/29` oracle, `20/29` structured) 实际执行的冻结写回策略为 **`standard_kl`**（`repair_retain_weight=50.0`, `repair_steps=32`）。不得将 `standard_kl` 的 27/29 封存结果归功于 `orthogonal_retain_proj`。
- **是否影响历史数值:** 否（历史 21/21、14/16、27/29 数值本身真实对应 `standard_kl`）。
- **是否需要新实验才能确认:** 若要宣称 `orthogonal_retain_proj` 在全量 `dev/cal/test` 上通过确认性测试，标记为 **`PENDING_GPU_VALIDATION`**。

### Discrepancy 2: D2 Regularization Misdescription
- **原始陈述:**
  - `docs/v3/GATE_REPORT.md` line 59: `D2_response_suppression without KL regularization (exp04) yielded only 5/12 effective and 3/12 strict-eligible units`
  - `docs/v3/EXPERIMENTS.md` line 366: `D2 (D2_response_suppression) without explicit KL constraint remains fragile`
- **源码或配置证据:**
  - `configs/v3/p0_d2_steps64.json`: `"damage_type": "D2_suppression"`, `"damage_steps": 64`, **`"damage_retain_weight": 50.0`**, `"damage_l2_weight": 0.0001`.
  - `krmm/v3/damage.py:146-161`:
    ```python
    target_loss = -torch.log1p(-torch.exp(-ce).clamp(max=1.0 - 1e-6))
    ...
    loss = target_loss + retain_weight * kl + l2_weight * l2
    ```
  - 实际损失由三项组成：(1) 无上界（仅在 `1 - 1e-6` 处数值截断，上限约 `13.8`）的目标 unlikelihood 损失 `target_loss`（权重 `1.0`）；(2) 保留池 KL 正则 `kl`（权重 `damage_retain_weight = 50.0`）；(3) LoRA 参数 L2 正则 `l2`（权重 `1e-4`）。
- **问题类型:** 损失函数描述不准确与机制归因错误（中-高严重性）。
- **修正后的建议表述:**
  - `exp04`（配置名 `D2_suppression`）**启用了与 D1 相同的 KL 保留正则（`damage_retain_weight=50.0`）和 L2 正则（`1e-4`）**，但目标项采用未设置边界/margin 的 unlikelihood 损失 `-log(1 - P(gold))`，其梯度仍导致 `2/5` 有效损伤样本在 `same_entity_other_relation` 上破坏独立局部性（`l2 ≈ 49.7`）；这说明现有 `50.0` 权重的保留池 KL 正则不足以抵消无界 unlikelihood 在同实体邻近关系上的溢出。
- **是否影响历史数值:** 否（`5/12` 有效、`3/12` 严格合格、`3/3` oracle 数值不变）。
- **是否需要新实验才能确认:** 若要验证有界/margin D2 抑制能否稳定通过 G0，标记为 **`PENDING_GPU_VALIDATION`**。

### Discrepancy 3: `repair_steps` Step Count Misstatement in Failure Analysis
- **原始陈述:**
  - `docs/v3/EXPERIMENTS.md` line 321: `Two (concert bass drum, bodhran) stopped at 66.7% heldout accuracy within 24 LoRA steps (under-written)`
- **源码或配置证据:**
  - `configs/v3/p2_cal36_full.json` & `configs/v3/p2_test48_frozen.json`: `"repair_steps": 32`（`24` 是 `p0_baseline_d1.json` 中的旧 `damage_steps`）。
- **问题类型:** 笔误 / 超参数引用错误（低严重性）。
- **修正后的建议表述:** `concert bass drum` 与 `bodhran` 在 `repair_steps=32` 的预算下仅达到 `66.7%` heldout 准确率。
- **是否影响历史数值:** 否。
- **是否需要新实验才能确认:** 否。

### Discrepancy 4: Uncalibrated Logistic Regression Labeled `"calibrated"`
- **原始陈述:**
  - `results/v3-new/e0_e7_full_final.json` 与 `docs/v3/EXPERIMENTS.md` 将 `DecomposedPredictor` 的拟合状态标记为 `"calibration_status": "calibrated"`。
- **源码或配置证据:**
  - `krmm/v3/repairability.py:107-109` 在 `LogisticRegression.fit()` 后直接赋值 `self.calibration_status = "calibrated"`，没有经过独立的概率校准集合与校准步骤。
- **问题类型:** 术语与统计状态误标（高严重性，见 P1 修复）。
- **修正后的建议表述:** 原实现仅完成了有监督拟合（应为 `fitted_uncalibrated`），未经独立概率校准。
- **是否影响历史数值:** 不改变排序指标（AUROC/AUPRC），但影响对 Brier 分数与概率校准声明的解释。
- **是否需要新实验才能确认:** 离线可使用 `dev` 拟合 + `cal` 独立校准复算；新一轮确认性校准验证标记为 **`PENDING_GPU_VALIDATION`**。

### Discrepancy 5: M2 Historical Artifacts Lack Per-Unit Logs & Reuse Same 8 Unknown Controls
- **原始陈述:**
  - `docs/v3/GATE_REPORT.md` 将 `dev36`/`cal36`/`test48` 上的 `Unknown FPR = 0.000` 并列展示，且未标明三组复用了同一批 8 个 `unknown_control` 单元；也未说明三组平均查询成本不等（`3.00` vs `4.10–4.57`）。
- **源码或配置证据:**
  - `krmm/v3/pipeline_p1_p3.py:40`: `unk_units = all_evaluator_units(split="unknown_control")` 固定加载同一组 8 个单元。
  - `results/v3-new/exp05c-m2-discovery-dev36/`, `exp10-m2-discovery-cal36/`, `exp11-m2-discovery-test48/` 均仅包含汇总文件 `m2_discovery_metrics.json`，无逐单元打分与逐查询日志。
- **问题类型:** 实验设计缺陷与归档粒度不足（高严重性，见 P2 修复）。
- **修正后的建议表述:** 三组 `Unknown FPR = 0.000` 是对**同一批 8 个未知单元**的重复观测，并非三次独立未知集复现；历史 M2 的等预算曲线与全池指标因缺乏逐单元原始记录，标记为 **`NOT_RECOMPUTABLE_FROM_AVAILABLE_ARTIFACTS`**。
- **是否影响历史数值:** 历史条件化汇总 AUROC 保留原值但必须加限定说明；全池/等成本指标不可复算。
- **是否需要新实验才能确认:** 是，标记为 **`PENDING_GPU_VALIDATION`**。
