# KRMM v0.8 历史基线预检

预检日期为 2026-11-27（Asia/Singapore）。父提交为 `92501018228213bac8c9235169857ce7673ed964`，与服务器原 v0.7 HEAD 和 GitHub v0.7 分支一致。从该提交建立了 `codex/krmm-v08-constrained-scale-audit`。原工作树已有未跟踪的历史结果和运行日志，已保留。没有写入历史目录。

## 完整性与可复现性

`sha256sum -c results/v7-new/SHA256SUMS` 列出的 35 项中，34 项吻合，1 项路径缺失。

- 路径为 `results/v7-new/OW-replay/D4_joint_adapter/adapter_config.json`。
- 预期 SHA256 为 `941d060e9b2fccdd0e2b7e95bf59d5edda59640d2035912a3a6c80e0c87bbfdb`。
- 该路径不在父提交树中，在服务器已知 Git refs 中也未检索到该路径的提交记录。现有证据只能确认缺件，无法判断缺件发生的时间和原因。
- 历史 D3 与 W0 的 fresh job 均指向 `results/v7-new/OW-replay/damaged_state_M0.pt`，预期状态哈希为 `6df2e427b7b9bd6f341b51ae5ef892ce53a3c7fbf71f97fec586bbc7c5fa90ef`。该文件和 v7 适配器权重不在当前历史结果树中。两份历史 fresh 输出均记录了 8 个适配器重载成功和 `1e-4` 概率容差，本轮只能核对已保存输出。

`BASELINE_INTEGRITY=FAIL`，本轮按协议停在 `STOP_INTEGRITY`。非零 merge 复核为 `NOT_RUN`。

## D3 与 W0 的已保存证据

| 历史运行 | 恢复 | 最终同实体正确→错误 | 最终同实体错误→正确 | 轨迹曾受损事实 | 严格收益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OW-replay D3 | 16/18 | 2 | 8 | 2 | 0 |
| Writer matrix W0 | 16/18 | 1 | 8 | 2 | 0 |

D3 的 halibut 与 barracuda 分别在第 4、7 步首次出错。W0 的两条事实都在第 4 步首次出错，最终只剩 halibut 错误。第 4 步的写回目标均为 cedar shoe tree。两条鱼类问题在历史固定问法中的正确位置均为 A，语义归因仍为 `INCONCLUSIVE`。最终重载输出中的各组转移计数与运行摘要一致。W0 留存的摘要没有完整逐步预测及状态哈希，无法独立重算全部事实的逐步转移。

两次运行记录相同的初始 M0 哈希和八个目标的顺序，并均调用 v7 的标准 KL LoRA 写回，LoRA dropout 为 0。第 1 步的有效更新 Frobenius 范数已经不同（D3 约 2.431919，W0 约 2.490199）。D3 先执行 no-op 和独立写回，W0 则重载 M0 并另行计算 teacher logits。写回函数在挂载 LoRA 前没有按步复位 Torch RNG，历史产物也未保存对应的 RNG、初始适配器或 teacher logits 哈希。后续复现应在同一经核验的 M0 上固定每步 seed、对照 teacher logits 和初始化哈希，并比较 unmerged、merged 与独立 fresh 三条数值路径。现有记录不足以归因给初始化、teacher logits 或 BF16 merge 中的某一项。

## CPU 回归

共通过 106 项。合跑所有 pytest 目录首先遇到两份同名测试文件的导入冲突。改用 importlib 后，合跑仍在 v3 CPU 测试的 `sys.modules` 断言失败，因为前面的测试已导入 `krmm.model`。保持历史测试文件原样，将测试按进程隔离后通过。

在仓库根目录运行以下命令，使用环境 `CUDA_VISIBLE_DEVICES=''`、`PYTHONDONTWRITEBYTECODE=1`、`HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。

| 命令（Python 为 `/root/rivermind-data/krmm-env/bin/python`） | 通过 |
| --- | ---: |
| `python -m pytest -p no:cacheprovider tests/v3_cpu -q` | 46 |
| `python -m pytest --import-mode=importlib -p no:cacheprovider tests/v7 tests/v6 tests/v5 tests/v4 tests/v3 -q` | 55 |
| `python -m unittest tests/test_contracts.py` | 5 |

## 资源与下一次授权

仓库为 `/root/rivermind-data/KRMM`。GPU 0 是 RTX 3090，显存 24,576 MiB，预检时可用 24,125 MiB。预检找到了含三个权重分片的 `/root/rivermind-data/models/Qwen2.5-Omni-3B`。未找到 Omni-7B 权重或可加载的 Llama-3.2-3B 快照，`SCALE_7B=NOT_RUN_RESOURCE`。

仓库卷预检可用约 19.84 GiB，按 15 GiB 保留线，实际新增空间约 4.84 GiB。合同中可提议 16 GPU 小时、28 次主启动、80 次独立重载和 800 次 fit，单卡并发 1；具体命令、单次时限和磁盘上限随可执行队列另行审批。未冻结协议或队列，GPU 时间、主启动、重载、fit 均为 0，`OW_CONFIRM=NOT_RUN`，`BLIND_E2E=NOT_RUN_GATED`。

恢复原始历史件后应先核对其 SHA256 及 M0 重建或存档来源。若原件无法取得，须由项目所有者批准新版基线协议；应将任何重建结果保存在 v0.8 新运行目录中，并留存本次缺件记录。
