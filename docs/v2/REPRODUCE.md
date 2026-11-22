# KRMM v0.2 复现

全部命令在授权云服务器执行。使用同一Python/CUDA/PyTorch环境；当前记录为Python3.12.13、torch2.12.1+cu130、transformers4.57.6，完整依赖见environment-packages.json。此文档不自动授权新的GPU实验；新批次仍须独立预算。

## 源码与环境

仓库分支codex/krmm-v02包含结果快照。source-history.bundle保存云端各次实验的精确源码提交；运行记录中的git.head可据此检出。源码提交与GitHub归档提交SHA不同，使用bundle恢复相应实验源版本。

在云端克隆后，可执行以下只读验证/准备命令：

```bash
git bundle verify docs/v2/source-history.bundle
git fetch docs/v2/source-history.bundle HEAD:refs/heads/codex/recorded-v02-source
/root/rivermind-data/krmm-env/bin/python -m pip list --format=json
```

官方底座：Qwen/Qwen2.5-Omni-3B，revision f75b40e3da2003cdd6e1829b1f420ca70797c34e。模型完整文件校验见model-integrity.json；三个权重分片和所有已下载辅助文件均与官方API摘要匹配。下载器scripts/v2/download_model.py支持分块续传和摘要校验。网络代理配置与账户凭据保留在云端私有setup目录，未放入仓库。

## 核心制品

compact-damage-01下的41个damage-adapter可与官方底座合并，重建五个修订版条件的损伤状态。每个reproduction.json列出预期/实得tensor SHA；所有41个均再次序列化、重载、合并后验证。repair子目录包含oracle、generative、structured适配器（无候选时没有structured更新）。
大体积damaged_state.pt与teacher.pt保留在云端原结果目录，公共核心状态可由紧凑damage-adapter重建，teacher可在相同损伤模型的12条保留提示上重新计算。早期被否定的工程变体大快照只在云端保留，公共仓库保留其原始预测、适配器、配置、日志和SHA。

评测端先用scripts/v2/prepare_reproduction.py把evaluator-damage-schedules.json恢复到云端私有目录。表内仅记录错误目标，不给修复API。严格复现已记录的平衡试验时使用sweep加已记录assignment-source；不要重新运行随机采样器来替换已记录分配。

固定条件：rank8、alpha16、最后4层q_proj/v_proj/down_proj、writer32步、lr0.0005、KL50、factor-L2 0.0001、12条保留提示、每个分支seed107。新任务训练模板为2，独立评测4/5；它是v0.2新任务，不冒充完整v0.1再现。

## 独立评测与证据

krmm.v2.verify在全新进程中只加载损伤权重与序列化适配器，不读取辅助证据。compare probability_reload_max_error阈值1e-6。新模型进程下的原始结果在各unit目录fresh-evaluation.json，资源计数另存sidecar（早期工程试验未有sidecar）。
independent-locality-01使用每目标82个未参与KL的控制，所有原始概率、top token和答案字母质量都保留。SUMMARY.json记录严格恢复定义；不得把训练内保留或原本独立题错误后的新学习算作可靠恢复。

运行日志打包在run-logs.tar.gz；run-index.json可直接检索28次启动。所有失败保留，G0失败下的分析均为开发性，禁止将其称为确认性G1或G3。

Built with Qwen. LoRA文件是KRMM训练参数，适用QWEN_RESEARCH_LICENSE.txt及Notice。
