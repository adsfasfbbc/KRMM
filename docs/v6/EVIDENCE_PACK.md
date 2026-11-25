# KRMM v0.6 Master Evidence Pack (`EVIDENCE_PACK.md`)

- **Branch:** `codex/krmm-v06-preprint-icml` (branched from `7a41d02`)
- **Hardware:** 1× NVIDIA GeForce RTX 3090 (24 GB VRAM)
- **Batch A Budget Used:** ~`0.78` GPU wall hours (`~47 min` active GPU across `7` launches including `2` engineering-fix retries) vs `12.0` hours / `24` launches approved limit; `/root/rivermind-data` free disk remained `>= 20 GiB` (`>= 15 GiB` limit).
- **Test Suite:** `100/100` PASS (`7` v6 incl. real subprocess smoke + `6` v5 + `23` v4 + `46` v3_cpu + `13` v3 + `5` contracts); `0` modifications to protected v0.1–v0.5 directories.

## Section 13 Gate & Readiness Status
- `ENGINEERING_G0 = PASS`
- `REAL_BLIND_E2E = FAIL` *(Honest negative result: all strategies including Oracle achieved `Y = 0.0` under 8 sequential cumulative LoRA merges due to per-fact same-entity regressions despite `0.000` general drop; `M_full`/`M_evidence_gated` still achieved `8/19` and `6/19` raw recovery vs `2/19` generative on Qwen with `0` known wrong writes)*
- `FAIR_DETECTION = PASS` *(Clear complementary trade-off characterized: `neg_entropy_rev` wins AUPRC `0.676` at 50 queries with `0/7` collateral recall; `direct_vs_constraint` wins AUROC `0.724`, 3-class acc `0.540`, healthy FPR `0.143`, and `71.4%` collateral recall)*
- `SELECTIVE_MECHANISM = INCONCLUSIVE` *(Structured reconstruction `+52.6 pp` over generative and `100%` sham/hitting-set/twin-world interventions PASS, but within-coverage AURC `0.084` does not beat `direct_confidence` `0.072` or `fact_count` `0.076`)*
- `CROSS_MODEL_PERSISTENCE = PASS` *(Exact `0.00e+00` fresh-subprocess reload verified across all 6 strategies on `Llama-3.2-3B-Instruct`, replicating both cumulative same-entity interference and unknown-entity write leakage; `G0` mastery `23/36 = 63.9%` honestly logged as below 75%)*
- `OPEN_RELATION_TRANSFER = NOT_RUN` *(Batch B)*
- `CROSS_DAMAGE_TRANSFER = NOT_RUN` *(Batch B)*
- `PREPRINT = READY_FOR_HUMAN_REVIEW` *(Scoped strictly as a Selective Repair Feasibility & Sequential-Interference Diagnosis preprint; human author review & BibTeX author verification required)*
- `ICML_2027 = CONTINUE_POLISHING` *(Requires Batch B: interference-free sequential writer for `Y > 0`, open-relation CounterFact/zsRE, `D3` mixed fine-tuning, and `>= 3` disjoint episodes)*
