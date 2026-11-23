# KRMM v0.5 Baseline Audit & Interpretation Boundaries

## 1. Inherited Baseline Anchor
- **Baseline Branch:** `codex/krmm-v04-plan-cpu` (`8136f3e`)
- **v0.5 Working Branch:** `codex/krmm-v05-blind-repair`
- **Historical Protection:** `krmm/v3/`, `krmm/v4/`, `docs/v2/`, `docs/v3/`, `docs/v4/`, `results/v3-new/`, `results/v4-new/`, `configs/v3/`, `configs/v4/`, `tests/v3/`, `tests/v3_cpu/`, and `tests/v4/` remain untouched.

## 2. Verified Historical Evidence (Preserved Verbatim)
From **v0.3 (`42043bd`)**:
- Under `D1 + standard_kl` on `test48` (`n=29` strict-eligible targets), structured blind strict repair achieved `20/29`, generative baseline `13/29` (paired gain `+24.1 pp`).
- Structured wrong pseudo-label writes dropped from `16` (generative) to `4` (structured).
- Cross-process adapter reload checks passed `142/142`.

From **v0.4 CPU Audit (`8136f3e`)**:
- `redundant_support=27/27` represents **intra-model semantic redundancy** (`redundant_distinct_attributes`), NOT independent external data sources (`independent_source_support=0`).
- `single_chain=1/4` had insufficient sample size for universal claims.
- `142/142` reload checks corresponded to `(target, branch)` evaluations across `66` unique targets, not `142` distinct repaired knowledge points.
- Case audits (`timpani`, `crystal wine glass`) showed same-entity other-relation degradation can occur even when general commonsense controls pass.

## 3. Enforced Claim Boundaries in v0.5
v0.5 strictly forbids claiming:
1. Intra-model supporting attributes constitute independent external evidence (`independent_source_support` remains `0` in single-model experiments).
2. `27/27` proves redundancy guarantees 100% accuracy (indeed, v0.5 brand-new entities show `9/10 = 90.0%` on `redundant_distinct_attributes`, directly proving redundancy is high-reliability rather than infallible).
3. General multimodal self-repair has been demonstrated (v0.5 experiments evaluate text-probed factual repair on brand-new entities).
4. Universal cross-damage or small-model transfer without capacity bounds (v0.5 G3 explicitly documents the bounded-D2 phase boundary and 1.7B model capacity floor).
