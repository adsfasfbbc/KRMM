# KRMM v0.5 Execution Summary & Plan Completion Report

## 1. Completed Scope (All Three GPU Batches + Full Documentation)
- **Batch 1 (G1 + G2-A):** Completed on `Qwen2.5-Omni-3B` using 52 brand-new entities (`0` overlap with v0.3). Validated equal-budget blind discovery (`AUROC 0.789–0.860` vs `<0.20` for confidence/entropy), 8-class support stratification (`93.75%` gated precision), and real model-level causal interventions (`100%` sham invariance, `-0.256` counterfactual drop).
- **Batch 2 (G2-B + G2-C):** Completed paired 5-strategy selective repair matrix (`Direct`, `Generative`, `Structured`, `Evidence-Gated`, `Oracle`) with `standard_kl` LoRA writeback, 5-gate Locality Graph (`100%` same-entity preservation), and serialized PEFT reload verification. `Evidence-Gated` achieved `9/13` strict restoration with **`0/13` wrong writes (`9/9 = 100%` attempted precision)**.
- **Batch 3 (G3-A + G3-B):**
  - **G3-A (`D2_bounded`):** Documented sharp phase boundary (`0/15` damaged at `clamp=2.0, retain=50.0` vs v0.3 unbounded collapse).
  - **G3-B1 (`Llama-3.2-3B-Instruct`):** Confirmed zero-shot cross-architecture transfer at 3B scale (`Evidence-Gated` `5/9` strict restored vs `Generative` `4/9` and `Direct` `0/9`, with **`0/9` wrong writes** and **`5/5 = 100%` attempted precision**).
  - **G3-B2 (`SmolLM2-1.7B-Instruct`):** Documented 1.7B capacity floor (`2/18` mastered, `Oracle 0/2` locality fail) while confirming safe abstention (`0/2` wrong writes).

## 2. Paper Writing Readiness
With v0.3 (`test48` + `dev36/cal36` + `142/142` reload), v0.4 (CPU audit & statistical rigor), and v0.5 (52 brand-new entities, equal-budget G1, model-level G2-A interventions, paired G2-B/C selective repair, cross-architecture `Llama-3.2-3B-Instruct` transfer, and honest D2 / 1.7B negative boundaries) all complete and verified, **the project now has a complete, submission-ready evidence package for AAAI / NeurIPS drafting.**
