# KRMM v0.3 Experiment Matrix (Section 10.2)

| Cell | Candidate Source | Writeback Strategy | Research Purpose | Run Artifacts |
|---|---|---|---|---|
| **E0** | None | No writeback | Damage discovery (M2) & damaged heldout baseline | `exp05c-m2-discovery-dev36`, `exp07-p2-dev36`, `exp08-p2-cal36` |
| **E1** | Direct residual / Generative | No writeback / Standard LoRA | Direct & generative recovery baselines | `exp07-p2-dev36`, `exp08-p2-cal36` |
| **E2** | Structured (`EvidenceGraph`) | No writeback | Candidate accuracy, coverage, precision vs single-hop | `exp05a`, `exp07-p2-dev36`, `exp08-p2-cal36` |
| **E3** | Structured (`EvidenceGraph`) | Constrained LoRA (`standard_kl`) | End-to-end strict persistent repair | `exp07-p2-dev36`, `exp08-p2-cal36` |
| **E4** | Oracle gold answer | Same constrained LoRA | Write feasibility upper bound (G0 positive control) | `exp03`, `exp07-p2-dev36`, `exp08-p2-cal36` |
| **E5** | Oracle gold answer | Unconstrained LoRA (no retention KL) | Ablation isolating retention constraint necessity | `exp06-e5-e6-dev12` |
| **E6** | Oracle / Structured / Generative | `orthogonal_retain_proj` | Alternative low-interference gradient projection | `exp06-e5-e6-dev12` |
| **E7** | Same candidates & writeback | `DecomposedPredictor` vs baselines (`candidate_change`, etc.) | Independent repairability calibration & selection gain | `exp07` -> `exp08` cross-eval, `test` confirmatory |
