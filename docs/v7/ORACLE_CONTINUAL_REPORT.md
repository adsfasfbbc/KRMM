# KRMM v0.7 G0-Diagnosis Report: Oracle Continual Interference (`ORACLE_CONTINUAL_REPORT.md`)

- **Artifact:** `results/v7-new/OW-replay/diagnosis_summary.json`
- **Model:** `/root/rivermind-data/models/Qwen2.5-Omni-3B` (`m0_sha256 = 6df2e427b7b9...`, `D = 18`, `8` Oracle targets)
- **Runtime:** `753.6s` | **Gate Verdict:** **`diagnosis_G0 = PASS`**

## 1. Matched Diagnosis Matrix (`D0–D4`)
| Control / Branch | Description | Restored (`R`) | `H_new` | `H_ever` | `Forget_t` | Fresh Subprocess | Finding |
|---|---|---:|---:|---:|---:|---|---|
| `D0_no_update` | Empty loop + control check | 0/18 | **0** | **0** | 0.00 | N/A | Zero evaluator noise (`agree = 1.000`). |
| `D1_noop_merge` (1/2/4/8x) | Mathematical zero-`DeltaW` PEFT merge | 0/18 | **0** | **0** | 0.00 | N/A | Exact `hash_eq=True`, `max_dw=0.00e+00`, `max_wdiff=0.00e+00` across all 1/2/4/8 merges — **rules out PEFT/bf16 merge drift**. |
| `D2_single_from_initial` | Each of 8 targets independently from `M_0` | **8/8** | `6/8` have `0`; **`4/8 cedar shoe tree` (`H_new=2`, `dw=2.88`) & `7/8 slate chalkboard` (`H_new=2`, `dw=3.25`)** | — | — | N/A | Proves 2 high-norm single-step targets cause immediate collateral regression even without prior steps. |
| `D3_sequential_v6` | Step-by-step `M_0 -> M_8` (`W0`) | **16/18** | **2** | **2** | **0.00** | **Verified (`se_regs=2, loc=False`)** | Steps 1–3: `H_new=0`; Step 4 (`cedar shoe tree`): `0 -> 1`; Step 7 (`slate chalkboard`): `1 -> 2`; **zero forgetting of prior repairs (`Forget_t=0.00`)**. |
| `D4_joint_multitarget` | Matched 256-step joint LoRA | 14/18 | **1** | **1** | — | **Verified (`se_regs=1, loc=False`)** | Cuts harm in half (`2 -> 1`), but still fails strict `Y_strict = 0.0`. |

## 2. Mechanistic Conclusion
`H-D` is **CONFIRMED**: sequential `Y_strict = 0.0` in v0.6 is **not** evaluator noise, **not** bf16/PEFT merge drift, and **not** catastrophic forgetting of previously repaired targets (`Forget_t = 0.00` at all 8 steps). It is caused specifically by **immediate high-norm single-step updates (`cedar shoe tree`, `slate chalkboard`) introducing option-letter logit bias into immutable same-entity/sibling controls**.
