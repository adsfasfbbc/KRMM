# KRMM v0.7 P1 Writer Matrix & Out-of-Sample Confirmation Report (`WRITER_MATRIX_REPORT.md`)

- **Episode 1 (Opened v0.6 Replay):** `results/v7-new/OW-writer-matrix/writer_matrix_summary.json` (`D = 18`, `8` Oracle targets, `1344.3s`)
- **Episode 2 (Unseen Out-of-Sample `DEV-V7`):** `results/v7-new/OW-dev-v7/dev_writer_summary.json` (`36` brand-new entities, `0` overlap with `281` historical names/aliases, `mastered = 30/36`, `D = 15`, `8` Oracle targets, `seed = 3407`, `1122.6s`)
- **Gate Verdicts:** **`OW_PILOT = PASS`** (on replay), **`OW_CONFIRM (Out-of-Sample DEV-V7) = FAIL`** -> **STOP per Section 5.4 / 13**.

## 1. Episode 1 — `OW-writer-matrix` (v0.6 Replay, `D = 18`, Isolated Fresh-Subprocess Verified)
| Writer Branch | Comm | Rollbacks | Oracle 8 | Restored `D` | `H_new` | `H_ever` | `Forget` | Fresh Reload | Strict Loc | **Fresh `Y_strict`** | `Σ‖ΔW‖_F` |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| `W0_standard_kl` (v6 baseline) | 8/8 | 0 | 8/8 | 16/18 | 1 | 2 | 0.00 | PASS | False (`se_regs=1`) | `0.000` | 13.72 |
| `W1_generic_matched` (+8 balanced generic KL) | 8/8 | 0 | 8/8 | 16/18 | **0** | **0** | **0.00** | PASS | **True (`se_regs=0, ag=0.913`)** | **`0.889`** | 14.26 |
| `W2_relation_retain` (+8 public relation KL) | 8/8 | 0 | 8/8 | 14/18 | 1 | 1 | 0.09 | PASS | False (`se_regs=0, ag=0.891`) | `0.000` | 16.18 |
| `W3_history_replay` (provisional replay only) | 8/8 | 0 | 8/8 | 16/18 | 1 | 2 | 0.09 | PASS | False (`se_regs=1`) | `0.000` | 13.61 |
| `W4_relation_plus_history` (W2 + W3) | 8/8 | 0 | 8/8 | 14/18 | **0** | **0** | **0.00** | PASS | **True (`se_regs=0`)** | **`0.778`** | 15.79 |
| `T4_W4_public_transaction` (W4 + public monitor) | 6/8 | 2 | 8/8 | 14/18 | **0** | **0** | 0.12 | PASS | **True (`se_regs=0`)** | **`0.778`** | 12.75 |
| `W5_joint_with_relation` (joint + relation KL) | 1 | 0 | 8/8 | 14/18 | **0** | **0** | — | PASS | **True (`se_regs=0`)** | **`0.778`** | — |

## 2. Episode 2 — Out-of-Sample `DEV-V7` Confirmation (`D = 15`, Unseen 36 Entities, Fresh-Subprocess Verified)
| Writer Branch | Oracle 8 | Restored `D` | `H_new` | `H_ever` | Fresh Reload | Strict Loc | `se_regs` | **Fresh `Y_strict`** |
|---|---:|---:|---:|---:|---|---|---:|---:|
| `W0_standard_kl` | 8/8 | 15/15 (100%) | 1 | 1 | PASS | False | 1 | `0.000` |
| `W1_generic_matched` | 8/8 | 15/15 (100%) | 2 | 2 | PASS | False | 2 | `0.000` |
| `W2_relation_retain` | 8/8 | 15/15 (100%) | 3 | 5 | PASS | False | 3 | `0.000` |
| `W4_relation_plus_history` | 8/8 | 15/15 (100%) | 1 | 2 | PASS | False | 1 | `0.000` |

## 3. Scientific Verdict (Honest Negative Out-of-Sample Generalization)
1. **Why Out-of-Sample Validation Mattered:** While `W1_generic_matched` (`Y = 0.889`) and `W4_relation_plus_history` (`Y = 0.778`) eliminated same-entity regressions on the opened v0.6 replay episode (`OW-replay`), **neither `W1` (`se_regs = 2`) nor `W2` (`se_regs = 3`) nor `W4` (`se_regs = 1`) generalized to zero same-entity regressions on the brand-new unseen `DEV-V7` episode**, even though all 4 restored `15/15 (100%)` degraded targets!
2. **Core Mechanism Conclusion:** Soft KL retention over either balanced generic prompts (`W1`) or public entity-relation prompts (`W2`/`W4`) is **insufficient to guarantee per-fact zero-regression continual LoRA updates** across unseen entity distributions. Even under gold Oracle supervision, `1–3` immutable same-entity regressions persist per 8-step continual sequence.
3. **Stop Decision (Section 5.4 & 13):** Batch C1 cleanly stops here without launching unauthorized Batch C2/C3. The evidence supports pivoting the manuscript framing to *"Beyond Oracle Knowledge Editing: Evidence-Guided Repair and the Limits of Continual Knowledge Restoration"* (Section 12.2).
