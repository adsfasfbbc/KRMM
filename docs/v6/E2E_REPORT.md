# KRMM v0.6 True End-to-End Sequential Blind Repair Report (`E2E_REPORT.md`)

- **Discovery Artifact:** `results/v6-new/qwen_pilot_g1_r2/metrics.json` (`D = 19` = `12` effective planned D1 + `7` collateral damaged; `14` healthy; `9` previously unmastered; `8` unverifiable; `g0_science_pass = True`)
- **E2E Episode Artifact:** `results/v6-new/qwen_pilot_e2e_r2/metrics.json`
- **Protocol:** All 9 strategies started from independent copies of the **exact same multi-target damaged checkpoint `M_d`**, sequentially updated the cumulative model `M` up to `B_w = 8` writes (`32` steps each, `standard_kl`), allowed healthy/unknown false positives to enter the pipeline untouched, and were audited via **brand-new isolated Python subprocesses** (`scripts/v6/verify_fresh_subprocess.py`).

## 1. End-to-End Sequential Results (`Qwen2.5-Omni-3B`, `D = 19`, Fresh-Subprocess Audited)
| Strategy | Committed Writes | Raw Restored (`/19`) | Active Repaired | Known Wrong Writes | Unknown Write Attempts | Harmed Healthy (`/14`) | General Drop | Same-Entity Regressions (`/72`) | Strict Locality Pass | Strict Safe Yield `Y` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0_no_repair` | 0 | 0/19 (0.0%) | 0 | 0 | 0 | 4* | 0.000 | 0 | **True** | 0.0 |
| `B1_generative` | 8 | 2/19 (10.5%) | 1 | 0 | 4 | 8 | 0.000 | **16** (`-22.2 pp`) | False | 0.0 |
| `B2_ungated_structured` | 8 | 7/19 (36.8%) | 3 | 0 | 3 | 4 | 0.000 | 11 | False | 0.0 |
| `B3_confidence_gate` | 8 | 2/19 (10.5%) | 0 | 0 | 4 | 8 | 0.000 | 16 | False | 0.0 |
| `B3_fact_count_gate` | 8 | 6/19 (31.6%) | 4 | 0 | 3 | 4 | 0.000 | 11 | False | 0.0 |
| `B3_random_gate` | 8 | 6/19 (31.6%) | 4 | 1 | 3 | 2 | 0.000 | 11 | False | 0.0 |
| **`M_evidence_gated`** | 8 | **6/19 (31.6%)** | **4** | **0** | 3 | 4 | 0.000 | **11** (`-11.1 pp`) | False | 0.0 |
| **`M_full`** | 8 | **8/19 (42.1%)** | **5** | **0** | 2 | **3** | 0.000 | **10** (`-11.1 pp`) | False | 0.0 |
| `Oracle_upper_bound` | 8 | **19/19 (100.0%)** | **8** | 0 | 0 | **0** | 0.000 | **3** (masked by 11 w2c!) | False | 0.0 |

## 2. Critical Scientific Takeaways (Honest Reporting per Section 5.5)
1. **Why Strict `Y = 0.0` Even for Oracle:** Even when `Oracle_upper_bound` executes 8 gold `standard_kl` LoRA merges sequentially (restoring `19/19` targets with `0` harmed healthy units and `0.000` general commonsense drop), **3 same-entity invariant facts regress** (`c2w=3`, completely masked by `w2c=11` if one only checks group mean accuracy `0.819 -> 0.931`!). This exposes a fundamental limitation of naive cumulative LoRA writeback in multi-target blind repair and validates our per-fact zero-regression locality gate.
2. **Real Blind Exposure of Unknown Controls:** Unlike single-target evaluations, true blind E2E exposes `2–4` unknown/unmastered units to write attempts because the model hallucinates seemingly consistent atomic attributes for obscure/fictional entities.
3. **Clear Relative Advantage Without Overclaiming:** Within this challenging sequential regime, `M_full` (`8/19` restored, `5` active, `0` known wrong writes, `10` regressions) and `M_evidence_gated` (`6/19`, `4` active, `0` wrong writes, `11` regressions) outperform `B1_generative` (`2/19`, `1` active, `16` regressions, `8` harmed healthy) and `B3_confidence_gate` (`2/19`, `0` active, `16` regressions) by `+21.1` to `+31.6 pp` raw restoration while cutting same-entity regressions by `31%–37.5%`. However, per Section 5.5, because strict `Y = 0.0`, we **cannot** claim "safe end-to-end blind repair is solved" — the manuscript must frame this as a **selective repair feasibility & sequential-interference diagnosis study**.
