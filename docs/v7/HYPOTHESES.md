# KRMM v0.7 Hypotheses & Falsification Criteria (`HYPOTHESES.md`)

- **H-D (Diagnosis):** The `Y_strict = 0` failure of 8-step Oracle repair in v0.6 is caused by real per-fact sequential parameter interference (same-entity letter-bias bleed and sibling domain drift), NOT by evaluator noise, logging errors, or PEFT merge/bf16 numerical drift.
  - *Falsification:* `D0` (no-update) or `D1` (1/2/4/8x zero-DeltaW no-op merges) produce non-zero drift/harm, or single-target `D2` already fails on every target.
- **H-W (Relation-Aware Continual Writer):** Under matched target candidates and matched retain-prompt/compute budgets, adding strictly public entity-relation retain prompts (`W2`) and/or provisional history replay (`W3`/`W4`) reduces per-fact new harm (`H_new`, `H_ever`) and forgetting (`Forget_t`) compared to both `W0` (`standard_kl`) and `W1` (generic-count matched KL), achieving `OW-PILOT` (`Y_strict > 0`, `>= 4/8` restored, `0` same-entity regressions, fresh-subprocess reload PASS).
  - *Falsification:* Adding equal-count generic retain prompts (`W1`) explains the entire gain, or `W2`/`W3`/`W4` fail to eliminate same-entity regressions (`H_new > 0`).
- **H-T (Public Transactional Monitor):** Using strictly disjoint `PUBLIC_MONITOR` prompts to gate commits (`T4`) catches harmful updates before commit without reading hidden audit labels.
  - *Falsification:* `T4` only avoids harm by rolling back nearly all valid updates, or misses hidden regressions while rejecting clean updates.
- **H-U / H-E / H-X (Batch C2/C3):** Unknown-entity control, true blind continual E2E on new episodes, open-relation (`CounterFact/zsRE`), and `D3` mixed fine-tuning — gated strictly behind `OW-PILOT` PASS (`NOT_RUN` unless authorized).
