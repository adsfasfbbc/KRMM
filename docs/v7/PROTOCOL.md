# KRMM v0.7 Protocol & Resource Contract (`PROTOCOL.md`)

- **Branch:** `codex/krmm-v07-continual-safe-repair` (from `bdb628a27feb5a41f8fa1676e5f9eeef9ed81f2b`)
- **Frozen Config:** `configs/v7/protocol.json`
- **Batch Scope Executed:** **Batch C1 (Oracle Continual Diagnosis D0–D4 + P1 Writer Matrix W0–W5 + T4 + Out-of-Sample DEV-V7 Writer Verification)** on `NVIDIA GeForce RTX 3090 24GB`.
- **Budget Guardrails:** `<= 12h` GPU wall time, `<= 24` launches, `<= 20 GiB` net disk, `>= 15 GiB` free disk, single GPU concurrency.
- **Strict Non-Overlap of Prompts (verified by `test_train_monitor_audit_prompt_separation`):**
  - `TRAIN_RETAIN`: `train_retain_prompts` + `EXTRA_GENERIC_RETAIN_PROMPTS` (W1) / `_TRAIN_RELATION_TEMPLATES` (W2/W4).
  - `PUBLIC_MONITOR`: `_MONITOR_RELATION_TEMPLATES` (T4 only).
  - `HIDDEN_AUDIT`: Evaluator-only `same_entity_controls` + `AUDIT_GENERAL_CONTROLS`.
- **Unauthorized Batches Not Run:** Batch C2 (full blind E2E on fresh TEST-V7) and Batch C3 (CounterFact/zsRE open-relation & D3 mixed fine-tuning) remain explicitly `NOT_RUN` pending user review of Batch C1 results.
