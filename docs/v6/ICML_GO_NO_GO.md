# KRMM v0.6 ICML 2027 Internal GO/NO-GO Assessment (`ICML_GO_NO_GO.md`)

**Current Recommendation:** `CONTINUE_POLISHING` (Batch A Preprint Core complete; Batch B ICML Expansion `NOT_RUN`).

## Checklist Against Section 10.2 Criteria (Target Decision Date: 2026-12-25)
1. **True E2E + Fair Equal-Cost Baselines:** **MET (Pilot)** — `qwen_pilot_g1_r2` and `qwen_pilot_e2e_r2` establish clear, non-trivial findings (reversed entropy vs constraint discrepancy trade-off; sequential cumulative same-entity interference even under Oracle).
2. **Evidence Gating Incremental Value or Strong Falsifiable Negative Finding:** **MET (Nuanced)** — Structured reconstruction gives `+52.6 pp` over generative and `100%` hitting-set/twin-world abstention, while within-coverage AURC shows honest null gain over simple confidence/fact-count (`0.084` vs `0.072/0.076`).
3. **Beyond 3 Closed-Class Domains (Open-Relation CounterFact/zsRE):** **NOT_RUN** — Reserved for authorized Batch B.
4. **Second Effective Damage Mechanism (`D3` Mixed Fine-Tuning) & Multi-Episode Statistical Power (`>=3` Episodes):** **NOT_RUN** — Reserved for authorized Batch B.
5. **Reproducibility, Related Work, AI Disclosure & Audit Integrity:** **MET** — 100/100 tests pass, zero protected historical directories modified, full `AI_USAGE_LOG.md` & `CLAIM_LEDGER.md` maintained.

**Actionable Path to ICML 2027 `GO`:** Authorize Batch B to run (a) an interference-mitigating sequential writer (e.g., orthogonal/null-space projected LoRA or per-step rollback with same-entity retention) to achieve non-zero strict `Y > 0` in multi-target E2E, (b) open-relation CounterFact/zsRE subset, (c) `D3` mixed fine-tuning damage, and (d) `>= 3` disjoint multi-target episodes for confirmatory confidence intervals.
