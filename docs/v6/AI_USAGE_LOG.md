# KRMM v0.6 AI Usage Disclosure Log (`AI_USAGE_LOG.md`)

In accordance with ICML 2027 AI usage disclosure policies, all substantive generative AI assistance in KRMM v0.6 is transparently logged below.

| Date | Tool / Agent | Scope of Assistance | Verification Evidence | Human Review Status |
|---|---|---|---|---|
| 2026-11-25 | Antigravity Coding & Research Agent (Gemini) | P0 read-only audit (`scripts/v6/audit_v5.py`), v0.6 codebase (`krmm/v6/*`, `scripts/v6/*`), unit & subprocess smoke tests (`tests/v6/`), GPU execution on RTX 3090, LaTeX draft scaffolding (`paper/v6/`), and documentation (`docs/v6/`). | `reports/v6/legacy_recompute.json`, `pytest` 100/100 PASS, GPU JSON artifacts & SHA256SUMS under `results/v6-new/`. | Pending final human author review before any arXiv / conference submission. |
| 2026-11-25 | ChatGPT (Protocol Authoring) | Drafting `KRMM_v0.6_Agent_Technical_Requirements.txt` experimental protocol & audit checklist. | Checked against codebase & historical v0.5 JSONs by P0 audit. | Reviewed & provided by human author. |

**Statement:** No AI tool was used to fabricate data, alter historical v0.1–v0.5 artifacts, or bypass evaluation gates. Final scientific responsibility, authorship, and submission approval rest exclusively with the human authors.
