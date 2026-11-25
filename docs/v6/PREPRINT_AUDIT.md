# KRMM v0.6 P0 Preprint Audit (`reports/v6/legacy_recompute.json`)

- **Audited Baseline:** `codex/krmm-v05-blind-repair` (`7a41d02a0b7cfa226d8ef7bc47e13d5402d44c1f`)
- **Audit Script:** `scripts/v6/audit_v5.py --baseline-ref 7a41d02 --out reports/v6/legacy_recompute.json` (read-only on `results/v5-new/*.json`; zero historical files modified).

## 1. Key Findings & Required Claim Narrowing on Historical v0.5 Results
1. **Bug A (`clean_accuracy = 0.0` in v0.5 G1):** Confirmed all `45/45` records in `results/v5-new/g1_discovery_results.json` logged `clean_accuracy = 0.0` because `run_g1_discovery.py` read `clean_m.get("screen_accuracy", 0.0)` instead of `clean_m["screen"]["accuracy"]`. Paired pre/post held-out accuracy cannot be recomputed from v0.5 JSONs alone (`NOT_RECOMPUTABLE`); fixed in v0.6 `krmm/v6/truth_audit.py`.
2. **Bug E (Reversed Confidence & Entropy Beat M2 on v0.5 G1 D1):** Exact CPU recomputation on raw 5N scores (`N=45`, `20` positives) reveals:
   - `reverse_confidence` (`high_confidence`): **AUROC `0.834`**, **AUPRC `0.851`**, **`P@25% (k=11) = 1.000`**
   - `reverse_entropy` (`neg_entropy`): **AUROC `0.852`**, **AUPRC `0.862`**, **`P@25% (k=11) = 1.000`**
   - `m2_reference_free`: AUROC `0.789`, AUPRC `0.643`, `P@25% = 0.727`
   - `full_m2`: AUROC `0.860`, AUPRC `0.696`, `P@25% = 0.727`
   - **Action:** We explicitly include `high_confidence_rev` and `neg_entropy_rev` (direction frozen on dev) in v0.6 G1 and never claim M2 universally beats simple baselines on D1 detection.
3. **Bug F (Cross-Model Overlap & Reload Protocol Mismatch):**
   - Qwen `n=13` and Llama `n=9` share 3 entities (`canada goose`, `great horned owl`, `scarlet macaw`) -> **`19` distinct entities across `22` model-target instances**.
   - v0.5 Qwen reload was in-process PEFT disk reload checking target accuracy only; v0.5 Llama had no reload check. v0.6 unifies both under **true isolated fresh-subprocess cumulative reload** (`scripts/v6/verify_fresh_subprocess.py`).
   - Observing `0/14` wrong writes does **not** prove `0%` error rate: even under i.i.d. assumptions, the one-sided 95% Clopper-Pearson upper bound is **`19.27%`** (`1 - 0.05^(1/14)`).
4. **Bug H (`rule_only` & G2-A Prompt A Accuracy):** In v0.5 G2-A, `rule_only` received empty facts (abstention negative control, not a strong baseline), and generative complete-support prompt A achieved `11/22 (50.0%)`.
5. **Bug J (D2 & SmolLM2-1.7B Scope):** `raw_ul.clamp(max=2.0)` zeroes gradients above 2.0; both D2 (`0/15`) and SmolLM2 (`2/18` mastered, `0/2` Oracle) are documented strictly as tested-configuration failures, not universal phase transitions or parameter-count thresholds.
