# KRMM v0.5 Unified Evidence Pack (Paper-Ready Numbers)

All numbers below come strictly from verified GPU JSON artifacts on **52 brand-new entities (0 overlap with v0.3)** under branch `codex/krmm-v05-blind-repair`.

## 1. Artifact Registry
| Stage | Artifact Path | Key Headline |
|---|---|---|
| Blind Pool Audit | `results/v5-new/v5_blind_pool_manifest.json` | 52 brand-new entities (`36` known + `12` unknown + `4` ambiguous), `v3_overlap_count = 0` |
| G1 Blind Discovery | `results/v5-new/g1_discovery_results.json` | At `5N` (`B=225`, `N=45`): `m2_reference_free` AUROC `0.789` (`unk_FPR=0.167`, 3-cls `0.667`), `full_m2` AUROC `0.860` (collateral recall `3/3`) vs confidence `0.166` / entropy `0.148` |
| G2-A Mechanism & Interventions | `results/v5-new/g2_mechanism_results.json` | `n=22`: `evidence_gated` precision `93.75%` (`15/16`) vs CoT `9.1%` / generative `50.0%`; model-level sham invariance `100.0%`, key drop `-0.145`, CF drop `-0.256` |
| G2-B/C Selective Repair (`Qwen2.5-Omni-3B`) | `results/v5-new/g2_repair_results.json` | `n=13`: `Direct 0/13 (13 ww)`, `Generative 8/13 (5 ww)`, `Structured 11/13 (1 ww)`, **`Evidence-Gated 9/13 (0 ww, 9/9 attempted)`**, `Oracle 13/13`; `100%` same-entity preservation & PEFT reload |
| G3-A Bounded D2 (`Qwen2.5-Omni-3B`) | `results/v5-new/g3a_d2_bounded_results.json` | `15/18` mastered, `0/15` damaged at `clamp=2.0, retain=50.0` (honest phase-transition boundary) |
| G3-B1 Cross-Arch 3B (`Llama-3.2-3B-Instruct`) | `results/v5-new/g3b_llama32_3b_results.json` | `n=9`: `Direct 0/9 (9 ww)`, `Generative 4/9 (5 ww)`, `Structured 5/9 (1 ww)`, **`Evidence-Gated 5/9 (0 ww, 5/5 attempted)`**, `Oracle 9/9` |
| G3-B2 Capacity Floor (`SmolLM2-1.7B-Instruct`) | `results/v5-new/g3b_smollm2_1.7b_results.json` | `2/18` mastered; `Oracle 0/2` (locality fail); `Evidence-Gated 0/2` wrong writes (safe abstention) |

## 2. Combined 3B Cross-Architecture Selective Repair Summary (`Qwen2.5-Omni-3B` `n=13` + `Llama-3.2-3B-Instruct` `n=9` = `22` Strict-Eligible Targets)
- `Direct`: `0/22 (0.0%)` strict restored, `22` wrong writes (`100.0%`)
- `Generative`: `12/22 (54.5%)` strict restored, `10` wrong writes (`45.5%`)
- `Structured`: `16/22 (72.7%)` strict restored, `2` wrong writes (`9.1%`)
- **`Evidence-Gated`**: **`14/22 (63.6%)` strict restored**, **`0/22` wrong writes (`0.0%`)**, **`14/14 (100.0%)` precision on every attempted parametric write!**
- `Oracle`: `22/22 (100.0%)` strict restored, `0` wrong writes
