# G3 Report: Generalization & Boundary Stress Tests (Bounded D2 + Cross-Architecture Transfer)

## 1. G3-A: Bounded `D2_suppression` Phase Boundary (`Qwen2.5-Omni-3B`)
- **Artifact:** `results/v5-new/g3a_d2_bounded_results.json`
- **Setup:** `18` balanced brand-new entities, clamped unlikelihood (`max_unlikelihood = 2.0`) + `retain_weight = 50.0`, `damage_steps = 48`.
- **Result:** `15/18` cleanly mastered -> **`0/15` effectively damaged** (`n_strict_eligible = 0`, runtime `324.3s`).
- **Interpretation (Negative Result Preserved):** In v0.3, unbounded unlikelihood suppression caused catastrophic general collapse even at `retain_weight = 50.0`. Clamping unlikelihood at `2.0` prevents general collapse completely, but prevents the target held-out accuracy from dropping below `0.5`. Naive unlikelihood suppression therefore exhibits a sharp phase transition between ineffective (`0/15`) and catastrophic collapse; realistic localized suppression damage requires a dedicated constrained objective beyond clamped unlikelihood.

## 2. G3-B1: Second Model Family — Same 3B Scale (`Llama-3.2-3B-Instruct`, Llama Architecture)
- **Artifact:** `results/v5-new/g3b_llama32_3b_results.json`
- **Setup:** `18` balanced brand-new v0.5 entities (`6` per domain), D1 wrong-fact injection + `standard_kl` LoRA repair, `785.8s`.
- **Funnel:** `18` catalog -> `14` cleanly mastered -> `12` damage-effective -> **`9` strict-eligible**.

| Strategy | Attempted | Strict Restored | Eligible Rate | Attempted Precision | Wrong Writes |
|---|---:|---:|---:|---:|---:|
| `direct` | 9/9 | 0/9 | 0.0% | 0.0% | **9** |
| `generative` | 9/9 | 4/9 | 44.4% | 44.4% | **5** |
| `structured` | 6/9 | 5/9 | 55.6% | 83.3% | 1 (`single_chain`) |
| **`evidence_gated`** | **5/9** | **5/9** | **55.6%** | **100.0% (5/5)** | **0** |
| `oracle` (upper bound) | 9/9 | 9/9 | 100.0% | 100.0% | 0 |

- **Key Cross-Architecture Finding:** Without changing a single rule or threshold, `evidence_gated` transfers directly from `Qwen2.5-Omni-3B` to `Llama-3.2-3B-Instruct`, outperforming `generative` (`55.6%` vs `44.4%`, `+11.1 pp`) while eliminating **100% of wrong writes** (`0/9` vs `5/9` for `generative` and `1/9` for `structured`, specifically gating out the erroneous `single_chain` target `stainless steel thermos`), restoring targets across all 3 domains (`scarlet macaw`, `great horned owl`, `canada goose`, `oak cutting board`, `soprano recorder`).

## 3. G3-B2: Small-Model Capacity Floor (`SmolLM2-1.7B-Instruct`, 1.7B Llama Architecture)
- **Artifact:** `results/v5-new/g3b_smollm2_1.7b_results.json` (also preserved as `g3b_second_model_results.json`)
- **Result:** `2/18` cleanly mastered -> `2/2` damage-effective -> `2/2` strict-eligible (`100.9s`).
- `Direct`: `0/2` (`2` wrong writes); `Generative`: `0/2` (`2` wrong writes); `Structured` & `Evidence-Gated`: **`0/2` attempted, `0/2` wrong writes** (both targets flagged `conflicting_support` and safely abstained); **`Oracle`: `0/2`** (both failed `locality_pass` even with gold answers).
- **Interpretation (Capacity Boundary Preserved):** Below ~2B parameters, the model lacks both multi-permutation mastery (`2/18`) and sufficient parameter capacity for localized LoRA repair (even Oracle fails locality). Crucially, `Evidence-Gated` still maintains **0 wrong writes** under this capacity floor.
