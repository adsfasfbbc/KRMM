# KRMM v0.6 Frozen Experimental Protocol (`V6_PROTOCOL.md`)

## 1. Brand-New Disjoint Data Pool (`data/v6/pilot_manifest.json`)
- **Total Units:** `50` (`36` known real facts balanced across 3 domains + `6` obscure real unknown controls + `4` fictional unknown controls + `4` ambiguous/unverifiable controls).
- **Disjointness:** `0` normalized-name or alias overlap with all `140` v0.3 + `52` v0.5 = `192` historical entities (`pool_sha256 = 95b845901b2f...`).
- **Blindness:** Public manifest is deterministically shuffled (`seed=2026`), exposes uniform `PublicUnit` schema with zero gold answers or pool roles. Hidden labels live strictly in `data/v6/evaluator_only/pilot_hidden.json`.

## 2. Information Isolation Contract
- `select_action(...)` has **no `oracle_gold` parameter** (verified by `inspect.signature` unit test).
- `Oracle_upper_bound` executes in a separate isolated trajectory on its own copy of `M_d` under the same write cap `B_w = 8`.
- Method processes never receive evaluator hidden labels, clean weights, or hidden locality audit outcomes; only public retain prompts are used for unlabeled self-check and rollback (`min_public_retain_agreement = 0.90`).

## 3. Primary Endpoint & Risk Vector (Section 2.3)
- **Primary Endpoint:** `End-to-End Strict Safe Repair Yield (Y)` evaluated on ONE shared multi-target damaged checkpoint `M_d` with sequential cumulative LoRA updates (`standard_kl`, `rank=8`, `alpha=16`, `last_layers=4`, `repair_steps=32`, `B_w <= 8`, `<= 256` steps).
- **Denominator `D`:** All units in the pool confirmed by paired pre/post held-out audit (`clean_screen_passed AND clean_heldout == 1.0 AND post_heldout <= 0.5 AND delta >= 0.5`), including both `effective_planned_damage` and `collateral_damaged`. If `D == 0`, `Y = NA`.
- **Global Safety Gate:** Explicit `AND` of (1) isolated **fresh-subprocess cumulative reload** (`scripts/v6/verify_fresh_subprocess.py`) verifying `damaged_state` SHA256, label match, and probability diff tolerance (`<= 1e-4`), and (2) **strict per-fact locality audit** (`same_entity_other_relation` zero correct->wrong regression; other groups `acc_drop <= 0.05` and `token_agreement >= 0.90`).
