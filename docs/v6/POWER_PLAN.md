# KRMM v0.6 Statistical Power & Sample-Size Plan (`POWER_PLAN.md`)

## 1. Historical Funnel & Attrition Estimates
- v0.5 Qwen single-target funnel: `36` catalog -> `29` mastered (`80.6%`) -> `22` damage-effective (`75.9%`) -> `13` single-target locality-eligible (`36.1%` of catalog).
- v0.5 Llama-3.2-3B funnel: `18` catalog -> `14` mastered (`77.8%`) -> `12` damage-effective (`85.7%`) -> `9` strict-eligible (`50.0%`).
- **Lesson:** Planned pool size `N` is NEVER equal to the degraded denominator `D`.

## 2. Pilot Sizing (`N = 50`, `36` Known Real Facts)
- Expected clean mastery (`>=75%`): `~27–31` mastered known facts.
- Planned D1 damage fraction (`0.45`): `~13–14` planned targets + potential uninjected collateral degradation -> expected `D ≈ 10–16` degraded targets per multi-target episode `M_d`, exceeding the `G0-SCIENCE` pilot feasibility floor (`D >= 8`).
- Under `B_w = 8` writes per episode, pilot tests feasibility, end-to-end yield `Y`, actual unknown/healthy false-positive write exposure, and fresh-subprocess reload stability.

## 3. Confirmatory Power Reality Check (Honest Bounds)
- For a paired proportion gain of `20–25 pp` (discordant pairs ~`0.25–0.30`) at `80%` power (`alpha=0.05` two-sided), `n >= 40–60` independent degraded targets across `>= 3` disjoint multi-target episodes/source-groups are needed.
- For sparse wrong-write risk control (`< 5%` one-sided 95% exact binomial upper bound with `0` observed wrong writes), **`>= 59` independent i.i.d. accepted writes** are required (`1 - 0.05^(1/59) = 0.0495`).
- Therefore, **pilot results (`1` episode per model, `D ~ 10–16`, `<= 8` writes) are explicitly labeled pilot feasibility evidence**, NOT confirmatory low-risk statistical guarantees.
