# KRMM v0.7 Per-Fact Failure Forensics (`FAILURE_CASES.md`)

Directly extracted from `reports/v7/integrity.json` (`results/v6-new/*/fresh_audit_result.json`).

## 1. Qwen2.5-Omni-3B `Oracle_upper_bound` (`se_c2w = 3` out of 72)
Written sequence (steps 1–8): `polystyrene foam cooler`, `titanium watch case`, `cedar shoe tree`, `mahogany chess board`, `narwhal`, `slate chalkboard`, `halibut (D)`, `barracuda (D)`.

| Regressed Entity | Written? | Hidden Immutable Question (Gold = `A: Yes`) | Mechanistic Hypothesis |
|---|---|---|---|
| `halibut` | **Yes (step 7)** | *"Does a halibut breathe underwater using gills?"* | **Same-entity letter-bias bleed:** repairing `halibut -> D (Fish)` without same-entity non-target KL pulls `halibut` Yes/No prompts away from `A`. |
| `barracuda` | **Yes (step 8)** | *"Does a barracuda extract oxygen from water via gills?"* | **Same-entity letter-bias bleed:** repairing `barracuda -> D (Fish)` pulls `barracuda` Yes/No prompts away from `A`. |
| `kalimba` | No (unwritten) | *"Does a kalimba produce sound from the vibration of its own solid metal tines?"* | **Cumulative domain/option logit drift:** 8 sequential `standard_kl` merges with only generic retain shift multiple-choice `A`-vs-`B/C/D` margins on sibling entities. |

All 3 regressed controls are genuinely **immutable facts** (none are `expected_ripple` — repairing a fish's vertebrate class must NEVER make it stop breathing through gills!).

## 2. Qwen2.5-Omni-3B Blind Strategies (`M_evidence_gated` `11`, `M_full` `10`, `B1_generative` `16`)
- Writing multiple `sound_source_mechanism` targets (`koto`, `surdo drum`, `tabor drum`, `ocarina`, plus ambiguous `hybrid wind-string drone box`) caused both **same-entity regressions** (`ocarina`, `koto`, `tabor drum`, `halibut`) AND **7–12 unwritten sibling regressions** across `sound_source_mechanism` (`oud`, `tabla`, `steelpan`, `claves`, `erhu`, `kalimba`, `didgeridoo`) and `vertebrate_class` (`barracuda`, `flamingo`, `bearded dragon`).
- **Implication for P1 Writer Design (`W2`/`W4`):** Pure generic KL (`W0`, `W1`) cannot protect against same-entity letter-bias bleed or sibling domain shift; relation-specific balanced-option public retain prompts (`W2`) + provisional history replay (`W3`) directly target both failure modes without touching hidden audit prompts.
