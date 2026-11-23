# KRMM v0.5 Novelty & Positioning Ledger

| Dimension | Oracle Knowledge Editing (ROME / MEMIT / LoRA-Edit) | Self-Consistency / Generic CoT | KRMM v0.5 (Blind Evidence-Gated Repair) |
|---|---|---|---|
| Target Damage Identity Known? | Yes (given) | Assumed given | **No (Blind Discovery, G1)** |
| Gold Target Answer Given? | Yes (Oracle) | No | **No (Zero-Oracle Reconstruction, G2)** |
| Selective Writeback / Abstention? | No (always writes) | No | **Yes (8-class Support Structure Gating)** |
| Persistent Weight Update? | Yes | No (prompt-only) | **Yes (LoRA `standard_kl` + PEFT Reload)** |
| Locality Evaluation | General / neighbor | N/A | **5-Gate Locality Graph (separates ripple vs collateral)** |

## Core Scientific Contribution (Verified Scope)
KRMM v0.5 demonstrates on **52 brand-new entities (0 overlap with v0.3)** that when wrong-fact corruption (`D1`) makes a model *overconfident* in false answers (causing confidence/entropy detectors to collapse to AUROC `< 0.20`), combining **reference-free constraint/path discrepancy detection** (`AUROC 0.789–0.860`), **explicit 8-class support-structure gating** (`93.75%` candidate precision; `0/13` wrong parameter writes), and **KL-regularized LoRA writeback** enables safe, persistent, blind knowledge restoration — while transparently characterizing the coverage-vs-safety trade-off, D2 suppression phase boundary, and small-model capacity floor.
