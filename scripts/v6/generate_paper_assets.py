#!/usr/bin/env python3
"""Programmatically generate paper/v6/tables/*.tex and paper/v6/figures/*.png from hashed v0.6 JSON artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "paper" / "v6"
TAB_DIR = PAPER_DIR / "tables"
FIG_DIR = PAPER_DIR / "figures"


def main() -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    g1 = json.loads((REPO_ROOT / "results/v6-new/qwen_pilot_g1_r2/metrics.json").read_text(encoding="utf-8"))
    e2e = json.loads((REPO_ROOT / "results/v6-new/qwen_pilot_e2e_r2/metrics.json").read_text(encoding="utf-8"))
    g2a = json.loads((REPO_ROOT / "results/v6-new/qwen_pilot_g2a_r3/metrics.json").read_text(encoding="utf-8"))

    # Table 1: G1 Discovery
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Equal-budget blind discovery on the 50-entity v0.6 pool (\texttt{Qwen2.5-Omni-3B}, $D=19$: 12 planned D1 + 7 collateral, budget cap $5N=250$).}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Detector & LogQ & Tok & AUPRC & AUROC & P@25\% & H-FPR & ColRec \\",
        r"\midrule",
    ]
    det_names, logqs, auprcs, colrecs = [], [], [], []
    for det, v in g1["budget_curves"]["5N"].items():
        a = v["accounting"]; f = v["full_pool"]
        cr = f["collateral_recall"] if f["collateral_recall"] is not None else 0.0
        det_esc = det.replace("_", r"\_")
        lines.append(
            f"{det_esc} & {a['logical_queries']} & {a['total_tokens']} & {f['auprc_primary']:.3f} & {f['auroc_secondary']:.3f} & {f['precision_at_fixed_k']:.3f} & {f['healthy_fpr']:.3f} & {cr:.3f} \\\\"
        )
        det_names.append(det); logqs.append(a["logical_queries"]); auprcs.append(f["auprc_primary"]); colrecs.append(cr)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "table_g1_discovery.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Fig 2: Detection Cost vs AUPRC & Collateral Recall
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].scatter(logqs, auprcs, s=70, color="#1f77b4")
    for x, y, lbl in zip(logqs, auprcs, det_names):
        ax[0].annotate(lbl, (x, y), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
    ax[0].set_xlabel("Logical Queries (Budget Cap 250)")
    ax[0].set_ylabel("Full-Pool AUPRC (Primary)")
    ax[0].set_title("AUPRC vs. Query Cost")
    ax[0].grid(alpha=0.3)

    ax[1].scatter(logqs, colrecs, s=70, color="#d62728")
    for x, y, lbl in zip(logqs, colrecs, det_names):
        ax[1].annotate(lbl, (x, y), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
    ax[1].set_xlabel("Logical Queries (Budget Cap 250)")
    ax[1].set_ylabel("Collateral Degradation Recall (out of 7)")
    ax[1].set_title("Collateral Recall vs. Query Cost")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_detection_cost.png", dpi=180)
    plt.close(fig)

    # Table 2: True E2E Main Table
    elines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{True sequential multi-target E2E blind repair on \texttt{Qwen2.5-Omni-3B} ($D=19$, $B_w=8$, isolated fresh-subprocess reload audit).}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Strategy & Comm & Raw Rest. & Active & WW & UnkAtt & SE-Reg & Strict $Y$ \\",
        r"\midrule",
    ]
    for strat, m in e2e.items():
        if "error" in m:
            continue
        se_reg = m.get("new_collateral_by_group", {}).get("same_entity_other_relation", 0)
        s_esc = strat.replace("_", r"\_")
        y_val = f"{m['end_to_end_strict_safe_yield_Y']:.2f}" if m["end_to_end_strict_safe_yield_Y"] is not None else "NA"
        elines.append(
            f"{s_esc} & {m['n_committed_writes']} & {m['restored_in_D_raw']}/{m['denominator_D']} & {m['active_repaired_in_D']} & {m['wrong_write_count_committed']} & {m['unknown_write_attempts']} & {se_reg} & {y_val} \\\\"
        )
    elines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "table_e2e_main.tex").write_text("\n".join(elines) + "\n", encoding="utf-8")

    # Fig 3: Risk-Coverage Curve
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, v in g2a["risk_coverage_curves"].items():
        pts = v["points"]
        if not pts:
            continue
        ax.plot([p["coverage"] for p in pts], [p["risk"] for p in pts], marker="o", markersize=3.5, label=f"{k.replace('score_', '')} (AURC={v['aurc_reachable']:.3f})")
    ax.set_xlabel("Coverage (Accepted / D=19)")
    ax.set_ylabel("Selective Risk (Wrong Candidates / Accepted)")
    ax.set_title("Fixed-Candidate Risk-Coverage (v0.6 Pilot, Qwen2.5-Omni-3B)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_risk_coverage.png", dpi=180)
    plt.close(fig)

    # Table 3: Cross-Model Table (populated from llama32_pilot_e2e if complete, else TODO)
    llama_p = REPO_ROOT / "results/v6-new/llama32_pilot_e2e/metrics.json"
    clines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Cross-architecture true sequential E2E persistence on \texttt{Llama-3.2-3B-Instruct} (isolated fresh-subprocess reload).}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Strategy & Comm & Raw Rest. & Active & WW & UnkAtt & SE-Reg & Strict $Y$ \\",
        r"\midrule",
    ]
    if llama_p.exists():
        lm = json.loads(llama_p.read_text(encoding="utf-8"))
        for strat, m in lm.items():
            if "error" in m:
                continue
            se_reg = m.get("new_collateral_by_group", {}).get("same_entity_other_relation", 0)
            s_esc = strat.replace("_", r"\_")
            y_val = f"{m['end_to_end_strict_safe_yield_Y']:.2f}" if m["end_to_end_strict_safe_yield_Y"] is not None else "NA"
            clines.append(
                f"{s_esc} & {m['n_committed_writes']} & {m['restored_in_D_raw']}/{m['denominator_D']} & {m['active_repaired_in_D']} & {m['wrong_write_count_committed']} & {m['unknown_write_attempts']} & {se_reg} & {y_val} \\\\"
            )
    else:
        clines.append(r"\multicolumn{8}{c}{\texttt{NOT\_RUN / IN\_PROGRESS (llama32\_pilot\_e2e)}} \\")
    clines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "table_cross_model.tex").write_text("\n".join(clines) + "\n", encoding="utf-8")
    print("PAPER_ASSETS_GENERATED:", sorted(p.name for p in TAB_DIR.glob("*.tex")), sorted(p.name for p in FIG_DIR.glob("*.png")))


if __name__ == "__main__":
    main()
