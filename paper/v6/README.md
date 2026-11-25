# KRMM v0.6 LaTeX Preprint Source (`paper/v6/`)

- **Main entry point:** `paper/v6/main.tex`
- **Bibliography:** `paper/v6/refs.bib` (verified URLs; unverified author lists explicitly marked `PENDING_VERIFICATION` — never fabricated)
- **Generated tables:** `paper/v6/tables/table_g1_discovery.tex`, `table_e2e_main.tex`, `table_cross_model.tex`
- **Generated figures:** `paper/v6/figures/fig2_detection_cost.png`, `fig3_risk_coverage.png`
- **Regeneration command:**
  ```bash
  PYTHONPATH=. python scripts/v6/generate_paper_assets.py
  ```
- **Compilation note:** `pdflatex` / `latexmk` are not installed on the GPU compute node (`sc01:30436`). To compile locally or on Overleaf:
  ```bash
  pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
  ```
