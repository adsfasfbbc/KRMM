# KRMM v0.3 Novelty & Related-Work Boundary Ledger

- **Knowledge Editing (ROME, MEMIT, MEND, AlphaEdit):** Assume known target `(subject, relation, new_object)` ground truth for editing. KRMM addresses the *repair* setting where damaged units must first be discovered and target ground truth is hidden from the repair algorithm.
- **Unlearning & Relearning Audits:** Show that suppressed/unlearned knowledge can remain recoverable via fine-tuning or probing; KRMM tests answer-free structured reconstruction from retained premises and decomposed pre-update repairability prediction.
- **v0.1 / v0.2 Baselines:** `candidate_change` remains a mandatory strong baseline (`AUROC 0.90` on v0.1 pilot). Any v0.3 diagnostic method must demonstrate paired gains over `candidate_change` under matched coverage and budget.
