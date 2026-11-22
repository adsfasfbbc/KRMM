# KRMM v0.2 preregistration
Status: RUN, user-authorized 2026-11-22, single existing GPU, 8 hours, 30 launches, +80GB.
Question: Does structured intervenable evidence improve candidate reliability and reduce wrong persistent writes versus candidate_change?

H0-validity: explicit Hornbostel-Sachs family labels, independent wrong-target assignment, original mastery, local damage and oracle writeback are valid.
H1a: structured candidates beat BOTH direct residual and generic generation on paired NEW independent test entities; 95% paired bootstrap lower bound >0 against strongest dev-selected simple method.
H1b: at 50% coverage, ERE beats candidate_change with same candidates and fixed writer; 95% paired lower bound >0.
Primary: successful retained persistent writes / ALL eligible damaged units. Rejected units retain original damaged outcome (no invented success).
Counterfactual writer attempts for all eligible units remain evaluator-only.

P0 dev experiments may iterate damage configuration with new run/config, never rewrite failed trials. No confirmatory holdout access before freeze and power analysis.
G0: raw clean mastery >=75%, eligible local damage >=50% of clean-mastered, >=8 eligible units, oracle success >=75%, retained-token agreement >=90%, no deterministic invertible damage mapping.
Screen templates 0/1; diagnosis 2/3; update 2; independent evaluation 4/5. Do not inspect evaluation for eligibility.
Damage strengths: 24 and 48 AdamW steps, independent balanced association vs erasure (uniform-choice CE); rank8 last4 fixed. Repair remains 32 steps lr0.0005 KL50 factor-L2 0.0001 with 12 retention prompts.
Success: both heldout answers correct, >=0.5 accuracy gain from damaged evaluation, retained-token agreement >=0.9, fresh process probabilities reproduced <=1e-6 max error.
Power: after dev, estimate paired discordance q; n=ceil((1.96+0.84)^2*q/0.10^2), minimum128 independent NEW test entities. If corpus or budget cannot support n, mark G1 INCONCLUSIVE; never lower threshold to obtain significance.
Repeated prompts and damage seeds are NOT independent entities.
No G1 after failed G0; P3/P4 remain blocked until G1a/b pass.
