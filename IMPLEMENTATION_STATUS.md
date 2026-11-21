# Implementation and evidence status

Engineering v0.1 is implemented. Scientific validation is partial; the current decision is PIVOT, with no G1-G3 confirmation.

| Requirement | Implementation / evidence | Boundary |
|---|---|---|
| Known damaged knowledge units | Public entity/relation schema; independent screen/diagnosis/update/evaluation templates | Whole-model fault discovery is out of scope |
| Answer-free reconstruction | Damaged-model atomic facts plus entity-free physical constraints | Real tasks primarily test direct residual access; controlled worlds test complementary composition |
| Internal versus auxiliary evidence | Main 60-entity internal pilot; separate paired raw-media/text/shuffled-image study | Only four real images and one real audio source; no real audiovisual complementarity claim |
| Pre-update repairability | Regularized logistic model, 9 pre-update features, independent calibration, abstention | Full AUROC .80 versus simple candidate-change .90 on 16 eligible test entities; no incremental full-model evidence |
| Selective repair | Full calibrated service and candidate-change baseline entry points | Unknown target truth prevents service-side recovery certification |
| Budgeted parameter writing | Rank 8, last 4 language layers, fixed 32 steps, damaged-teacher KL and size penalty | Per-unit, one-round updates; no sequential/batched knowledge-edit study |
| Locality | 12 non-target prompts, retained-teacher distributions and next-token agreement | Not a comprehensive general-capability benchmark; early-stop-optimized unconstrained baseline not tested |
| Persistence | 39 main-cohort units /117 adapters independently reloaded, exact probability equality | Functional persistence only; auxiliary fresh-process audit is not included |
| Damage | Wrong association and alternate suppression, merged before repair | Suppression is not certified erasure or general unlearning |
| Positive/negative controls | Oracle, no-update candidates, unconstrained writing, selection baselines, controlled key/sham media, opposing-carrier twins | No second real model, no sham real damage, real repeated seed canceled for calibration extension |
| Dataset | 12 dev, 24 calibration including prospective extension, 24 original frozen test entities | Curator labels need expert audit; exciter/resonator ambiguities; substantial original mastery exclusions |
| Statistics | Exact all-eligible denominators, entity intervals, source-disjoint calibration and test | Small conditional pilot. Paired bootstrap resamples fixed selection contributions; does not re-rank at every resample. No population/architecture inference |
| Reproduction | Pinned weights, raw per-unit outputs, config/source hashes, environment, command logs and source bundle | Full frozen training rerun /real multi-seed replication remains missing |
| Literature | Verified nearest-work ledger with pending items | Novelty and publication status PENDING |

The supplied repair modules consume neither evaluator truth nor original reset weights. This is a logical interface audit, not OS isolation. Every failure and interrupted run remains visible in the unified experiment journal.
