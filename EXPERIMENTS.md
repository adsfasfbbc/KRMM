# KRMM 实验记录 / Experiment journal
> 最终摘要：本批完成 20 次受控运行，真实知识共 60 个实体。封存测试 24 个实体中 16 个形成合格损伤，持久修复 11/16；监督上界 16/16。复杂诊断 AUROC 0.80，简单候选变化基线 0.90，未证明复杂诊断的增量价值。相关真实媒体 2/4，内部先验/置换图像/匹配文字均 3/4，未证明媒体优势。39 个主实验目标的 117 个适配器在新进程中完全复现。所有失败与排除保留，G1–G3 未获验证，研究结论为 PIVOT。

All entries are exploratory until a frozen, independent confirmatory experiment is explicitly declared. Raw outcomes, failures and exclusions remain in the repository. Dates use Asia/Singapore.

## 2026-11-22 — P0 / infrastructure and protocol

- User authorized one RTX 3090, at most 8 wall-clock hours, 20 experiment launches, 80 GB additional storage; no local project checkout or environment. Runtime and code live only on the supplied server and GitHub.
- Empty public repository; GPU RTX 3090 24 GiB, torch 2.12.1+cu130, Python 3.12.13. Dedicated server venv uses system torch.
- Hugging Face direct access timed out. hf-mirror.com metadata reachable; download acceleration being configured. No credentials or subscription content are committed.
- Research question: predict unit-level success of a fixed budgeted persistent repairer before updates without target truth. Internal-prior and auxiliary-evidence settings must be separate.
- First test: G0 clean mastery, local targeted corruption, retained knowledge, oracle repair and parameter reload. Failure blocks scientific promotion, and triggers an explicitly logged engineering trial.
- Real knowledge and controlled synthetic worlds are separate evidence tracks. Text-only feasibility cannot establish multimodal complementarity.

## S01 — controlled neural-world sanity experiment (protocol controlled-v1)

- Hypothesis: known evidence structure separates reconstructable targets from ambiguous/random targets under a fixed 24-step shared linear residual update.
- Design: seeds 17/29/43; 128 entities/world; 48 dev, 32 calibration, 48 test per world. Generated 16x16 images and nonverbal sine-wave audio. Internal atomic neural memories versus auxiliary encoders. Four information regimes. All units receive counterfactual repair, and serialized weights are reloaded with no evidence.
- Result: clean target/image/audio accuracy 100% in all three worlds. 144 test entities across 3 worlds; internal candidate/repair success 68.75%; auxiliary 70.83%. At half coverage learned selector achieves 50% successful repairs over ALL eligible test entities. Confidence selector 35.42%.
- Critical audit: strongest single support feature ALSO achieves 50%. Confidence is constant by construction; known information regime supplies privileged structural information. Thus this is NOT evidence for incremental value of a learned diagnostic. Shared update succeeds whenever candidate correct, so H3 is not discriminated.
- Intervention limitation: key-deletion support values are structural annotations, not measured causal media interventions. Twin check establishes identity of remaining weights after removing a dedicated carrier, not independently learned opposing pretrained worlds.
- Cost: 90.09 seconds GPU execution. Raw files: results/synthetic-v1/seed-*.json; summary.json; audit.json. No gate promotion: main real-model G0 still pending. Decision: continue real-model feasibility; retain controlled results as sanity evidence only.

## E02 — dependency compatibility repair (engineering, no research gate)

- transformers 4.51.3 did not contain qwen2_5_omni. Import failed before any real-model experiment.
- Upgraded server-only environment to 4.57.6; Omni import now succeeds. No result or scientific hypothesis changed.
- Success is scored solely on question variants 2/3, both withheld from the two training paraphrases. Screening uses all four. Frozen test entities remain untouched.

## E03 — allowlist rejection before S02 launch

- researchctl rejected krmm.synthetic_v2 because only krmm.synthetic was allowlisted. No child experiment was executed and no scientific result was produced.
- Protocol v1.1 adds the specific preregistered module executable prefix; metrics, split and gate thresholds unchanged. S01 evidence remains linked to its original protocol hash. Explicitly retry S02 under v1.1; never relabel rejected launch as successful.

## S02 — shared-weight corruption and measured media interventions

- Preregistered design: configs/synthetic-v2-design.json before launch. Only conceptual change to repairability task: damage now uses 24 actual shared target-head parameter updates with retention KL, rather than a fixed wrong-output carrier. Other training, worlds, splits and 24-step repair budget stay fixed. Real media-removal/swap tests added as a separately reported mechanism measurement.
- All 128 entities/world satisfy corruption eligibility in all 3 seeds; clean, image and audio tasks remain 100% mastered. 144 test entities across 3 independent worlds. Internal persistent repair 68.75%; auxiliary 70.83%; oracle 100%. Full diagnostic at half coverage still ties support-only baseline at 50% successful repairs over all eligible units. H1/H3 remain unsupported.
- Measured complementary-media intervention accuracy (seeds 17/29/43): full evidence 100/100/100%; no-image 65.625/62.5/53.125%; no-audio 53.125/40.625/50%; swapped audio 59.375/56.25/37.5%; sham background 100/100/100%. Supports causal evidence dependence within this known-rule synthetic task only.
- Computation 103.94 seconds; raw per-unit records and world-level bootstrap uncertainty in results/synthetic-v2/. Three worlds are the independent replication units, not 384 independent pretrained models.
- Gate: no promotion to real-model G1 or G2. Decision: use this as controlled mechanism validation, proceed to genuine pretrained-model G0.
- Important twin-control limitation carried forward: opposing learned target carriers are not independently trained; identity check alone is not the full proposed twin-world experiment.

## E04 — model/configuration and download preflight

- Pinned official model has three weight shards totaling 11,972,663,208 bytes. Direct mirror is faster for bulk weights; working v2ray subscription node is used for blocked metadata/GitHub/media.
- Parallel range download validates official LFS SHA256 for every shard; readiness sentinel is written only after full validation. No training starts from partial files.
- Static model-config inspection found language layer count under thinker_config.text_config, not the top thinker config. Corrected adapter layer selection before any weight-edit run. Cache disabled in language config during optimization.
- Wikimedia metadata search returned HTTP 429; failed attempts retained. Known previously resolved source URLs yielded four images and one nonverbal violin clip. These support an I/O smoke only; no media complementarity claim for real knowledge, and no media binaries are redistributed.

## R01 — Qwen2.5-Omni-3B clean-mastery screen

- Immutable source e890f72; official weight checksums verified. 12 development entities, 4 paraphrases each, entity-disjoint from calibration/test. Closed-choice likelihood evaluation, no updates.
- Result: all 12 entities correct on all 4 prompts (100% mastery); peak allocated VRAM 11.16 GiB. This is a clean-screen feasibility result, not repair success. Generated voice disabled; processor's generic-system-prompt audio-output warnings are non-fatal and preserved.
- G0 not yet assessed: targeted local damage and oracle persistent repair still required. Next fixed trial: balanced first four development entities, 24 damage steps / 32 repair steps, rank 8 on last four language layers, lr 5e-4 and damaged-teacher retention KL weight 1.

## R02 — genuine-model G0 targeted damage (FAIL)

- Fixed 24-step rank-8 damage with KL weight 1 changes violin/flute/snare drum to wrong answers, but full-vocabulary next-token retention agreement is 0.75 / 0.8333 / 0.8333. All fail the preregistered >=0.9 locality requirement. Xylophone is excluded for a tied A/D screening prompt (3/4 correct).
- No eligible target: oracle repair was not run; G0 FAIL, not evidence of impossibility of knowledge reconstruction. No G1 progression.
- Correction to R01 wording: clean_mastery_rate used the two unseen prompts, not all four screening prompts. Raw files expose this; xylophone was not stably mastered. v1.3 corrects the summary calculation to strict screening; original results are immutable. No threshold or holdout changed.
- Next exploratory trial REAL2 changes one scientific factor: retention KL coefficient 1 -> 50 for damage and repair, leaving lr, steps, layer/rank, entities and prompts fixed. Engineering addition saves damaged selected weights and teacher logits for a future independent-process persistence audit. These snapshots are evaluator-owned and never supplied to candidate diagnosis.

## R03 — stronger retention coefficient alone (FAIL)

- KL coefficient 50 still yields no eligible target. No oracle or repair-success conclusion is possible.
- Suspected optimizer issue: cyclic single-retain sampling ends with several unrelated questions, allowing earlier neighbor tasks to drift even when the last reported KL is tiny. Next trial REAL3 changes only retention sampling from one cyclic question to all 12 retention prompts each step, with coefficient 50 and all other scientific settings fixed.
- Engineering: remove unnecessary input-gradient hooks because no gradient checkpointing is used; frozen prefix need not backpropagate. Add per-retain raw token, choice distributions and KL diagnostics without changing the predeclared strict retention gate.
- Clean screening correction: exactly 10/12 development entities, not 12/12, pass all four prompts. Two unseen prompts alone were 12/12. Original R01 files retained.

## E05 — interrupted SFT serialization metadata lookup (engineering failure)

- REAL3 batch-12 reached an eligible violin and produced evaluator-owned damaged weights / teacher logits, but PEFT save_pretrained attempted an unnecessary online config lookup at an invalid thinker subpath. Repeated network retries stalled serialization; no completed target-level repair result is claimed.
- Terminated only the validated experiment child; partial artifacts and terminal logs preserved. This interrupted launch counts toward the 20-launch budget.
- Engineering correction: strictly offline inference/training after verified model download; explicitly disable embedding serialization because embeddings are frozen; save correct local base-model metadata. Scientific config and all gates unchanged. Retry into a NEW output directory with identical seed/config.

- Immediate E05 retry was rejected by the workspace lock while the terminated child was being reaped. No child launched. Confirmed prior run FAILED and lock released before retrying; lock was never bypassed or removed manually.

## R04 — full retention minibatch (G0 FAIL; one positive oracle writeback)

- Offline retry completed. Strict clean mastery 3/4; eligibility 1/4. Violin is wrong after damage with retention 11/12, and oracle repair restores 2/2 unseen prompts with 12/12 damaged-teacher retention after adapter reload. Flute/snare remain correct under 24-step damage, so are ineligible; xylophone fails strict screening.
- Violin's two raw atomic facts correctly mention vibrating strings, but generative integration selects D (solid body); constrained pseudo-label repair persistently writes D and therefore fails. This directly separates retained information, candidate integration and writeback. No claim that candidates are reliable.
- G0 FAIL because eligibility rate 0.25 <0.5; no G1 promotion. Next REAL4 changes damage_steps 24 ->48 only, keeping 32 repair steps, lr/rank/layers/KL/batch/entity prompts fixed.
- v1.4 only adds an independent-process persistence audit command; semantic metrics and thresholds unchanged. Planned later candidate-policy experiment will compare explicit generic vibration constraints to generative integration, using the same atom calls and writer; it will be a new exploratory trial, not a retroactive rescue of this result.

## R05 — 48-step targeted damage (G0 mechanical PASS; proposed unsupervised repair FAIL)

- 3/4 dev units eligible, strict mastery 3/4, oracle persistent success 3/3, peak allocation 11.16 GiB. G0 numeric checks pass and raw outcomes support editor feasibility. Candidate generation succeeds 0/3; constrained pseudo-label persistent repair succeeds 0/3. Strong edit leaves correct physical atom statements accessible but generative integration still selects the contaminated category.
- Scientific boundary: this validates solvable local damage and writeback capacity; it does not validate the proposed unsupervised repair or diagnostic contribution. No automatic G1 claim.
- Next REAL5 is preregistered in configs/constraint-policy-design.json. Change only candidate integration to generic physical-source rules, retaining the same atom/probe/writer budgets. Rules do not contain instrument names and preserve conflict/missing-support flags. This is a strong structured baseline, not a claimed novel algorithm.

## M01 — real media I/O smoke (engineering FAIL)

- Failed at importing qwen_omni_utils because audioread was absent. No media prediction was produced. Installed audioread in the server venv; import preflight before retry. Failed trial and source logs retained; counts against launch budget.

## R06 — generic physical constraint integration (G0 PASS; exploratory candidate gain)

- Same 3/4 eligible development units, editor and query budgets as R05. Candidate correctness and persistent constrained success improve from 0/3 to 2/3; oracle 3/3. This is candidate generation gain, not proof of diagnostic selector gain. Failure preserved; rules remain entity-free and are not retuned per target.
- Six meaningful parser contracts pass (negation, missing support, conflict retention, no entity shortcut, and air transmission versus primary vibration).
- Protocol audit before any calibration/test use: initial templates separated training from updates but diagnosis also queried an evaluation paraphrase and clean screening reused evaluation templates. Treat ALL R01-R06 results as exploratory feasibility only. v2 will use independent clean-screen templates 4/5, diagnosis 0/1/6, update 0/1, final evaluation 2/3. Frozen entity partitions and thresholds remain unchanged. No confirmatory claim is made for the earlier overlapping-template experiments.
- Feature audit: symbolic-policy evidence deletion/sham comparisons must apply the SAME symbolic policy to all branches, not compare symbolic full support to generative ablations. Correct before fitting a diagnosis model or opening the holdout.

## M02 — real media I/O retry (PASS as I/O only)

- Six clean-model identification conditions across 4 entities and 5 source media; 5 correct. Violin image/audio/joint, flute image and snare image correct; xylophone image is misclassified. Not six independent sources, and only one audio source. No genuine repair or complementarity conclusion from this smoke.

## P01 — independent-process persistence audit (PASS on 3 units)

- Loaded official base weights, saved merged damaged selected weights, and all 9 serialized adapters in a new Python process. Every choice probability exactly matches the training process (maximum absolute difference 0). Correct and incorrect repairs both persist. This does not establish restoration of original internal representations.

## V2 freeze — before calibration/test entities are opened

- New version prospectively separates screen/diagnosis/update/evaluation templates as documented in protocol v2. Earlier overlapping-template results remain exploratory.
- Eligibility now uses independent damaged screening prompts, not heldout answer correctness; all eligible units remain in the denominator, including cases where heldout performance did not degrade. Success still requires >=0.5 gain and >=0.75 heldout accuracy plus >=0.9 retention.
- Same-policy symbolic support/ablation/sham transformations corrected before diagnosis training. No per-entity rule retuning.
- Cost instrumentation correction: PEFT bypasses the outer forward hook. Attach the hook to the inner language decoder; record actual decoder calls/tokens and per-unit deltas, plus separate diagnosis/training times. Earlier forward-call fields undercounted wrapped PEFT calls; loss steps and wall times remain valid.
- Additional frozen code paths: alternate targeted suppression loss (damage evaluator alone may use truth), optional auxiliary evidence for atomic extraction. These do not alter the internal/default path. No target truth is accepted by diagnosis/repair.
- Next: 12 development units under v2; only advance entity-heldout pilot evaluation if feasibility remains valid. Holdout comparison is a small pilot, not confirmatory publication evidence.

- Before calibration/test access, add the strongest obvious simple predictor: whether the candidate differs from the damaged direct answer. A candidate that repeats the existing wrong answer cannot show a repair gain. This is computed entirely from pre-update outputs. Share its feature transform between analysis and deployment; compare it independently and allow it in strongest-single selection. This strengthens the alternative explanation rather than presuming a complex diagnosis is needed. Fit/threshold rules remain fixed; no holdout labels inspected.
- Remaining bounded queue and all planned transfer/control conditions are preregistered in configs/remaining-batch-plan.json. They do not imply G1/G2/G3 success. v2 source hashes will be checked again before opening the holdout; core update/evaluation logic remains frozen.

## R07 — v2 independent-template development (G0 PASS)

- 12 development entities; 11/12 pass independent clean-screen templates 4/5. Ten units meet damaged-screen/locality eligibility. All 10 hidden-evaluator oracle repairs succeed; generic-constraint candidates and constrained persistent repairs succeed 8/10. Peak allocation 11.16 GiB.
- Hidden evaluation templates 2/3 are not used by the diagnosis (0/1/6), updates (0/1) or eligibility (4/5). Outcomes include both success and failure, so diagnosis training is possible if calibration/test eligible counts suffice.
- Final pre-holdout source extension is the already preregistered derived candidate-change feature and independent simple baseline. Raw diagnosis/update/evaluation code is unchanged from the v2 development trial; only analysis was extended. Generic rules are unchanged. v2.1 adds executable prefixes for the preregistered auxiliary, twin and answer-free service entry points; gate thresholds unchanged.
- Freeze all source/config hashes now. Next calibration and test entities are evaluated under the same fixed model/damage/writer budgets. No modifications based on their labels are permitted.

## R08 — original entity-held-out calibration (G0 coverage FAIL)

- 12 entities: 7/12 pass clean screening and all 7 yield local damage. Oracle 7/7; candidate/persistent repair 4/7. Clean mastery 58.33% fails preregistered 75% coverage check; original failure retained. No learned diagnosis is fitted because eligible calibration n=7 is below the prespecified 8 minimum.
- This reveals limited model/benchmark coverage on independent entities, not a broken optimizer. Do not lower eligibility, calibration count or success thresholds. All originally eligible outcomes remain in future reporting.
- Prospectively amend to v3 BEFORE ANY TEST QUERY: add 12 new calibration entities (viola, harpsichord, erhu; piccolo, cornet, pipe organ; taiko, surdo, mridangam; steelpan, kalimba, vibraslap). Total real units=60; original 24 test entities unchanged. Do not tune rules or writer. The original calibration failures are neither removed nor relabeled.
- Use one originally planned real seed29-repeat launch for this extension, preserving the 20-launch cap. Real multi-seed replication is therefore missing and must be disclosed. Extension's label provider exists only in its evaluator driver; no labels are passed to diagnosis/repair. Original frozen source/config hashes checked intact.

- Pre-test reproducibility amendment v3.1: equal-score selectors previously inherited filesystem file order. Before any final-test query, fix entity order to the existing SHA256-derived UID, then stable descending score. This order is independent of labels and avoids different tie selections after a clone. C, fitting, success metrics, coverage, prompts, neural code and writer remain unchanged. Record the analysis-only hash amendment explicitly; frozen checks continue rejecting all undeclared changes.

## R09 — prospective independent calibration extension (sample target reached; coverage G0 FAIL)

- 12 additional entities, 6 pass clean screening and all 6 yield local damage. Oracle 6/6; candidate/persistent constrained repair 3/6. Together with the original calibration cohort: 13 eligible calibration entities, above the fixed minimum of 8; both success/failure classes are present.
- Extension clean mastery is 50%, again failing the unchanged >=75% coverage gate. This is an important negative result, not hidden or repaired by sample extension. Candidate/writeback analyses remain conditional on initially mastered, eligible damaged units. No broad model-coverage or G1/G2 claim follows.
- Continue only the already frozen final-test G0 characterization and descriptive conditional prediction pilot. This is not promotion to G1 after the failed coverage checks. Original 24 test entities, all neural source/config hashes, success thresholds, rule set and fitting rules remain fixed; pre-test integrity record saved. No second extension or test-informed tuning is permitted.

## R10 — frozen final-test characterization (coverage G0 FAIL; conditional repair 11/16)

- All 24 original test entities evaluated once under frozen neural/config/analysis rules. Sixteen pass independent mastery/damage/locality eligibility; 8 fail original mastery. Candidate correctness and persistent repair 11/16 (68.75%); oracle 16/16. No target labels enter repair.
- Clean mastery 66.67% fails unchanged >=75% coverage criterion. No G1 promotion, second test tuning or sample extension. Pre/post-test frozen source/config checks pass, including the declared label-independent pre-test tie-order amendment.

## A01 — fixed CPU diagnosis fitting / descriptive held-out analysis

- Preregistered single logistic model C=1 with 9 pre-update features, 10 development /13 calibration /16 test eligible entities. Platt calibration uses the disjoint calibration cohort only; threshold is calibration-score median. This completes the existing frozen experiment's postprocessing, not a new model-query trial or hyperparameter search.
- Test AUROC 0.80, AUPRC 0.8088, Brier 0.0930. Confidence AUROC 0.4727; strongest single candidate-change flag AUROC 0.90. Thus current evidence does NOT support incremental value over the strongest simple predictor.
- At 50% coverage, full predictor repairs 7/16, confidence 5/16, candidate-change 8/16. Paired full-minus-confidence delta +12.5 pp, entity bootstrap 95% CI [-25.0,+43.75] pp; full-minus-strongest single -6.25 pp, CI [-31.25,+12.5] pp. Neither establishes superiority.
- Calibration-median deployment threshold 0.613438 accepts 12/16 test units and retains all 11 successful counterfactual repairs, with one accepted failure; this is a small conditional pilot, not certified general calibration.
- Constrained/unconstrained writers share candidates and 32 steps. Constrained persistent success 11/16; unconstrained success under the joint retention requirement is 0/16. This supports the usefulness of the combined retention/size constraints in this regime, not a uniquely isolated KL effect.
- All correct candidates succeed under the constrained writer; H3's additional writeability discrimination remains underidentified. No test-informed model, rule, feature or threshold changes follow.

## E06 — auxiliary checkpoint-hash preflight

- bfloat16 tensors cannot be passed directly to NumPy in the paired weight-hash helper. Added a separate engineering wrapper that hashes dtype/shape and raw uint8 bytes. Frozen auxiliary/neural/analysis files are unchanged; no prediction or outcome changes. Tested deterministic equality and changed-weight inequality on small CPU bfloat16 tensors before the auxiliary experiment. This is not a new GPU trial.

## P02 — complete primary-cohort fresh-process audit (PASS)

- All 39 eligible units across development, original calibration, prospective calibration extension and frozen test; 117 constrained/unconstrained/oracle adapters. Every target choice probability exactly matches the training process (maximum absolute difference 0). The same new process reconstructs each exact damaged checkpoint and supplies no auxiliary evidence at evaluation.
- This includes persistent wrong answers and failed unconstrained-retention outcomes. It validates reload behavior, not candidate correctness or restoration of original representations.

## R11 — independent targeted-suppression damage on development (G0 PASS)

- Fixed predeclared alternate suppression loss; 12 development units, 11 initially mastered, 9 local eligible suppressions. Constrained candidate/persistent success 6/9; oracle 9/9; peak allocation 11.16 GiB.
- This is an exploratory alternate-mechanism feasibility check on previously used development entities, not independent cross-damage G2 confirmation. No claim of complete knowledge erasure, no test-informed tuning and no classifier refit.

## Deployment and reporting verification

- All 16 exported test probabilities exactly equal the answer-free service's predict() output from the saved calibration artifact. Model artifact and numeric coefficients exported without another fit or model query.
- Added the preregistered candidate-change baseline as a separate deployment entry point. It returns no invented probability, reuses the tested constrained writer, reloads the attempted adapter, rolls back retention violations and labels target recovery unverified. No extra GPU experiment was launched for this wrapper; only the existing selector/serialization primitives and CPU contracts are verified.
- Reported paired intervals resample fixed per-entity selection contributions, rather than re-ranking each bootstrap batch. Treat them as small fixed-batch descriptive uncertainty, not confirmatory fixed-coverage inference. No interval or frozen analysis is changed after seeing the test.

## M03 — paired real-media / shuffled-image / matched-text repair (no media benefit)

- Four development entities; all four eligible in every condition. Per-entity RNG reset and raw merged-weight hashing confirm identical damaged parameters across conditions. Same candidates-to-writer interface and 32-step constrained writing budget; auxiliary evidence appears only during atomic extraction and is removed during writing/evaluation.
- Internal prior: 3/4 candidate/persistent successes. Related raw media: 2/4. Shuffled image: 3/4. Information-matched text: 3/4. One real audio source only; shuffled-image control keeps original audio. No real multimodal or sound-specific advantage is supported, and irrelevant media can improve an individual outcome. Actual raw observations/outcomes are preserved.
- 495.62 seconds. The paired evidence study is exploratory and development-only; its adapters are reloaded in-process, not part of the prior 39-unit fresh-process audit. No G2 promotion or post-test model/rule change.

## S03 — independently learned opposing-carrier twin worlds (information-boundary PASS)

- Seeds 17/29/43, 256 paired entities/world. Two original neural models independently optimize opposite random target carriers to 100% target accuracy while sharing identical retained image/audio networks. Completely removing both dedicated carriers leaves byte-identical damaged weights and identical allowed evidence, with opposite evaluator truths.
- Identical repair views produce identical predictions; no target is correct in both opposing worlds, and mean accuracy over the pair is exactly 50%. Retained image/audio tasks stay 100%. This is an exact synthetic indistinguishability construction, not a claim of complete erasure in real pretrained models or an assertion that a chance guess can never be correct.
- 3.52 seconds. This supplies the full opposing-learned-carrier negative control that the earlier S01/S02 identity sanity check did not supply.

## Batch close — 20 recorded launches completed

- All 20 authorized model-experiment launches ended; interrupted metadata lookup and missing-dependency failures remain in the run archive. Pre-launch allowlist/lock rejections are separately logged and did not start model children. No further model experiment is launched.
- Real data: 60 unique entities, 40 independently mastered, 39 eligible primary damaged units. Constrained persistent success: 26/39 overall; frozen-test success 11/16 eligible (11/24 originally screened). All 39 oracle controls succeed. All 117 primary adapters reproduce exactly in a fresh process.
- Full diagnostic has no demonstrated increment over the strongest simple predictor. Coverage gates fail on independent calibration/test cohorts. Real media show no benefit. H3 is underidentified under the constrained 32-step writer. G1-G3 remain unconfirmed; decision PIVOT, with no novelty/publication claim.
- Nineteen named unit-test contracts pass, plus dtype/hash, selector and exact deployed-score preflights. No local project checkout or environment was created. Source/config integrity remains intact. All source, raw per-unit outputs, configs, command logs, figures and negative evidence are synchronized before root shutdown.


# KRMM v0.2 — new batch 2026-11-22

User-approved 8h / 30 launches / +80GB; deadline15:23:25 UTC. New server RTX4090D24GB. Old source and results preserved. Target P0-P2; no real AV claim before G1. Model download uses verified official hashes. Protocol docs/v2/HYPOTHESES.md.

Setup: dependencies installed, public GitHub cloned; museum canonicalAPI403, official alternateAPI200 with442 instrument rows. Raw descriptions remain cloud-private; label metadata only enters evaluator dataset.

## v02-cpu-audit-01 — semantic and boundary contracts
Design: synthetic typed facts, source spans, role distinctions, key deletion/sham material, duplicate provenance, random wrong-label mapping over3200 artificial units, frozen-source checks. Fixed source1aacf2d. Result:13/13 pass,6.002s runner duration. This validates implementation contracts only; empirical reconstruction and independent expert annotation remain unproven. An initial preflight rejected an overlong hypothesis identifier before launching; corrected safe ID, no failed execution hidden.

## v02-survey-dev-01 — FAILED dependency preflight
Design:32 prospective dev entities, screen0/1 and independent4/5. Result: no entity queried; transformers4.57.6 loader requires spk_dict.pt even when speech is subsequently disabled. Original downloader skipped it. Failure receipt and full traceback preserved. Fix: downloaded revision-pinned259544-byte speaker dictionary, SHA2566a05609b28f5d42b7b748f0f07592545c8f1f6885b9ae8fff64baf56e86b2a18; include it in v0.2 downloader. No prompt/label/metric change. Restart as separate launch.

## v02-survey-dev-02 — G0 FAIL
Design unchanged after missing file fix.32 newdev entities; templates0/1 screen,4/5 independent. Results:10/32 stable screen(31.25%),7/32 independent(21.875%), peak11.158GiB. Baseline gate requires75%; G1 BLOCKED. Every idiophone failed stable screening; severe class/prompt/knowledge limitation. Next P0 diagnostic separates taxonomy wording from entity unfamiliarity using paired definitions, permutation, olddev controls. Dataset/splits and thresholds remain fixed; no newtest touched.

## v02-competence-01 — entity familiarity failure persists
44 entities (32 newdev +12 olddev engineering), four paired prompt conditions and fixed label permutation. Newdev stable accuracy: bareHS10/32, definedHS10/32, permuted14/32, legacy7/32. Olddev:11/12,11/12,10/12,10/12 respectively. Mean choice mass>0.996 in all conditions, so low answer-letter mass does not explain failure. Definitions do not improve newdev; rare-entity hallucinations (e.g. nkangala described as wing vibration) suggest unfamiliarity/homonym errors. Label permutation sensitivity is a further limitation. G0 remainsFAIL; no threshold/prompt tuning to claim pass. Next planned P0 association damage/oracle assay retains original task and all32dev including exclusions, solely to diagnose conditional writer feasibility.

## v02-association24-dev-01 — randomized local damage, G0 FAIL overall
32 raw dev entities,10 stably screened. All10 suffered effective local damage after24steps; retention minimum11/12. Oracle persistent success8/10; structured candidates correct4/10 with6/10 coverage; generative6/10. Fresh-process serialization checks retained. Raw mastery10/32 remainsbelow75%, so G1blocked. This cohort includes initially wrong independent answers; strict restoration flag prevents counting their successful induction as restoration.

## Canonical normalization replay preregistration
Model sometimes emitted semantically equivalent but out-of-schema values such as primary_oscillator=membrane. Normalize ONLY typed JSON values through generic aliases; never search raw prose or look up entities. Role test: primary membrane->C, resonator membrane->no candidate; reed alone ambiguous; negated strings ignored.4/4 pass. New replay preserves same damage snapshots/raw probes, reinitializes every writer branch withseed107, and evaluates all candidates regardless of later selection. It is P0 engineering, not G1 evidence.

## v02-canonical-replay-01 — typed normalization, common writer seed
Same10 eligible damage snapshots/probes; all branches seed107,32steps,12retention.6 candidates,5 correct;4 persistent writes, only2 strict restorations of independently mastered original answers. Role-aware normalization removed one wrong candidate and recovered one correct candidate, but aggregate candidate correctness5/10 remains below generative6/10. No heldout newtest used. No G1 claim.

## Paired factorial preregistration
The first trial used sequential RNG states for adapter initialization. To isolate damage strength, next four cells all share the evaluator-private wrong targets and reset damage/writer seeds107. Compare association24/48 and uniform24/48. Child resource counters added as sidecars; metric semantics unchanged. Corpus mastery remains a failed prerequisite. This finite queue diagnoses damage sensitivity only and stops on any failed execution.

## paired-association24-01 — controlled P0 cell

Design: same32 development entities, same private wrong-target schedule, per-branch writer seed107, rank8 last4 layers,32repair steps,12retention prompts. Only damage mode/strength changes among these four cells. Canonical typed normalization fixed across cells. All available candidates receive counterfactual writes, without selecting by truth.

Results: {"independent_units": 10, "survey_units": 32, "clean_mastery_rate": 0.3125, "damage_eligibility_rate": 1.0, "oracle_success_rate": 0.7, "oracle_strict_restorations": 4, "retention_agreement": 1.0, "split_verified": true, "no_inverse_shortcut": true, "candidate_accuracy_structured_all_eligible": 0.7, "candidate_coverage_structured": 1.0, "candidate_accuracy_generative": 0.7, "resources": {"decoder_tokens_processed": 1382678, "model_forward_calls": 7880, "query_calls": 254, "tokens": 31128, "peak_vram_gb": 11.157513618469238}, "status": "P0 conditional factorial; G0 mastery failure not overridden", "assignment_source": "results/v2-new/association24-dev-01", "writer_seed_every_branch": 107}

Analysis: raw mastery remains below75%; this is conditional damage/writeability evidence only. G0FAIL, G1 remainsblocked. Fresh-process probabilities and strict original-heldout restoration are recorded per unit.

## paired-association48-01 and ontology-v1 audit
All32dev,10 eligible; fixed writer seed107 and same wrong targets as paired24. Oracle success5/10 (paired24:7/10), strict restorations2 vs4. Structured candidate correct5/10, generative5/10. This points to strength sensitivity, not method superiority. Before remaining uniform cells, scientific audit rejected unary action restrictions: airflow can excite strings (Aeolian harp); generic actions do not determine primary oscillator. Queue parent stopped after active48 cell completed; two uniform-v1 cells were never launched. Existing v1 outputs retained as rejected heuristic variant.

Primary counterexample sources verified2026-11-22: Detroit Historical Society Aeolian harp https://www.detroithistorical.org/learn/online-research/collection/object/harp-aeolian ; Met museum https://www.metmuseum.org/art/collection/search/503711 ; MIMO2011 https://mimo-international.com/documents/Hornbostel%20Sachs.pdf explicitly classifies air-driven reeds as aerophones. New RED/GREEN regression is frozen before rule correction. Prior CPU tests passed limited examples but encoded an overstrong generic-airflow assumption; they were insufficient physical validation.

## v02-physics-RED — rejected ontology-v1
Frozen independent counterexamples executed against original ontology; failure receipt retained. Generic airflow excited strings, plucked/bowed membranes, generic reed ambiguity and explicit conflict handling exposed invalid assumptions. These are model-independent physical validity failures, not heldout tuning. Corrected rules remove ALL unary action restrictions; only explicit primary oscillator taxonomy and explicitly direct air-driven reed conjunction constrain class. A direct relation is never inferred from generic airflow. Old source and RED tests retained in Git history; same regression rerun GREEN. Original CPU reed example is clarified to explicitly direct airflow, because its old generic assumption was invalid; independent regression source remainsunchanged.

## Corrected four-cell factorial and independent locality audit
Run fresh suffix02 cells consistently under corrected ontology/schema; preserve suffix01 failures. Wrong-target assignments and all neural settings unchanged. Also freeze62 non-target newdev heldout-template questions plus20 elementary general controls, none used in12-prompt KL training. Per-target82 controls exclude its own2questions. Thresholds token agreement>=0.9, neighbor/general accuracy drop<=0.05. Additional strict restoration combines original success, originally correct independent answers, and independent damage/repair locality. This can only narrow claims; original metrics remainimmutable. G0 stillfailed; cal/newtest untouched.

## Physics correction startup failures — preserved
The unchanged6counterexamples all failed against ontologyv1 (RED), as predicted. First corrected GREEN launch and CPU-audit03 failed before tests due to a literal backslash-n inserted while transferring the graph source. No model predictions or training occurred. Both failure receipts and sourceeba65e4 preserved. Fix only newline encoding, compile all v2 modules, then launch separately named retries; no metric/test changes.

## paired-association24-02 — controlled P0 cell

Design: same32 development entities, same private wrong-target schedule, per-branch writer seed107, rank8 last4 layers,32repair steps,12retention prompts. Only damage mode/strength changes among these four cells. Canonical typed normalization fixed across cells. All available candidates receive counterfactual writes, without selecting by truth.

Results: {"independent_units": 10, "survey_units": 32, "clean_mastery_rate": 0.3125, "damage_eligibility_rate": 1.0, "oracle_success_rate": 0.7, "oracle_strict_restorations": 4, "retention_agreement": 1.0, "split_verified": true, "no_inverse_shortcut": true, "candidate_accuracy_structured_all_eligible": 0.5, "candidate_coverage_structured": 0.7, "candidate_accuracy_generative": 0.7, "resources": {"decoder_tokens_processed": 1275750, "model_forward_calls": 8042, "query_calls": 248, "tokens": 31513, "peak_vram_gb": 11.157513618469238}, "status": "P0 conditional factorial; G0 mastery failure not overridden", "assignment_source": "results/v2-new/association24-dev-01", "writer_seed_every_branch": 107}

Analysis: raw mastery remains below75%; this is conditional damage/writeability evidence only. G0FAIL, G1 remainsblocked. Fresh-process probabilities and strict original-heldout restoration are recorded per unit.

## paired-association48-02 — controlled P0 cell

Design: same32 development entities, same private wrong-target schedule, per-branch writer seed107, rank8 last4 layers,32repair steps,12retention prompts. Only damage mode/strength changes among these four cells. Canonical typed normalization fixed across cells. All available candidates receive counterfactual writes, without selecting by truth.

Results: {"independent_units": 10, "survey_units": 32, "clean_mastery_rate": 0.3125, "damage_eligibility_rate": 1.0, "oracle_success_rate": 0.5, "oracle_strict_restorations": 2, "retention_agreement": 1.0, "split_verified": true, "no_inverse_shortcut": true, "candidate_accuracy_structured_all_eligible": 0.4, "candidate_coverage_structured": 0.5, "candidate_accuracy_generative": 0.5, "resources": {"decoder_tokens_processed": 1466382, "model_forward_calls": 8398, "query_calls": 244, "tokens": 31191, "peak_vram_gb": 11.157513618469238}, "status": "P0 conditional factorial; G0 mastery failure not overridden", "assignment_source": "results/v2-new/association24-dev-01", "writer_seed_every_branch": 107}

Analysis: raw mastery remains below75%; this is conditional damage/writeability evidence only. G0FAIL, G1 remainsblocked. Fresh-process probabilities and strict original-heldout restoration are recorded per unit.

## paired-uniform24-02 — controlled P0 cell

Design: same32 development entities, same private wrong-target schedule, per-branch writer seed107, rank8 last4 layers,32repair steps,12retention prompts. Only damage mode/strength changes among these four cells. Canonical typed normalization fixed across cells. All available candidates receive counterfactual writes, without selecting by truth.

Results: {"independent_units": 5, "survey_units": 32, "clean_mastery_rate": 0.3125, "damage_eligibility_rate": 0.5, "oracle_success_rate": 1.0, "oracle_strict_restorations": 4, "retention_agreement": 1.0, "split_verified": true, "no_inverse_shortcut": true, "candidate_accuracy_structured_all_eligible": 0.6, "candidate_coverage_structured": 0.6, "candidate_accuracy_generative": 1.0, "resources": {"decoder_tokens_processed": 763640, "model_forward_calls": 4468, "query_calls": 175, "tokens": 20225, "peak_vram_gb": 11.157513618469238}, "status": "P0 conditional factorial; G0 mastery failure not overridden", "assignment_source": "results/v2-new/association24-dev-01", "writer_seed_every_branch": 107}

Analysis: raw mastery remains below75%; this is conditional damage/writeability evidence only. G0FAIL, G1 remainsblocked. Fresh-process probabilities and strict original-heldout restoration are recorded per unit.

## Realized wrong-target balance audit and prospective control
Original schedule balanced8each across planned32, but only10 qualified: realizedA2/B1/C2/D5, optimistically fitted inverse70%, cyclic inverse20%. This is not the old deterministic cyclic algorithm, but planned-pool balance does not survive mastery filtering and cannot justify a shortcut-free claim for the realized cohort. New preregistered control randomizes AFTER clean screen qualification; accept only class count spread<=1 and >=2 possible truth classes per wrong target. It never reads candidate/repair outcomes or independent-clean answers to choose assignments. One new association24 trial; do not resample on its success. All four original physical cells remaindescriptive sensitivity evidence, not confirmatory.

## paired-uniform48-02 — controlled P0 cell

Design: same32 development entities, same private wrong-target schedule, per-branch writer seed107, rank8 last4 layers,32repair steps,12retention prompts. Only damage mode/strength changes among these four cells. Canonical typed normalization fixed across cells. All available candidates receive counterfactual writes, without selecting by truth.

Results: {"independent_units": 6, "survey_units": 32, "clean_mastery_rate": 0.3125, "damage_eligibility_rate": 0.6, "oracle_success_rate": 1.0, "oracle_strict_restorations": 3, "retention_agreement": 1.0, "split_verified": true, "no_inverse_shortcut": true, "candidate_accuracy_structured_all_eligible": 0.3333333333333333, "candidate_coverage_structured": 0.6666666666666666, "candidate_accuracy_generative": 0.6666666666666666, "resources": {"decoder_tokens_processed": 1134957, "model_forward_calls": 5621, "query_calls": 190, "tokens": 22393, "peak_vram_gb": 11.157513618469238}, "status": "P0 conditional factorial; G0 mastery failure not overridden", "assignment_source": "results/v2-new/association24-dev-01", "writer_seed_every_branch": 107}

Analysis: raw mastery remains below75%; this is conditional damage/writeability evidence only. G0FAIL, G1 remainsblocked. Fresh-process probabilities and strict original-heldout restoration are recorded per unit.

## balanced-association24-01 — conditional randomization control
Qualification uses only clean screen0/1. Rejection sampler accepted draw15; realizedA2/B3/C3/D2, >=2truth classes per wrongtarget, optimistically fitted inverse60%. All10qualified became effective/local under original12 retention prompts; no attrition, so target balance survived damage eligibility. Structured correct4/10(5candidates), generative6/10, oracle operational success8/10, originally independently mastered restorations5/10 before new locality audit. No affirmative G1 evidence.

## Final audit plan
Reproduce compact damage LoRA for five corrected cells, compare exact merged tensor SHA against original saved damage, and reload/merge saved adapter again. Never call an approximate reconstruction exact. Then audit all legacy/corrected branches using82per-target controls (62neighbor +20toy general) held out from KL. Freeze A: candidate_change selector at50% for both generation methods, identical seeded writer; B: same structured candidates/writer comparing candidate_change vs ERE/full LOOCV. All analysis remainsP0 exploratory because G0failed; no calibration/test queried and no formal calibration artifact produced.

## v02-compact-damage-01 — exact portable damage reconstruction
Five corrected cells contain41 eligible unit-config cases (at most10 independent entities). Recreated every damage-only LoRA with original wrong target, mode/strength,seed107 and12clean retention prompts. Merged tensor SHA matched all41 original saved damaged states; serialized adapters were unloaded, reloaded and merged again, and all41 still matched. This proves portable reconstruction on the same environment, not cross-model transfer or G3.

## v02-independent-locality-01 — scientific G0 FAIL
Audited81 unit-config cases across9 historical/corrected cells, at most10 independent new entities. Each target excluded from its own controls:62 neighbor questions +20 elementary general controls, all held out from KL training. Corrected cells damage-locality pass:association24 0/10,association48 0/10,uniform24 0/5,uniform48 1/6,balancedassociation24 0/10. Uniform repaired variants all failed independent locality. Original12 training-retention agreement was often1.0, but does not establish independent locality. Strict restorations require original heldout mastery, operational success, independent damage locality and independent repair locality; none can be counted across41 corrected unit-config cases. This invalidates repair claims under this benchmark, not proof that knowledge is permanently irreparable. No G1/cal/test or real-media stages launched.

## Analysis parser failure — preserved
First final analysis failed because controls.json is a list while per-unit audit records are objects. No model execution occurred. Add object-type guard when selecting audit records, preserve failure receipt and source5886a23, rerun uniquely named output directories. Raw predictions/metrics unchanged.


<!-- KRMM_V02_FINAL_APPENDIX -->

# v0.2 最终逐次启动索引

所有失败均保留；详细设计在前述协议/实验段落，原始stdout/stderr见docs/v2/run-logs.tar.gz。

| 启动标签 | 假设/设计 | 结果 | 秒 | 源码SHA |
|---|---|---|---:|---|
|v02-cpu-audit-01|H0-typed-evidence-contracts|SUCCEEDED|6.002|1aacf2d897a7|
|v02-survey-dev-01|H0-new-entity-mastery|FAILED|11.002|9a7300228474|
|v02-survey-dev-02|H0-new-entity-mastery|SUCCEEDED|14.003|386946669867|
|v02-competence-01|H0-label-wording-vs-entity-knowledge|SUCCEEDED|34.007|6c1d91e46d51|
|v02-association24-dev-01|H0-random-damage-conditional-writeability|SUCCEEDED|365.09|cd1624b459d5|
|v02-canonical-replay-01|H0-typed-normalization-same-damage|SUCCEEDED|234.069|52fef43550fb|
|v02-cpu-audit-02|H0-public-service-boundary|SUCCEEDED|6.002|359eedf26597|
|v02-analysis-01|H0-dev-descriptive-reliability|SUCCEEDED|2.001|359eedf26597|
|paired-association24-01|H0-paired-damage-factorial|SUCCEEDED|371.081|359eedf26597|
|paired-association48-01|H0-paired-damage-factorial|SUCCEEDED|384.056|359eedf26597|
|v02-physics-RED|H0-physical-constraint-validity|FAILED|1.001|fde3a82f8d72|
|v02-physics-GREEN|H0-physical-constraint-validity|FAILED|1.001|eba65e497417|
|v02-cpu-audit-03|H0-public-service-boundary|FAILED|1.001|eba65e497417|
|v02-physics-GREEN-02|H0-physical-constraint-validity|SUCCEEDED|1.001|fbdd5091e270|
|v02-cpu-audit-04|H0-public-service-boundary|SUCCEEDED|6.002|fbdd5091e270|
|paired-association24-02|H0-physical-paired-damage-factorial|SUCCEEDED|369.05|fbdd5091e270|
|paired-association48-02|H0-physical-paired-damage-factorial|SUCCEEDED|382.052|fbdd5091e270|
|paired-uniform24-02|H0-physical-paired-damage-factorial|SUCCEEDED|207.029|fbdd5091e270|
|paired-uniform48-02|H0-physical-paired-damage-factorial|SUCCEEDED|260.043|fbdd5091e270|
|balanced-association24-01|H0-conditional-balanced-randomization|SUCCEEDED|355.06|ed5ea6fa633d|
|v02-compact-damage-01|H0-exact-portable-damage-reconstruction|SUCCEEDED|223.036|5886a23610f8|
|v02-independent-locality-01|H0-independent-nontarget-locality|SUCCEEDED|208.036|90fcb775dec4|
|analysis-paired-association24-02|H0-descriptive-generation-selection-isolation|FAILED|1.001|90fcb775dec4|
|analysis-final-paired-association24-02|H0-descriptive-generation-selection-isolation|SUCCEEDED|2.001|31f9d17d686f|
|analysis-final-paired-association48-02|H0-descriptive-generation-selection-isolation|SUCCEEDED|2.002|31f9d17d686f|
|analysis-final-paired-uniform24-02|H0-descriptive-generation-selection-isolation|SUCCEEDED|2.001|31f9d17d686f|
|analysis-final-paired-uniform48-02|H0-descriptive-generation-selection-isolation|SUCCEEDED|3.001|31f9d17d686f|
|analysis-final-balanced-association24-01|H0-descriptive-generation-selection-isolation|SUCCEEDED|2.001|31f9d17d686f|

# 修订版逐单元设计、结果与分析

共同设计：新开发实体；独立错误目标；rank8最后4层；writer seed107、32步、12条KL保留。原操作性结果与独立审计分开，未根据独立审计重新选择训练超参。

## paired-association24-02

| 实体/记录 | 筛查掌握 | 原始独立正确率 | 原操作性资格 | 结构化/生成候选 | Oracle操作成功 | 分析 |
|---|---:|---:|---|---|---|---|
|[nkangala](results/v2-new/paired-association24-02/01115bbe019c197a.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[chipendani](results/v2-new/paired-association24-02/039b3187d454e68f.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[inanga](results/v2-new/paired-association24-02/0a017e2878a3307f.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[atwreshie](results/v2-new/paired-association24-02/0a712ba814e85a17.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[jingle ring](results/v2-new/paired-association24-02/0b7ed182007d45ef.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[gandira](results/v2-new/paired-association24-02/0dbead561e097cb4.json)|1.0|0.0|True|None/C|True|损伤独立局部性未过|
|[chiboni](results/v2-new/paired-association24-02/0dcca257dc08b6f5.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[valiha](results/v2-new/paired-association24-02/1266a78769752bf1.json)|1.0|1.0|True|A/A|True|损伤独立局部性未过|
|[hurdy-gurdy](results/v2-new/paired-association24-02/1386178acfc8171c.json)|1.0|1.0|True|A/A|True|损伤独立局部性未过|
|[gusle](results/v2-new/paired-association24-02/1605d3da214891eb.json)|1.0|1.0|True|A/A|True|损伤独立局部性未过|
|[svirala](results/v2-new/paired-association24-02/16077b130ac30acc.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[dobachi](results/v2-new/paired-association24-02/160a362326b9cca7.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[Lobi whistle](results/v2-new/paired-association24-02/16a5abe3c333c14b.json)|1.0|1.0|True|None/D|False|损伤独立局部性未过|
|[mbiu](results/v2-new/paired-association24-02/182a086a1e48803e.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[akukuadwo](results/v2-new/paired-association24-02/198481c123964c13.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[Anglo concertina](results/v2-new/paired-association24-02/1bb8d48beb7f2eee.json)|1.0|1.0|True|None/B|True|损伤独立局部性未过|
|[shudraga](results/v2-new/paired-association24-02/1cf12949776ae646.json)|1.0|0.5|True|A/A|True|损伤独立局部性未过|
|[tsenatsil](results/v2-new/paired-association24-02/1d4740684ee819e1.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[yueqin](results/v2-new/paired-association24-02/1d688445d1a4ad35.json)|1.0|1.0|True|A/A|False|损伤独立局部性未过|
|[harmonium](results/v2-new/paired-association24-02/203c6e605e81393e.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[turu](results/v2-new/paired-association24-02/2184165bbaa70526.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[kā’eke’eke](results/v2-new/paired-association24-02/224594a5bdae4542.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[nagara](results/v2-new/paired-association24-02/228430a2d2e3cd1c.json)|1.0|0.0|True|A/D|False|损伤独立局部性未过|
|[mukupela](results/v2-new/paired-association24-02/22d377519ec18a1d.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[tangi](results/v2-new/paired-association24-02/24a6e418b5c840d4.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[firikyiwa](results/v2-new/paired-association24-02/258b8707ed731e05.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[hwamanda](results/v2-new/paired-association24-02/2714bfcc0e2fc9c8.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[cajón](results/v2-new/paired-association24-02/2779987edac99a31.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[cuica](results/v2-new/paired-association24-02/2974ce8d84473269.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[ngoma](results/v2-new/paired-association24-02/297572f964e164ca.json)|1.0|0.0|True|A/A|True|损伤独立局部性未过|
|[‘ili’ili](results/v2-new/paired-association24-02/2e4033d870a8c85f.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[agogô](results/v2-new/paired-association24-02/3800bbd8f0956b62.json)|0.0|—|False|—/—|—|not_stably_mastered|

## paired-association48-02

| 实体/记录 | 筛查掌握 | 原始独立正确率 | 原操作性资格 | 结构化/生成候选 | Oracle操作成功 | 分析 |
|---|---:|---:|---|---|---|---|
|[nkangala](results/v2-new/paired-association48-02/01115bbe019c197a.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[chipendani](results/v2-new/paired-association48-02/039b3187d454e68f.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[inanga](results/v2-new/paired-association48-02/0a017e2878a3307f.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[atwreshie](results/v2-new/paired-association48-02/0a712ba814e85a17.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[jingle ring](results/v2-new/paired-association48-02/0b7ed182007d45ef.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[gandira](results/v2-new/paired-association48-02/0dbead561e097cb4.json)|1.0|0.0|True|None/A|True|损伤独立局部性未过|
|[chiboni](results/v2-new/paired-association48-02/0dcca257dc08b6f5.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[valiha](results/v2-new/paired-association48-02/1266a78769752bf1.json)|1.0|1.0|True|None/A|False|损伤独立局部性未过|
|[hurdy-gurdy](results/v2-new/paired-association48-02/1386178acfc8171c.json)|1.0|1.0|True|A/C|False|损伤独立局部性未过|
|[gusle](results/v2-new/paired-association48-02/1605d3da214891eb.json)|1.0|1.0|True|A/A|False|损伤独立局部性未过|
|[svirala](results/v2-new/paired-association48-02/16077b130ac30acc.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[dobachi](results/v2-new/paired-association48-02/160a362326b9cca7.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[Lobi whistle](results/v2-new/paired-association48-02/16a5abe3c333c14b.json)|1.0|1.0|True|None/D|False|损伤独立局部性未过|
|[mbiu](results/v2-new/paired-association48-02/182a086a1e48803e.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[akukuadwo](results/v2-new/paired-association48-02/198481c123964c13.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[Anglo concertina](results/v2-new/paired-association48-02/1bb8d48beb7f2eee.json)|1.0|1.0|True|None/B|True|损伤独立局部性未过|
|[shudraga](results/v2-new/paired-association48-02/1cf12949776ae646.json)|1.0|0.5|True|A/A|True|损伤独立局部性未过|
|[tsenatsil](results/v2-new/paired-association48-02/1d4740684ee819e1.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[yueqin](results/v2-new/paired-association48-02/1d688445d1a4ad35.json)|1.0|1.0|True|A/A|True|损伤独立局部性未过|
|[harmonium](results/v2-new/paired-association48-02/203c6e605e81393e.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[turu](results/v2-new/paired-association48-02/2184165bbaa70526.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[kā’eke’eke](results/v2-new/paired-association48-02/224594a5bdae4542.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[nagara](results/v2-new/paired-association48-02/228430a2d2e3cd1c.json)|1.0|0.0|True|A/D|False|损伤独立局部性未过|
|[mukupela](results/v2-new/paired-association48-02/22d377519ec18a1d.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[tangi](results/v2-new/paired-association48-02/24a6e418b5c840d4.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[firikyiwa](results/v2-new/paired-association48-02/258b8707ed731e05.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[hwamanda](results/v2-new/paired-association48-02/2714bfcc0e2fc9c8.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[cajón](results/v2-new/paired-association48-02/2779987edac99a31.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[cuica](results/v2-new/paired-association48-02/2974ce8d84473269.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[ngoma](results/v2-new/paired-association48-02/297572f964e164ca.json)|1.0|0.0|True|None/D|True|损伤独立局部性未过|
|[‘ili’ili](results/v2-new/paired-association48-02/2e4033d870a8c85f.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[agogô](results/v2-new/paired-association48-02/3800bbd8f0956b62.json)|0.0|—|False|—/—|—|not_stably_mastered|

## paired-uniform24-02

| 实体/记录 | 筛查掌握 | 原始独立正确率 | 原操作性资格 | 结构化/生成候选 | Oracle操作成功 | 分析 |
|---|---:|---:|---|---|---|---|
|[nkangala](results/v2-new/paired-uniform24-02/01115bbe019c197a.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[chipendani](results/v2-new/paired-uniform24-02/039b3187d454e68f.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[inanga](results/v2-new/paired-uniform24-02/0a017e2878a3307f.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[atwreshie](results/v2-new/paired-uniform24-02/0a712ba814e85a17.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[jingle ring](results/v2-new/paired-uniform24-02/0b7ed182007d45ef.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[gandira](results/v2-new/paired-uniform24-02/0dbead561e097cb4.json)|1.0|0.0|False|—/—|—|damage_not_local_or_effective|
|[chiboni](results/v2-new/paired-uniform24-02/0dcca257dc08b6f5.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[valiha](results/v2-new/paired-uniform24-02/1266a78769752bf1.json)|1.0|1.0|True|None/A|True|损伤独立局部性未过|
|[hurdy-gurdy](results/v2-new/paired-uniform24-02/1386178acfc8171c.json)|1.0|1.0|True|A/A|True|损伤独立局部性未过|
|[gusle](results/v2-new/paired-uniform24-02/1605d3da214891eb.json)|1.0|1.0|False|—/—|—|damage_not_local_or_effective|
|[svirala](results/v2-new/paired-uniform24-02/16077b130ac30acc.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[dobachi](results/v2-new/paired-uniform24-02/160a362326b9cca7.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[Lobi whistle](results/v2-new/paired-uniform24-02/16a5abe3c333c14b.json)|1.0|1.0|False|—/—|—|damage_not_local_or_effective|
|[mbiu](results/v2-new/paired-uniform24-02/182a086a1e48803e.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[akukuadwo](results/v2-new/paired-uniform24-02/198481c123964c13.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[Anglo concertina](results/v2-new/paired-uniform24-02/1bb8d48beb7f2eee.json)|1.0|1.0|True|None/B|True|损伤独立局部性未过|
|[shudraga](results/v2-new/paired-uniform24-02/1cf12949776ae646.json)|1.0|0.5|True|A/A|True|损伤独立局部性未过|
|[tsenatsil](results/v2-new/paired-uniform24-02/1d4740684ee819e1.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[yueqin](results/v2-new/paired-uniform24-02/1d688445d1a4ad35.json)|1.0|1.0|True|A/A|True|损伤独立局部性未过|
|[harmonium](results/v2-new/paired-uniform24-02/203c6e605e81393e.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[turu](results/v2-new/paired-uniform24-02/2184165bbaa70526.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[kā’eke’eke](results/v2-new/paired-uniform24-02/224594a5bdae4542.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[nagara](results/v2-new/paired-uniform24-02/228430a2d2e3cd1c.json)|1.0|0.0|False|—/—|—|damage_not_local_or_effective|
|[mukupela](results/v2-new/paired-uniform24-02/22d377519ec18a1d.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[tangi](results/v2-new/paired-uniform24-02/24a6e418b5c840d4.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[firikyiwa](results/v2-new/paired-uniform24-02/258b8707ed731e05.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[hwamanda](results/v2-new/paired-uniform24-02/2714bfcc0e2fc9c8.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[cajón](results/v2-new/paired-uniform24-02/2779987edac99a31.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[cuica](results/v2-new/paired-uniform24-02/2974ce8d84473269.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[ngoma](results/v2-new/paired-uniform24-02/297572f964e164ca.json)|1.0|0.0|False|—/—|—|damage_not_local_or_effective|
|[‘ili’ili](results/v2-new/paired-uniform24-02/2e4033d870a8c85f.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[agogô](results/v2-new/paired-uniform24-02/3800bbd8f0956b62.json)|0.0|—|False|—/—|—|not_stably_mastered|

## paired-uniform48-02

| 实体/记录 | 筛查掌握 | 原始独立正确率 | 原操作性资格 | 结构化/生成候选 | Oracle操作成功 | 分析 |
|---|---:|---:|---|---|---|---|
|[nkangala](results/v2-new/paired-uniform48-02/01115bbe019c197a.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[chipendani](results/v2-new/paired-uniform48-02/039b3187d454e68f.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[inanga](results/v2-new/paired-uniform48-02/0a017e2878a3307f.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[atwreshie](results/v2-new/paired-uniform48-02/0a712ba814e85a17.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[jingle ring](results/v2-new/paired-uniform48-02/0b7ed182007d45ef.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[gandira](results/v2-new/paired-uniform48-02/0dbead561e097cb4.json)|1.0|0.0|True|None/C|True|损伤独立局部性未过|
|[chiboni](results/v2-new/paired-uniform48-02/0dcca257dc08b6f5.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[valiha](results/v2-new/paired-uniform48-02/1266a78769752bf1.json)|1.0|1.0|True|None/A|True|损伤独立局部性未过|
|[hurdy-gurdy](results/v2-new/paired-uniform48-02/1386178acfc8171c.json)|1.0|1.0|False|—/—|—|damage_not_local_or_effective|
|[gusle](results/v2-new/paired-uniform48-02/1605d3da214891eb.json)|1.0|1.0|False|—/—|—|damage_not_local_or_effective|
|[svirala](results/v2-new/paired-uniform48-02/16077b130ac30acc.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[dobachi](results/v2-new/paired-uniform48-02/160a362326b9cca7.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[Lobi whistle](results/v2-new/paired-uniform48-02/16a5abe3c333c14b.json)|1.0|1.0|True|B/B|True|损伤独立局部性未过|
|[mbiu](results/v2-new/paired-uniform48-02/182a086a1e48803e.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[akukuadwo](results/v2-new/paired-uniform48-02/198481c123964c13.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[Anglo concertina](results/v2-new/paired-uniform48-02/1bb8d48beb7f2eee.json)|1.0|1.0|False|—/—|—|damage_not_local_or_effective|
|[shudraga](results/v2-new/paired-uniform48-02/1cf12949776ae646.json)|1.0|0.5|False|—/—|—|damage_not_local_or_effective|
|[tsenatsil](results/v2-new/paired-uniform48-02/1d4740684ee819e1.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[yueqin](results/v2-new/paired-uniform48-02/1d688445d1a4ad35.json)|1.0|1.0|True|A/A|True|损伤独立局部性未过|
|[harmonium](results/v2-new/paired-uniform48-02/203c6e605e81393e.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[turu](results/v2-new/paired-uniform48-02/2184165bbaa70526.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[kā’eke’eke](results/v2-new/paired-uniform48-02/224594a5bdae4542.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[nagara](results/v2-new/paired-uniform48-02/228430a2d2e3cd1c.json)|1.0|0.0|True|A/A|True|损伤独立局部性未过|
|[mukupela](results/v2-new/paired-uniform48-02/22d377519ec18a1d.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[tangi](results/v2-new/paired-uniform48-02/24a6e418b5c840d4.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[firikyiwa](results/v2-new/paired-uniform48-02/258b8707ed731e05.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[hwamanda](results/v2-new/paired-uniform48-02/2714bfcc0e2fc9c8.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[cajón](results/v2-new/paired-uniform48-02/2779987edac99a31.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[cuica](results/v2-new/paired-uniform48-02/2974ce8d84473269.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[ngoma](results/v2-new/paired-uniform48-02/297572f964e164ca.json)|1.0|0.0|True|A/D|True|损伤局部性通过；写回独立保留未过|
|[‘ili’ili](results/v2-new/paired-uniform48-02/2e4033d870a8c85f.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[agogô](results/v2-new/paired-uniform48-02/3800bbd8f0956b62.json)|0.0|—|False|—/—|—|not_stably_mastered|

## balanced-association24-01

| 实体/记录 | 筛查掌握 | 原始独立正确率 | 原操作性资格 | 结构化/生成候选 | Oracle操作成功 | 分析 |
|---|---:|---:|---|---|---|---|
|[nkangala](results/v2-new/balanced-association24-01/01115bbe019c197a.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[chipendani](results/v2-new/balanced-association24-01/039b3187d454e68f.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[inanga](results/v2-new/balanced-association24-01/0a017e2878a3307f.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[atwreshie](results/v2-new/balanced-association24-01/0a712ba814e85a17.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[jingle ring](results/v2-new/balanced-association24-01/0b7ed182007d45ef.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[gandira](results/v2-new/balanced-association24-01/0dbead561e097cb4.json)|1.0|0.0|True|None/C|True|损伤独立局部性未过|
|[chiboni](results/v2-new/balanced-association24-01/0dcca257dc08b6f5.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[valiha](results/v2-new/balanced-association24-01/1266a78769752bf1.json)|1.0|1.0|True|None/A|True|损伤独立局部性未过|
|[hurdy-gurdy](results/v2-new/balanced-association24-01/1386178acfc8171c.json)|1.0|1.0|True|A/A|True|损伤独立局部性未过|
|[gusle](results/v2-new/balanced-association24-01/1605d3da214891eb.json)|1.0|1.0|True|A/A|True|损伤独立局部性未过|
|[svirala](results/v2-new/balanced-association24-01/16077b130ac30acc.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[dobachi](results/v2-new/balanced-association24-01/160a362326b9cca7.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[Lobi whistle](results/v2-new/balanced-association24-01/16a5abe3c333c14b.json)|1.0|1.0|True|None/A|False|损伤独立局部性未过|
|[mbiu](results/v2-new/balanced-association24-01/182a086a1e48803e.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[akukuadwo](results/v2-new/balanced-association24-01/198481c123964c13.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[Anglo concertina](results/v2-new/balanced-association24-01/1bb8d48beb7f2eee.json)|1.0|1.0|True|None/C|True|损伤独立局部性未过|
|[shudraga](results/v2-new/balanced-association24-01/1cf12949776ae646.json)|1.0|0.5|True|A/A|True|损伤独立局部性未过|
|[tsenatsil](results/v2-new/balanced-association24-01/1d4740684ee819e1.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[yueqin](results/v2-new/balanced-association24-01/1d688445d1a4ad35.json)|1.0|1.0|True|A/A|True|损伤独立局部性未过|
|[harmonium](results/v2-new/balanced-association24-01/203c6e605e81393e.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[turu](results/v2-new/balanced-association24-01/2184165bbaa70526.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[kā’eke’eke](results/v2-new/balanced-association24-01/224594a5bdae4542.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[nagara](results/v2-new/balanced-association24-01/228430a2d2e3cd1c.json)|1.0|0.0|True|A/A|False|损伤独立局部性未过|
|[mukupela](results/v2-new/balanced-association24-01/22d377519ec18a1d.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[tangi](results/v2-new/balanced-association24-01/24a6e418b5c840d4.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[firikyiwa](results/v2-new/balanced-association24-01/258b8707ed731e05.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[hwamanda](results/v2-new/balanced-association24-01/2714bfcc0e2fc9c8.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[cajón](results/v2-new/balanced-association24-01/2779987edac99a31.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[cuica](results/v2-new/balanced-association24-01/2974ce8d84473269.json)|0.5|—|False|—/—|—|not_stably_mastered|
|[ngoma](results/v2-new/balanced-association24-01/297572f964e164ca.json)|1.0|0.0|True|None/D|True|损伤独立局部性未过|
|[‘ili’ili](results/v2-new/balanced-association24-01/2e4033d870a8c85f.json)|0.0|—|False|—/—|—|not_stably_mastered|
|[agogô](results/v2-new/balanced-association24-01/3800bbd8f0956b62.json)|0.0|—|False|—/—|—|not_stably_mastered|

## 最终决策

G0 FAIL / PIVOT。新开发掌握率10/32，独立局部性与严格恢复要求未通过。28/30次启动、41例核心损伤精确重建、81例历史/修订保留审计均记录。正式ERE校准、新测试、真实视听与第二模型未启动。详细统计见docs/v2/REPORT.md。完成同步后执行root shutdown；实际观察单列docs/v2/shutdown-observation.json。
