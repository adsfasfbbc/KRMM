# KRMM experiment journal

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
