# KRMM

Unit-level repairability diagnosis and selective persistent repair for damaged multimodal models.

**Research status:** exploratory implementation. Controlled neural-world experiments completed; real Qwen2.5-Omni G0 in progress. No novel-method, real multimodal-recovery or paper-ready claim.

## What is implemented

- Evaluator-only ground truth with a separate public task schema; 48 musical-instrument knowledge units split by entity (12 development / 12 calibration / 24 test).
- Fixed-budget targeted wrong-association editing merged into the damaged model; repair receives no clean checkpoint, original answers or damage differences.
- Internal-prior candidate reaccess from the damaged model, key/sham evidence probes, pre-update gradient conflict and regularized logistic diagnosis.
- Rank-8 updates to the last four language layers, damaged-teacher retention KL and update penalty. Unconstrained pseudo-label and truth-supervised upper-bound controls use the same candidate/input/step budgets where appropriate.
- Removal of auxiliary evidence during writeback; serialized adapter unload/reload before scoring the two unseen paraphrases. Oracle truth is evaluator-only.
- Per-unit raw predictions, eligibility/exclusions, actual query/forward counts, peak VRAM, source SHA and environment provenance.
- Generated image + nonverbal audio controlled worlds with redundancy / complementarity / ambiguity / random mappings, measured media interventions and explicit claim limitations.
- Bounded experiment runner (20 launches, 8h approved batch, 80GB storage cap) and one unified [experiment journal](EXPERIMENTS.md).

## Cloud-only reproduction

All commands below execute on the authorized Linux GPU server; no local checkout or environment is needed.

```bash
cd /root/rivermind-data/KRMM
PY=/root/rivermind-data/krmm-env/bin/python
$PY -m unittest discover -s tests
python scripts/researchctl.py validate --workspace .autoresearch
python scripts/researchctl.py run --workspace .autoresearch --hypothesis REAL1 --gate G0 --label pilot --approved -- "$PY" -m krmm.experiment --split dev --limit 4 --out results/real-g0-v1
$PY -m krmm.analysis results/real-dev results/real-calibration results/real-test --out results/diagnosis.json
```

Model release: `Qwen/Qwen2.5-Omni-3B@f75b40e3da2003cdd6e1829b1f420ca70797c34e`. `scripts/download_model_ranges.py` resumes official LFS downloads and verifies full SHA256 before readiness. Runtime uses server-only venv, torch 2.12.1+cu130, transformers 4.57.6. Exact environment: [docs/environment.freeze.txt](docs/environment.freeze.txt).

## Interpretation boundaries

The real pilot uses closed-choice likelihoods conditional on A/B/C/D, not open-ended response accuracy. First two prompts train; final two prompts evaluate. Atomic question reaccess is **direct residual access**, not proven compositional reconstruction. Media encoders are frozen; image/audio conditioning is supported, but internal-prior and auxiliary-evidence results must be reported separately.

Synthetic diagnostic scores use known constraint structure, a privileged structural baseline. The strongest single support feature currently matches the learned selector, so the experiments do not establish an incremental learned diagnostic contribution. Synthetic worlds and three world seeds do not substitute for pretrained real-world multimodal knowledge. The current twin-identity check is a limited carrier-erasure sanity check, not the full two-original-model construction.

Real-media provenance remains on the server; binaries are not redistributed. Missing downloads, failures and negative results remain visible. A failed G0 blocks promotion. Holdout evaluation is frozen and exploratory pilot sample counts cannot support publication claims.

## Artifacts

- [Technical requirements](docs/REQUIREMENTS.md)
- [Protocol](.autoresearch/protocol.json)
- [Experiment journal](EXPERIMENTS.md)
- [Novelty ledger](NOVELTY_LEDGER.md)
- `results/`: immutable per-trial predictions, metrics and controls
- `.autoresearch/runs/`: command, source SHA, timing and terminal logs

No credentials, SSH password, proxy subscription, model weights or private runtime files belong in this repository.
