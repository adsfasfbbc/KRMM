"""Finite P0 queue: four predeclared cells, stop on any execution failure."""
import json,subprocess,sys,time
from pathlib import Path
root=Path('/root/rivermind-data/KRMM')
python='/root/rivermind-data/krmm-env/bin/python'
cells=[('association',24),('association',48),('uniform',24),('uniform',48)]
for mode,steps in cells:
 label=f"paired-{mode}{steps}-02";out=f"results/v2-new/{label}"
 before=set((root/'.autoresearch-v02/runs').glob('*/run.json'))
 cmd=[python,'scripts/researchctl.py','run','--workspace','.autoresearch-v02','--hypothesis','H0-physical-paired-damage-factorial','--gate','G0','--label',label,'--repair-reason','physical-ontology-v2-controls-with-failed-G0','--approved','--',python,'-m','krmm.v2.sweep','--config',f'configs/v2/{mode}-{steps}.json','--out',out,'--assignment-source','results/v2-new/association24-dev-01']
 print('LAUNCH',label,flush=True)
 p=subprocess.run(cmd,cwd=root)
 records=set((root/'.autoresearch-v02/runs').glob('*/run.json'))-before
 if p.returncode or len(records)!=1:raise SystemExit('QUEUE_STOP_EXECUTION_OR_RECEIPT_ERROR')
 receipt=json.loads(next(iter(records)).read_text())
 if receipt['status']!='SUCCEEDED':raise SystemExit('QUEUE_STOP_FAILED_CELL')
 metrics=json.loads((root/out/'metrics.json').read_text())
 with (root/'EXPERIMENTS.md').open('a') as journal:
  journal.write('\n## '+label+' — controlled P0 cell\n\nDesign: same32 development entities, same private wrong-target schedule, per-branch writer seed107, rank8 last4 layers,32repair steps,12retention prompts. Only damage mode/strength changes among these four cells. Canonical typed normalization fixed across cells. All available candidates receive counterfactual writes, without selecting by truth.\n\nResults: '+json.dumps(metrics,ensure_ascii=False)+'\n\nAnalysis: raw mastery remains below75%; this is conditional damage/writeability evidence only. G0FAIL, G1 remainsblocked. Fresh-process probabilities and strict original-heldout restoration are recorded per unit.\n')
 print('COMPLETED',label,json.dumps(metrics),flush=True)
print('FINITE_P0_QUEUE_COMPLETE',flush=True)
