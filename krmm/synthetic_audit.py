import json
from pathlib import Path
import numpy as np
root=Path('results/synthetic-v1');rows=[]
for p in sorted(root.glob('seed-*.json')):rows+=json.loads(p.read_text())['rows']
report={'status':'G0_SANITY_ONLY','statistical_unit':'world/seed, not entity for across-world uncertainty','n_worlds':3,'settings':{}}
for setting in ['internal','auxiliary']:
 rr=[r for r in rows if r['setting']==setting and r['split']=='test']
 y=np.array([r['success'] for r in rr]);support=np.array([r['features'][2] for r in rr]);confidence=np.array([r['features'][0] for r in rr]);k=len(rr)//2
 ids=np.argsort(-support,kind='stable')[:k]
 report['settings'][setting]={'single_support_baseline_at_half_coverage':float(y[ids].sum()/len(rr)),
 'confidence_unique_values':len(set(confidence.tolist())),
 'oracle_success_rate':float(np.mean([r['oracle_success'] for r in rr])),
 'interpretation':'Constraint-support baseline matches learned selector. Confidence is constant by construction. No evidence for incremental learned-diagnosis value or writeability discrimination.'}
report['missing_controls']=['Real intervention on key media (current key-deletion support is an analytic structural annotation)','Learned rejection without known type/information regime','Opposite mappings learned in two independent original networks','Second damage mechanism and model','Real pretrained multimodal repair']
(root/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
