"""Export the already fitted frozen diagnostic; no new fit or parameter search."""
import json,hashlib,datetime
from pathlib import Path
import numpy as np,joblib
from krmm.features import vector
artifact=joblib.load('results/diagnosticator.joblib');rows=[]
for p in sorted(Path('results/v2-test').glob('*.json')):
 r=json.loads(p.read_text())
 if isinstance(r,dict) and r.get('eligible'):rows.append(r)
rows.sort(key=lambda r:r['unit']['uid'])
x=np.array([vector(r['diagnosis']) for r in rows]);p=artifact['model'].predict_proba(x)[:,1]
if artifact['calibrator'] is not None:
 logit=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))[:,None];p=artifact['calibrator'].predict_proba(logit)[:,1]
out=[]
for r,prob in zip(rows,p):out.append({'uid':r['unit']['uid'],'entity':r['unit']['entity'],'candidate':r['diagnosis']['candidate'],'features_preupdate':vector(r['diagnosis']),'predicted_success_probability':float(prob),'calibration_threshold_accept':bool(prob>=artifact['threshold']),'counterfactual_persistent_success_evaluator_only':r['constrained']['success']})
Path('results/diagnosis-predictions.json').write_text(json.dumps(out,indent=2))
model=artifact['model'].named_steps['logisticregression'];scaler=artifact['model'].named_steps['standardscaler'];cal=artifact['calibrator']
meta={'feature_names':rows[0]['diagnosis']['feature_names']+['candidate_changes_direct_answer'],'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'classifier_coef':model.coef_.tolist(),'classifier_intercept':model.intercept_.tolist(),'calibrator_coef':cal.coef_.tolist() if cal is not None else None,'calibrator_intercept':cal.intercept_.tolist() if cal is not None else None,'threshold':artifact['threshold'],'calibration_status':artifact['calibration_status'],'scope':'One conditional, curator-labeled instrument pilot; not broad calibration guarantee','joblib_sha256':hashlib.sha256(Path('results/diagnosticator.joblib').read_bytes()).hexdigest()}
Path('artifacts').mkdir(exist_ok=True);Path('artifacts/diagnosticator.json').write_text(json.dumps(meta,indent=2))
inputs=[]
for root in ['v2-dev','v2-calibration','v3-calibration-extension','v2-test']:
 for file in sorted((Path('results')/root).glob('*.json')):inputs.append({'file':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
Path('results/analysis-provenance.json').write_text(json.dumps({'kind':'fixed preregistered CPU postprocessing of completed model trials; no new model queries or hyperparameter search','source_sha256':hashlib.sha256(Path('krmm/analysis.py').read_bytes()).hexdigest(),'feature_source_sha256':hashlib.sha256(Path('krmm/features.py').read_bytes()).hexdigest(),'inputs':inputs},indent=2))
print('EXPORTED',len(out),'test decisions; no model queries')
