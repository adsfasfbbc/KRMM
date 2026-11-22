"""Evaluator-only descriptive report; never promotes failed G0 or opens holdout."""
import argparse,json,math
from pathlib import Path
import numpy as np
from .io import dump
from .reliability import fit_ere,probability,FEATURES
from .selector import select
def ci(x,nboot=20000):
 x=np.asarray(x,dtype=float)
 if not len(x):return [None,None]
 rng=np.random.default_rng(107);means=x[rng.integers(0,len(x),(nboot,len(x)))].mean(1)
 return [float(v) for v in np.quantile(means,[.025,.975])]
def metric(y,p):
 from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
 return {"n":len(y),"positives":int(sum(y)),"AUROC":float(roc_auc_score(y,p)) if len(set(y))>1 else None,"AUPRC":float(average_precision_score(y,p)) if len(set(y))>1 else None,"Brier":float(brier_score_loss(y,p)) if len(y) else None}
def loo(X,y):
 pred=[]
 for i in range(len(y)):
  mask=np.arange(len(y))!=i
  if mask.sum()==0:pred.append(.5);continue
  pred.append(float(probability(fit_ere(np.array(X)[mask],np.array(y)[mask]),[X[i]])[0]))
 return pred
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--source",required=True);ap.add_argument("--replay");ap.add_argument("--out",required=True);a=ap.parse_args()
 source=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 raw=[json.loads(p.read_text()) for p in source.glob("*.json")]
 raw=[r for r in raw if "unit" in r];eligible=sorted([r for r in raw if r.get("eligible")],key=lambda r:r["unit"]["uid"])
 rep={}
 if a.replay:
  rep={r["unit"]["uid"]:r for p in Path(a.replay).glob("*.json") for r in [json.loads(p.read_text())] if "unit" in r}
 rs=[]
 for r in eligible:
  d=rep.get(r["unit"]["uid"],{})
  c=d.get("candidate",r["diagnosis"]["candidate"]);phi=d.get("features",r["diagnosis"]["features"]);ev=d.get("repair",r.get("repairs",{}).get("structured",{}))
  rs.append({"uid":r["unit"]["uid"],"entity":r["unit"]["entity"],"candidate":c,"correct":int(c==r["gold_evaluator_only"]),"features":phi,"candidate_change":int(c is not None and c!=r["diagnosis"]["direct_candidate"]),"direct_confidence":max(r["diagnosis"]["direct"][0]),"repair_success":ev.get("success",False),"strict_restoration":ev.get("restoration_valid",False),"bad_pseudolabel":c is not None and c!=r["gold_evaluator_only"],"wrong_retained_write":c is not None and c!=r["gold_evaluator_only"] and not ev.get("rollback_required",True),"generative_correct":int(r["diagnosis"]["generative_candidate"]==r["gold_evaluator_only"]),"oracle_success":d.get("repairs",r["repairs"])["oracle"]["success"],"oracle_restoration":d.get("repairs",r["repairs"])["oracle"].get("restoration_valid",False)})
 valid=[r for r in rs if r["candidate"] is not None];X=[r["features"] for r in valid];y=[r["correct"] for r in valid]
 ere=loo(X,y) if valid else [];full=loo([x+[r["candidate_change"],r["direct_confidence"]] for x,r in zip(X,valid)],y) if valid else []
 for r,e,f in zip(valid,ere,full):r.update(ere_loo=e,full_loo=f)
 for r in rs:
  r.setdefault("ere_loo",0);r.setdefault("full_loo",0)
 curves=[]
 for cov in [.25,.5,.75,1.]:
  for score in ["candidate_change","ere_loo","full_loo"]:
   acts=select(rs,cov,score);chosen=[r for r in rs if acts[r["uid"]]=="attempt"]
   curves.append({"budget_coverage":cov,"score":score,"realized_coverage":len(chosen)/max(1,len(rs)),"selected":len(chosen),"candidate_risk":sum(r["bad_pseudolabel"] for r in chosen)/max(1,len(chosen)),"persistent_repair_rate_all_eligible":sum(r["repair_success"] for r in chosen)/max(1,len(rs)),"strict_restoration_rate_all_eligible":sum(r["strict_restoration"] for r in chosen)/max(1,len(rs)),"wrong_retained_writes_all_eligible":sum(r["wrong_retained_write"] for r in chosen)/max(1,len(rs))})
 delta=[r["correct"]-r["generative_correct"] for r in rs];q=float(np.mean([x!=0 for x in delta])) if delta else 1
 required=max(128,math.ceil((1.96+.84)**2*q/.1**2))
 primary={}
 for score in ["candidate_change","ere_loo","full_loo"]:
  acts=select(rs,.5,score);primary[score]=np.array([float(acts[r["uid"]]=="attempt" and r["repair_success"]) for r in rs])
 gain=(primary["ere_loo"]-primary["candidate_change"]).tolist()
 metrics={"status":"P0_EXPLORATORY_G1_BLOCKED","all_dev_entities":len(raw),"eligible_entities":len(rs),"valid_candidates":len(valid),"correct_candidates":sum(y),"candidate_accuracy_all_eligible":sum(y)/max(1,len(rs)),"candidate_gain_vs_generative":float(np.mean(delta)) if delta else None,"candidate_gain_CI95":ci(delta),"selection_gain_ERE_LOO_vs_candidate_change":float(np.mean(gain)) if gain else None,"selection_gain_CI95_descriptive":ci(gain),"oracle_successes":sum(r["oracle_success"] for r in rs),"oracle_strict_restorations":sum(r["oracle_restoration"] for r in rs),"minimum_confirmatory_eligible_n_for_10pp_80pct_power":required,"available_sealed_entities":151,"ERE_LOO":metric(y,ere),"candidate_change":metric(y,[r["candidate_change"] for r in valid]),"feature_variance_valid":dict(zip(FEATURES,np.var(X,axis=0).tolist())) if valid else {},"limitations":["Dev-only, small sample, failed original mastery gate; no confirmatory inference.","LOO scores are descriptive, NOT an independently calibrated deployable estimator.","Bootstrap conditions on learned scores; uncertainty understates model-selection/calibration uncertainty.","Clean heldout mastery is audited; successful induction from previously wrong clean answers is not called restoration.","Related entities and museum source correlations limit independence."]}
 dump(out/"metrics.json",metrics);dump(out/"rows.json",rs);dump(out/"risk-curves.json",curves)
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
 for score in ["candidate_change","ere_loo","full_loo"]:
  v=[r for r in curves if r["score"]==score]
  axes[0].plot([r["realized_coverage"] for r in v],[r["candidate_risk"] for r in v],"o-",label=score)
  axes[1].plot([r["realized_coverage"] for r in v],[r["persistent_repair_rate_all_eligible"] for r in v],"o-",label=score)
 axes[0].set(xlabel="Realized coverage",ylabel="Candidate risk",ylim=(-.02,1.02))
 axes[1].set(xlabel="Realized coverage",ylabel="Persistent successes / all eligible",ylim=(-.02,1.02))
 for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.suptitle("P0 development diagnostics only — G1 blocked");fig.savefig(out/"risk-curves.png",dpi=180);plt.close(fig)
 print(json.dumps(metrics),flush=True)
if __name__=="__main__":main()
