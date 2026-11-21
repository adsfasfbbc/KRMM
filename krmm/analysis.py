"""Entity-held-out diagnosis; evaluate all eligible units, including rejected ones."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss

def bootstrap_delta(a,b,seed=99):
    a,b=np.asarray(a),np.asarray(b);rng=np.random.default_rng(seed)
    d=a-b
    boot=[d[rng.integers(len(d),size=len(d))].mean() for _ in range(2000)]
    return {"delta":float(d.mean()),"ci95":np.quantile(boot,[.025,.975]).tolist(),"unit":"entity","n":len(d)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("runs",nargs="+");ap.add_argument("--out",required=True);args=ap.parse_args()
    rows=[]
    for root in args.runs:
        for p in Path(root).glob("*.json"):
            r=json.loads(p.read_text())
            if isinstance(r,dict) and r.get("eligible"):rows.append(r)
    tr=[r for r in rows if r["unit"]["split"]=="dev"]
    cal=[r for r in rows if r["unit"]["split"]=="calibration"]
    te=[r for r in rows if r["unit"]["split"]=="test"]
    report={"dev":len(tr),"calibration":len(cal),"test":len(te),"status":"INCONCLUSIVE"}
    if len(tr)<8 or len(cal)<8 or len(te)<12 or len({r["constrained"]["success"] for r in tr})<2:
        report["reason"]="Insufficient independent units or degenerate outcome labels; no AUROC or invented fit."
    else:
        X=lambda rr:np.array([r["diagnosis"]["features"] for r in rr])
        Y=lambda rr:np.array([r["constrained"]["success"] for r in rr],dtype=int)
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,random_state=17,max_iter=2000)).fit(X(tr),Y(tr))
        pc=model.predict_proba(X(cal))[:,1];pt=model.predict_proba(X(te))[:,1]
        y=Y(te);k=max(1,len(te)//2)
        selectors={"diagnosis":pt,"confidence":np.array([r["diagnosis"]["features"][0] for r in te]),"consistency":np.array([r["diagnosis"]["features"][2] for r in te]),"random":np.random.default_rng(17).random(len(te))}
        report["fixed_coverage"]={}
        for name,score in selectors.items():
            mask=np.zeros(len(te));mask[np.argsort(-score,kind="stable")[:k]]=1
            success=mask*y
            report["fixed_coverage"][name]={"coverage":float(mask.mean()),"success_rate_all_eligible":float(success.mean()),"accepted_failure":float((mask*(1-y)).sum()/k)}
            if name=="diagnosis":primary=success
            elif name=="confidence":comparison=success
        report.update(status="COMPLETE",auroc=float(roc_auc_score(y,pt)) if len(set(y))>1 else None,
          auprc=float(average_precision_score(y,pt)),brier=float(brier_score_loss(y,pt)),
          paired_diagnosis_vs_confidence=bootstrap_delta(primary,comparison))
    Path(args.out).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
