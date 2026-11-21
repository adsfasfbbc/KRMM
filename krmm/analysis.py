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
    rows.sort(key=lambda r:r["unit"]["uid"])  # Label-independent deterministic tie order.
    tr=[r for r in rows if r["unit"]["split"]=="dev"]
    cal=[r for r in rows if r["unit"]["split"]=="calibration"]
    te=[r for r in rows if r["unit"]["split"]=="test"]
    uids=[r["unit"]["uid"] for r in rows]
    if len(uids)!=len(set(uids)):raise ValueError("Duplicate entities across run roots; do not pool exploratory repeats as independent samples")
    report={"dev":len(tr),"calibration":len(cal),"test":len(te),"status":"INCONCLUSIVE"}
    report["descriptive_by_split"]={}
    for name,rr in [("dev",tr),("calibration",cal),("test",te)]:
        if rr:
            report["descriptive_by_split"][name]={"eligible":len(rr),"candidate_accuracy":float(np.mean([r["candidate_correct_evaluator_only"] for r in rr])),"constrained_success":float(np.mean([r["constrained"]["success"] for r in rr])),"unconstrained_success":float(np.mean([r["unconstrained"]["success"] for r in rr])),"oracle_success":float(np.mean([r["oracle"]["success"] for r in rr])),"mean_retained_token_agreement":float(np.mean([r["constrained"]["retain_agreement"] for r in rr]))}
    if te:
        yy=np.array([r["constrained"]["success"] for r in te]);k=max(1,len(te)//2)
        report["simple_selectors_half_coverage"]={}
        for name,j,sign in [("confidence",0,1),("negative_entropy",1,-1),("consistency",2,1),("support_margin",3,1)]:
            score=np.array([r["diagnosis"]["features"][j]*sign for r in te]);ix=np.argsort(-score,kind="stable")[:k]
            report["simple_selectors_half_coverage"][name]={"coverage":k/len(te),"successful_repairs_over_all_eligible":float(yy[ix].sum()/len(te))}
    if len(tr)<8 or len(cal)<8 or len(te)<12 or len({r["constrained"]["success"] for r in tr})<2:
        report["reason"]="Insufficient independent units or degenerate outcome labels; no AUROC or invented fit."
    else:
        from .features import vector
        X=lambda rr:np.array([vector(r["diagnosis"]) for r in rr])
        Y=lambda rr:np.array([r["constrained"]["success"] for r in rr],dtype=int)
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,random_state=17,max_iter=2000)).fit(X(tr),Y(tr))
        pc=model.predict_proba(X(cal))[:,1];pt=model.predict_proba(X(te))[:,1]
        calibration_status="unavailable_single_class"
        if len(set(Y(cal)))==2:
            logit=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))[:,None]
            calibrator=LogisticRegression(C=1,random_state=17).fit(logit(pc),Y(cal))
            pt=calibrator.predict_proba(logit(pt))[:,1]
            calibration_status="Platt_fit_on_disjoint_calibration_entities"
        report["calibration_status"]=calibration_status
        cal_for_threshold=model.predict_proba(X(cal))[:,1]
        if calibration_status.startswith("Platt"):
            cal_for_threshold=calibrator.predict_proba(logit(cal_for_threshold))[:,1]
        threshold=float(np.median(cal_for_threshold))
        import joblib
        artifact=Path(args.out).parent/"diagnosticator.joblib"
        joblib.dump({"model":model,"calibrator":calibrator if calibration_status.startswith("Platt") else None,"threshold":threshold,"calibration_status":calibration_status,"feature_count":X(tr).shape[1]},artifact)
        deployed=pt>=threshold
        report["calibration_median_threshold"]={"threshold":threshold,"test_coverage":float(deployed.mean()),"successful_repairs_over_all_eligible":float((deployed*Y(te)).mean())}
        report["outcome_rates"]={name:float(np.mean([r[name]["success"] for r in te])) for name in ["constrained","unconstrained","oracle"]}
        y=Y(te);k=max(1,len(te)//2)
        selectors={"diagnosis":pt,"confidence":np.array([r["diagnosis"]["features"][0] for r in te]),"consistency":np.array([r["diagnosis"]["features"][2] for r in te]),"random":np.random.default_rng(17).random(len(te))}
        # Strongest single feature and its direction selected ONLY on calibration entities.
        singles=[]
        if len(set(Y(cal)))==2:
            for j in range(X(cal).shape[1]):
                auc=roc_auc_score(Y(cal),X(cal)[:,j]);direction=1 if auc>=.5 else -1
                singles.append((max(auc,1-auc),j,direction))
            best=max(singles)
            selectors["strongest_single"]=X(te)[:,best[1]]*best[2]
            report["strongest_single_selected_on_calibration"]={"index":best[1],"direction":best[2],"calibration_auroc":best[0]}
        selectors["negative_entropy"]=-X(te)[:,1]
        selectors["candidate_change"]=X(te)[:,-1]
        report["predictor_metrics"]={name:{"auroc":float(roc_auc_score(y,score)) if len(set(y))>1 else None,"auprc":float(average_precision_score(y,score)) if sum(y) else None} for name,score in selectors.items()}
        report["fixed_coverage"]={}
        selections={}
        for name,score in selectors.items():
            mask=np.zeros(len(te));mask[np.argsort(-score,kind="stable")[:k]]=1
            success=mask*y
            selections[name]=success
            report["fixed_coverage"][name]={"coverage":float(mask.mean()),"success_rate_all_eligible":float(success.mean()),"accepted_failure":float((mask*(1-y)).sum()/k)}
            if name=="diagnosis":primary=success
            elif name=="confidence":comparison=success
        report["paired_vs_all_selectors"]={name:bootstrap_delta(selections["diagnosis"],values) for name,values in selections.items() if name!="diagnosis"}
        report.update(status="COMPLETE",auroc=float(roc_auc_score(y,pt)) if len(set(y))>1 else None,
          auprc=float(average_precision_score(y,pt)),brier=float(brier_score_loss(y,pt)),
          paired_diagnosis_vs_confidence=bootstrap_delta(primary,comparison))
    Path(args.out).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
