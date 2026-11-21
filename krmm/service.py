"""Answer-free selective repair entry point. No evaluator or gold-label imports."""
import argparse,json,hashlib,time
from pathlib import Path
import joblib,numpy as np,torch
from peft import PeftModel
from .task import PublicUnit
from .model import Omni
from .repair import diagnose,writeability,train

def predict(diagnosis,artifact):
    from .features import vector
    x=np.asarray(vector(diagnosis),dtype=float)[None]
    if x.shape[1]!=artifact["feature_count"]:raise ValueError("Feature schema mismatch")
    p=artifact["model"].predict_proba(x)[:,1]
    if artifact.get("calibrator") is None:
        return None,"uncalibrated"
    logits=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))[:,None]
    return float(artifact["calibrator"].predict_proba(logits)[0,1]),"calibrated_pilot_only"

def run(omni,unit,retains,cfg,out,artifact=None,policy="calibrated"):
    if not retains:raise ValueError("A non-target retention pool is required")
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    teacher=omni.retain_logits(retains)
    d=diagnose(omni,unit,cfg,retains)
    d["features"][-1]=writeability(omni,unit,d["candidate"],retains,teacher,cfg)
    report={"unit":unit.__dict__,"diagnosis":d,"probability":None,"decision":"uncertain","policy":policy,"accepted_adapter":None}
    if policy=="calibrated":
        if artifact is None:
            report["reason"]="No independently calibrated diagnosticator supplied"
            (out/"decision.json").write_text(json.dumps(report,indent=2));return report
        probability,status=predict(d,artifact);report.update(probability=probability,calibration_status=status)
        accept=probability is not None and probability>=artifact["threshold"]
    elif policy=="support":
        c=d.get("constraint_trace") or {}
        accept=c.get("unique_support",False) and not c.get("conflict",True)
        report["calibration_status"]="uncalibrated_structural_baseline"
    else:raise ValueError("Unknown selection policy")
    if not accept:
        report["decision"]="abstain"
        (out/"decision.json").write_text(json.dumps(report,indent=2));return report
    report["training"]=train(omni,unit,d["candidate"],retains,teacher,cfg,cfg["repair_steps"],True)
    attempted=out/"attempted-adapter";omni.model.save_pretrained(attempted,save_embedding_layers=False)
    omni.model=omni.model.unload();omni.model=PeftModel.from_pretrained(omni.model,str(attempted),is_trainable=False);omni.model.eval()
    after=omni.retain_logits(retains)
    agreement=float(np.mean([int(a.argmax())==int(b.argmax()) for a,b in zip(teacher,after)]))
    report["retain_agreement"]=agreement
    if agreement<1-cfg["retain_tolerance"]:
        omni.model=omni.model.unload();report["decision"]="rollback"
    else:
        report["decision"]="repaired_unverified_target";report["accepted_adapter"]=str(attempted.resolve())
    report["resources"]=omni.state()
    report["claim_boundary"]="Target truth unavailable: successful target recovery cannot be certified by this service; independent evaluator required."
    (out/"decision.json").write_text(json.dumps(report,indent=2));return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--entity",required=True);ap.add_argument("--retains",required=True);ap.add_argument("--config",default="configs/pilot-constraints.json");ap.add_argument("--damaged-weight-patch");ap.add_argument("--diagnosticator");ap.add_argument("--policy",choices=["calibrated","support"],default="calibrated");ap.add_argument("--out",required=True)
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text());omni=Omni(cfg["model_path"])
    if args.damaged_weight_patch:
        # Benchmark loader only: patch is a complete set of merged changed weights, not a known subtractable damage adapter.
        state=torch.load(args.damaged_weight_patch,map_location="cpu",weights_only=True);params=dict(omni.model.named_parameters())
        with torch.no_grad():
            for k,v in state.items():params[k].copy_(v)
        del state
    unit=PublicUnit(hashlib.sha256(args.entity.encode()).hexdigest()[:12],args.entity,"deployment")
    retains=json.loads(Path(args.retains).read_text());artifact=joblib.load(args.diagnosticator) if args.diagnosticator else None
    r=run(omni,unit,retains,cfg,args.out,artifact,args.policy)
    print(json.dumps({k:v for k,v in r.items() if k not in ["training"]},indent=2))
if __name__=="__main__":main()
