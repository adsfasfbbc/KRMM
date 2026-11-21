"""Preregistered candidate-change baseline as an answer-free selective repair entry point."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from peft import PeftModel
from .task import PublicUnit
from .model import Omni
from .repair import diagnose,writeability,train,choose

def should_attempt(d):
    return d["candidate"]!=choose(d["direct"][0])

def run(omni,unit,retains,cfg,out):
    if not retains:raise ValueError("Non-target retention prompts required")
    out=Path(out);out.mkdir(parents=True,exist_ok=False);teacher=omni.retain_logits(retains)
    d=diagnose(omni,unit,cfg,retains)
    # Same pre-update probe and RNG consumption as the full research repairer.
    d["features"][-1]=writeability(omni,unit,d["candidate"],retains,teacher,cfg)
    r={"unit":unit.__dict__,"diagnosis":d,"policy":"candidate_change_baseline","probability":None,"accepted_adapter":None,"decision":"abstain"}
    if should_attempt(d):
        r["training"]=train(omni,unit,d["candidate"],retains,teacher,cfg,cfg["repair_steps"],True)
        checkpoint=out/"attempted-adapter";omni.model.save_pretrained(checkpoint,save_embedding_layers=False)
        omni.model=omni.model.unload();omni.model=PeftModel.from_pretrained(omni.model,str(checkpoint),is_trainable=False);omni.model.eval()
        after=omni.retain_logits(retains);agreement=float(np.mean([int(a.argmax())==int(b.argmax()) for a,b in zip(teacher,after)]));r["retain_agreement"]=agreement
        if agreement<1-cfg["retain_tolerance"]:omni.model=omni.model.unload();r["decision"]="rollback"
        else:r["decision"]="repaired_unverified_target";r["accepted_adapter"]=str(checkpoint.resolve())
    r["resources"]=omni.state();r["claim_boundary"]="No target truth is consumed and no target-recovery certification is possible. This simple policy does not output a calibrated probability."
    (out/"decision.json").write_text(json.dumps(r,indent=2));return r

def main():
    import hashlib
    ap=argparse.ArgumentParser();ap.add_argument("--entity",required=True);ap.add_argument("--retains",required=True);ap.add_argument("--config",default="configs/pilot-constraints.json");ap.add_argument("--damaged-weight-patch");ap.add_argument("--out",required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text());torch.manual_seed(cfg["seed"]);omni=Omni(cfg["model_path"])
    if args.damaged_weight_patch:
        weights=torch.load(args.damaged_weight_patch,map_location="cpu",weights_only=True);params=dict(omni.model.named_parameters())
        with torch.no_grad():
            for k,v in weights.items():params[k].copy_(v)
        del weights
    u=PublicUnit(hashlib.sha256(args.entity.encode()).hexdigest()[:12],args.entity,"deployment")
    r=run(omni,u,json.loads(Path(args.retains).read_text()),cfg,args.out);print(json.dumps({k:v for k,v in r.items() if k!="training"},indent=2))
if __name__=="__main__":main()
