"""Fresh-process audit of saved damaged weights and serialized repair adapters."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from .model import Omni
from .task import PublicUnit,question
from .benchmark import retention_questions
from .repair import choose

def copy_weights(model,state):
    params=dict(model.named_parameters())
    with torch.no_grad():
        for k,v in state.items():params[k].copy_(v)
    model.eval()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("roots",nargs="+");ap.add_argument("--out",required=True);ap.add_argument("--config",default="configs/pilot-retain50.json");args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text());torch.set_num_threads(8)
    omni=Omni(cfg["model_path"]);original=None;results=[]
    for root in args.roots:
        for p in sorted(Path(root).glob("*.json")):
            r=json.loads(p.read_text())
            if not isinstance(r,dict) or not r.get("eligible"):continue
            u=PublicUnit(**r["unit"]);checkpoint=Path(root)/u.uid
            damaged=torch.load(checkpoint/"damaged_state.pt",map_location="cpu",weights_only=True)
            if original is None:original={k:dict(omni.model.named_parameters())[k].detach().cpu().clone() for k in damaged}
            copy_weights(omni.model,damaged)
            teacher=torch.load(checkpoint/"retain_teacher.pt",map_location="cpu",weights_only=True)
            for label in ["constrained","unconstrained","oracle"]:
                omni.model=PeftModel.from_pretrained(omni.model,str(checkpoint/label),is_trainable=False);omni.model.eval()
                scores=[omni.probabilities(question(u.entity,j))[0] for j in r[label].get("variants",list(range(4)))]
                ret=omni.retain_logits(retention_questions(u))
                agreement=float(np.mean([int(a.argmax())==int(b.argmax()) for a,b in zip(teacher,ret)]))
                error=float(np.max(np.abs(np.array(scores)-np.array(r[label]["probabilities"]))))
                results.append({"uid":u.uid,"entity":u.entity,"split":u.split,"label":label,"probabilities":scores,
                  "retention_agreement":agreement,"max_probability_difference_vs_training_process":error,
                  "predictions_equal":all(choose(a)==choose(b) for a,b in zip(scores,r[label]["probabilities"])),
                  "fresh_process":True})
                omni.model=omni.model.unload()
            copy_weights(omni.model,original)
            print("VERIFIED",u.entity,flush=True)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    report={"n_units":len({r["uid"] for r in results}),"all_predictions_equal":all(r["predictions_equal"] for r in results) if results else False,
      "max_probability_difference":max((r["max_probability_difference_vs_training_process"] for r in results),default=None),
      "rows":results,"resources":omni.state(),"claim_boundary":"Actual fresh process loads official base weights + merged damaged selected weights + saved adapter. No evidence is supplied at test. Only same-checkpoint functional persistence, not proof of original representation recovery."}
    out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
if __name__=="__main__":main()
