"""Evaluator orchestration: truth is used only for corruption, oracle upper bound and scoring."""
import argparse,copy,hashlib,json,os,random,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from .benchmark import public_units,gold,question,retention_questions,manifest
from .model import Omni
from .repair import diagnose,choose,train,writeability
from peft import PeftModel

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)

def evaluate(omni,u):
    scores=[omni.probabilities(question(u.entity,j))[0] for j in range(4)]
    return {"accuracy":float(np.mean([choose(p)==gold(u) for p in scores])),"probabilities":scores}

def relevant_snapshot(omni,cfg):
    n=omni.model.config.num_hidden_layers
    return {k:p.detach().cpu().clone() for k,p in omni.model.named_parameters()
      if any(f"layers.{j}." in k for j in range(n-cfg["last_layers"],n)) and any(x in k for x in ["q_proj.weight","v_proj.weight","down_proj.weight"])}

def restore(omni,state):
    d=dict(omni.model.named_parameters())
    with torch.no_grad():
        for k,v in state.items():d[k].copy_(v)
    for p in omni.model.parameters():p.requires_grad_(False)
    omni.model.eval()

def one(omni,u,cfg,outdir,oracle=True):
    t=time.monotonic();clean=evaluate(omni,u);retains=retention_questions(u)
    result={"unit":u.__dict__,"gold_evaluator_only":gold(u),"clean":clean,"eligible":False}
    if clean["accuracy"]<1:
        result["exclusion"]="not_stably_mastered";return result
    original=relevant_snapshot(omni,cfg)
    teacher_clean=omni.retain_logits(retains)
    wrong="ABCD"[("ABCD".index(gold(u))+1)%4]
    result["damage_training"]=train(omni,u,wrong,retains,teacher_clean,cfg,cfg["damage_steps"],True)
    omni.model=omni.model.merge_and_unload()
    damaged=evaluate(omni,u);teacher_damaged=omni.retain_logits(retains)
    retain_damage=float(np.mean([int(a.argmax())==int(b.argmax()) for a,b in zip(teacher_clean,teacher_damaged)]))
    result.update(damaged=damaged,damage_retain_agreement=retain_damage)
    if damaged["accuracy"]>0.25 or retain_damage<0.9:
        result["exclusion"]="damage_not_local_or_not_effective";restore(omni,original);return result
    result["eligible"]=True
    damaged_state=relevant_snapshot(omni,cfg)
    # Public repair receives neither gold nor original snapshot.
    diag=diagnose(omni,u,cfg,retains)
    diag["features"][-1]=writeability(omni,u,diag["candidate"],retains,teacher_damaged,cfg)
    result["diagnosis"]=diag
    result["candidate_correct_evaluator_only"]=diag["candidate"]==gold(u)
    for label,answer,constrained in [("constrained",diag["candidate"],True),("unconstrained",diag["candidate"],False)]+([("oracle",gold(u),True)] if oracle else []):
        restore(omni,damaged_state)
        result[label+"_training"]=train(omni,u,answer,retains,teacher_damaged,cfg,cfg["repair_steps"],constrained)
        adapter_path=Path(outdir)/u.uid/label
        omni.model.save_pretrained(adapter_path)
        # Explicitly unload adapter and reload its serialized weights onto unchanged damaged base.
        omni.model=omni.model.unload()
        omni.model=PeftModel.from_pretrained(omni.model,str(adapter_path),is_trainable=False)
        after=evaluate(omni,u);ret=omni.retain_logits(retains)
        agree=float(np.mean([int(a.argmax())==int(b.argmax()) for a,b in zip(teacher_damaged,ret)]))
        after["retain_agreement"]=agree
        after["success"]=bool(after["accuracy"]>=cfg["success_accuracy"] and after["accuracy"]-damaged["accuracy"]>=cfg["minimum_gain"] and agree>=1-cfg["retain_tolerance"])
        after["reloaded_adapter"]=True
        result[label]=after
        omni.model=omni.model.unload()
    restore(omni,original)
    result["seconds"]=time.monotonic()-t;result["resources"]=omni.state()
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/pilot.json");ap.add_argument("--split",default="dev");ap.add_argument("--limit",type=int,default=4);ap.add_argument("--out",required=True);ap.add_argument("--screen-only",action="store_true")
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text())
    random.seed(cfg["seed"]);np.random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"])
    torch.set_num_threads(8)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    dump(out/"config.json",cfg);dump(out/"manifest.json",manifest())
    dump(out/"environment.json",{"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),"source_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"start_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
    omni=Omni(cfg["model_path"])
    units=[u for u in public_units() if u.split==args.split]
    # Balanced ordering across classes for pilot limit.
    units=sorted(units,key=lambda u:(next(i for i,v in enumerate(__import__("krmm.benchmark",fromlist=["GROUPS"]).GROUPS[gold(u)]) if v==u.entity),gold(u)))[:args.limit]
    results=[]
    for u in units:
        print("START",u.entity,flush=True)
        if args.screen_only:r={"unit":u.__dict__,"clean":evaluate(omni,u)}
        else:r=one(omni,u,cfg,out)
        results.append(r);dump(out/(u.uid+".json"),r)
        print("END",u.entity,json.dumps({k:r.get(k) for k in ["eligible","exclusion","clean","damaged","constrained","oracle"]}),flush=True)
    good=[x for x in results if x.get("eligible")]
    metrics={"independent_units":len(results),"eligible_units":len(good),"clean_mastery_rate":float(np.mean([r["clean"]["accuracy"]==1 for r in results])),
      "damage_eligibility_rate":len(good)/max(1,len(results)),"oracle_success_rate":float(np.mean([r["oracle"]["success"] for r in good])) if good else 0,
      "candidate_accuracy":float(np.mean([r["candidate_correct_evaluator_only"] for r in good])) if good else 0,
      "persistent_repair_rate_all_eligible":float(np.mean([r["constrained"]["success"] for r in good])) if good else 0,
      "peak_vram_gb":omni.state()["peak_vram_gb"],"split_verified":len(set(u.entity for u in public_units()))==48}
    dump(out/"metrics.json",metrics);print(json.dumps(metrics),flush=True)
if __name__=="__main__":main()
