"""P0 evaluator coordinator. No target truth crosses repair-facing interfaces."""
import argparse,json,time,secrets,gc,subprocess,hashlib,os
from pathlib import Path
import numpy as np,torch
from peft import PeftModel
from krmm.model import Omni
from .task import PublicUnit,question,retain_prompts,OPTIONS
from .io import dump,rows,environment
from .evaluation import choose,evaluate,snapshot,restore,agreement,state_hash,wrong_assignments
from .repairer import train,repair
from .extractor import extract
from .reconstruct import reconstruct
from .reliability import features,FEATURES
def diagnose(omni,u):
 direct=[omni.probabilities(question(u.entity,j))[0] for j in (2,3)]
 facts,records=extract(omni,u.entity);structured=reconstruct(facts)
 texts=[r["text"] for r in records]
 prompt="Infer the Hornbostel-Sachs family using only these observations:\n"+"\n".join(texts)+"\n"+"\n".join(k+": "+v for k,v in OPTIONS.items())+"\nReply with one letter only."
 generative=omni.probabilities(prompt)[0]
 return {"candidate":structured["candidate"],"structured":structured,"facts":[f.serial() for f in facts],"probes":records,"features":features(facts),"feature_names":FEATURES,
 "direct":direct,"direct_candidate":choose(direct[0]),"generative":generative,"generative_candidate":choose(generative),
 "candidate_change":int(structured["candidate"] is not None and structured["candidate"]!=choose(direct[0])),
 "scope":"internal residual knowledge only; source labels never supplied"}
def one(o,r,cfg,out,wrong):
 u=PublicUnit(r["uid"],r["entity"],r["split"]);t=time.monotonic()
 res={"unit":u.__dict__,"gold_evaluator_only":r["gold"],"eligible":False}
 res["clean"]=evaluate(o,u.entity,r["gold"],(0,1))
 if res["clean"]["accuracy"]!=1:res["exclusion"]="not_stably_mastered";return res
 original=snapshot(o,cfg);retains=retain_prompts();tc=o.retain_logits(retains)
 res["damage_training"]=train(o,u,wrong,retains,tc,cfg,cfg["damage_steps"],cfg["damage_mode"],variants=(0,1))
 o.model=o.model.merge_and_unload()
 res["damage_screen"]=evaluate(o,u.entity,r["gold"],(0,1))
 td=o.retain_logits(retains);res["damage_retention"]=agreement(tc,td)
 # Evaluator-only hidden outcome never enters eligibility.
 if res["damage_screen"]["accuracy"]>0 or res["damage_retention"]<0.9:
  res["exclusion"]="damage_not_local_or_effective";restore(o,original);return res
 res["eligible"]=True;damaged=snapshot(o,cfg);res["damaged"]=evaluate(o,u.entity,r["gold"],(4,5))
 d=diagnose(o,u);res["diagnosis"]=d
 folder=out/u.uid;folder.mkdir()
 torch.save(damaged,folder/"damaged_state.pt");torch.save(td,folder/"teacher.pt")
 res["damage_state_sha256"]=state_hash(damaged);res["same_damage_for_all_branches"]=True
 branches={"oracle":r["gold"],"generative":d["generative_candidate"]}
 if d["candidate"] is not None:branches["structured"]=d["candidate"]
 local={};adapter_paths={}
 for name,candidate in branches.items():
  restore(o,damaged)
  res[name+"_training"]=repair(o,u,candidate,retains,td,cfg,str(folder/name))
  o.model=o.model.unload();o.model=PeftModel.from_pretrained(o.model,str(folder/name),is_trainable=False)
  ev=evaluate(o,u.entity,r["gold"],(4,5));ev["retention_agreement"]=agreement(td,o.retain_logits(retains));local[name]=ev
  o.model=o.model.unload();adapter_paths[name]=str((folder/name).resolve())
 # Every eligible unit is written counterfactually, irrespective of selector acceptance.
 job={"model_path":cfg["model_path"],"entity":u.entity,"gold":r["gold"],"damaged_accuracy":res["damaged"]["accuracy"],
 "damaged_state":str((folder/"damaged_state.pt").resolve()),"teacher":str((folder/"teacher.pt").resolve()),
 "adapters":adapter_paths,"output":str((folder/"fresh-evaluation.json").resolve())}
 dump(folder/"verify-job.json",job)
 o.model.eval();torch.cuda.empty_cache()
 with (folder/"verify.log").open("w") as log:
  p=subprocess.run([os.sys.executable,"-m","krmm.v2.verify","--job",str(folder/"verify-job.json")],stdout=log,stderr=subprocess.STDOUT,timeout=480)
 if p.returncode:raise RuntimeError(f"Fresh verification failed: {folder}")
 fresh=json.loads((folder/"fresh-evaluation.json").read_text())
 for name,ev in fresh.items():
  error=float(np.max(np.abs(np.array(local[name]["probabilities"])-np.array(ev["probabilities"]))))
  ev["probability_reload_max_error"]=error;ev["fresh_process_verified"]=error<=1e-6
  ev["success"]=ev["success"] and ev["fresh_process_verified"]
 res["repairs"]=fresh;res["seconds"]=time.monotonic()-t
 restore(o,original);gc.collect();torch.cuda.empty_cache()
 return res
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--out",required=True);ap.add_argument("--split",default="dev");ap.add_argument("--limit",type=int,default=16);a=ap.parse_args()
 cfg=json.loads(Path(a.config).read_text());torch.set_num_threads(8);torch.manual_seed(cfg["seed"]);np.random.seed(cfg["seed"])
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);dump(out/"config.json",cfg);dump(out/"environment.json",environment())
 candidates=rows(a.split)[:a.limit]
 private_seed=secrets.randbits(64);assignment,audit=wrong_assignments(candidates,private_seed)
 private=Path("/root/rivermind-data/setup-v02/private-assignments");private.mkdir(exist_ok=True)
 path=private/(out.name+".json");path.write_text(json.dumps({"seed":private_seed,"targets":assignment}));path.chmod(0o600)
 dump(out/"damage-randomization-audit.json",audit)
 o=Omni(cfg["model_path"]);results=[]
 for r in candidates:
  print("START",r["entity"],flush=True);result=one(o,r,cfg,out,assignment[r["uid"]]);results.append(result);dump(out/(r["uid"]+".json"),result)
  print("END",r["entity"],result["eligible"],result.get("exclusion"),result.get("repairs",{}).get("oracle",{}).get("success"),flush=True)
 mastered=sum(x["clean"]["accuracy"]==1 for x in results);good=[x for x in results if x["eligible"]]
 m={"independent_units":len(good),"survey_units":len(results),"clean_mastery_rate":mastered/len(results),"damage_eligibility_rate":len(good)/max(1,mastered),"eligible_units":len(good),
 "oracle_success_rate":float(np.mean([x["repairs"]["oracle"]["success"] for x in good])) if good else 0,
 "retention_agreement":min([x["damage_retention"] for x in good] or [0]),
 "split_verified":len({r["group"] for r in rows(a.split)})==len(rows(a.split)),"no_inverse_shortcut":audit["no_deterministic_inverse"],
 "candidate_accuracy_structured_all_eligible":sum(x["diagnosis"]["candidate"]==x["gold_evaluator_only"] for x in good)/max(1,len(good)),
 "candidate_coverage_structured":sum(x["diagnosis"]["candidate"] is not None for x in good)/max(1,len(good)),
 "candidate_accuracy_generative":sum(x["diagnosis"]["generative_candidate"]==x["gold_evaluator_only"] for x in good)/max(1,len(good)),
 "resources":o.state(),"status":"P0_ONLY; not confirmatory G1"}
 dump(out/"metrics.json",m);print(json.dumps(m),flush=True)
if __name__=="__main__":main()
