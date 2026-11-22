"""P0 paired replay; identical damaged weights, probes and per-branch writer initialization."""
import argparse,json,time,subprocess,os,gc
from pathlib import Path
import numpy as np,torch
from peft import PeftModel
from krmm.model import Omni
from .task import PublicUnit,retain_prompts
from .io import dump,environment
from .evaluation import restore,evaluate,agreement,state_hash
from .repairer import repair
from .extractor_canonical import replay_records
from .reconstruct import reconstruct
from .reliability import features,FEATURES
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--source",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
 source=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(8);cfg=json.loads((source/"config.json").read_text());dump(out/"config.json",cfg);dump(out/"environment.json",environment())
 o=Omni(cfg["model_path"]);results=[]
 rs=sorted([r for p in source.glob("*.json") for r in [json.loads(p.read_text())] if r.get("eligible")],key=lambda r:r["unit"]["uid"])
 for r in rs:
  u=PublicUnit(**r["unit"]);start=time.monotonic();folder=out/u.uid;folder.mkdir()
  facts,traces=replay_records(u.entity,r["diagnosis"]["probes"]);g=reconstruct(facts)
  result={"unit":r["unit"],"eligible":True,"source":str(source),"gold_evaluator_only":r["gold_evaluator_only"],"candidate":g["candidate"],"structured":g,"facts":[f.serial() for f in facts],"normalization_traces":traces,"features":features(facts),"feature_names":FEATURES,"candidate_change":int(g["candidate"] is not None and g["candidate"]!=r["diagnosis"]["direct_candidate"]),"mode":"P0 engineering; not G1"}
  dump(folder/"prewrite.json",result)
  state=torch.load(source/u.uid/"damaged_state.pt",map_location="cpu",weights_only=True)
  assert state_hash(state)==r["damage_state_sha256"]
  td=torch.load(source/u.uid/"teacher.pt",map_location="cpu",weights_only=True)
  branches={"generative":r["diagnosis"]["generative_candidate"],"oracle":r["gold_evaluator_only"]}
  if g["candidate"] is not None:branches["structured"]=g["candidate"]
  local={};adapters={}
  for name,candidate in branches.items():
   restore(o,state);torch.manual_seed(cfg["seed"]);np.random.seed(cfg["seed"])
   logs=repair(o,u,candidate,retain_prompts(),td,cfg,str(folder/name));dump(folder/(name+"-training.json"),logs)
   o.model=o.model.unload();o.model=PeftModel.from_pretrained(o.model,str(folder/name),is_trainable=False)
   ev=evaluate(o,u.entity,r["gold_evaluator_only"],(4,5));ev["retention_agreement"]=agreement(td,o.retain_logits(retain_prompts()))
   local[name]=ev;o.model=o.model.unload();adapters[name]=str((folder/name).resolve())
  torch.cuda.empty_cache()
  job={"model_path":cfg["model_path"],"entity":u.entity,"gold":r["gold_evaluator_only"],"damaged_accuracy":r["damaged"]["accuracy"],"damaged_state":str((source/u.uid/"damaged_state.pt").resolve()),"teacher":str((source/u.uid/"teacher.pt").resolve()),"adapters":adapters,"output":str((folder/"fresh-evaluation.json").resolve())}
  dump(folder/"verify-job.json",job)
  with (folder/"verify.log").open("w") as log:p=subprocess.run([os.sys.executable,"-m","krmm.v2.verify","--job",str(folder/"verify-job.json")],stdout=log,stderr=subprocess.STDOUT,timeout=480)
  if p.returncode:raise RuntimeError("fresh verification failed")
  fresh=json.loads((folder/"fresh-evaluation.json").read_text())
  for name,ev in fresh.items():
   error=float(np.max(np.abs(np.array(local[name]["probabilities"])-np.array(ev["probabilities"]))))
   ev.update(probability_reload_max_error=error,fresh_process_verified=error<=1e-6,writer_initialization_seed=cfg["seed"])
   ev["success"]=ev["success"] and ev["fresh_process_verified"];ev["restoration_valid"]=ev["success"] and r["clean_heldout_audit"]["accuracy"]==1
  result["repairs"]=fresh
  if "structured" in fresh:result["repair"]=fresh["structured"]
  result["seconds"]=time.monotonic()-start;dump(out/(u.uid+".json"),result);results.append(result)
  del state;gc.collect();torch.cuda.empty_cache()
  print(u.entity,g["candidate"],{k:v["success"] for k,v in fresh.items()},flush=True)
 dump(out/"metrics.json",{"eligible_units":len(results),"candidates":sum(r["candidate"] is not None for r in results),"correct_candidates":sum(r["candidate"]==r["gold_evaluator_only"] for r in results),"persistent_successes":sum(r.get("repair",{}).get("success",False) for r in results),"strict_restorations":sum(r.get("repair",{}).get("restoration_valid",False) for r in results),"mode":"P0 only; three branches share seed and damaged state","resources":o.state()})
if __name__=="__main__":main()
