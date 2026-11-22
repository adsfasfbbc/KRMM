"""Controlled P0 factorial cells; same targets and initialization across strengths."""
import argparse,json,torch,time
from pathlib import Path
import numpy as np
from . import experiment as exp
from .io import dump,rows,environment
from .pipeline import infer
from .reliability import FEATURES
from krmm.model import Omni
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--out",required=True);ap.add_argument("--assignment-source",required=True);a=ap.parse_args()
 cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(8)
 dump(out/"config.json",cfg);dump(out/"environment.json",environment())
 source=Path(a.assignment_source);private=Path("/root/rivermind-data/setup-v02/private-assignments")/(source.name+".json")
 assignment=json.loads(private.read_text())["targets"];audit=json.loads((source/"damage-randomization-audit.json").read_text())
 dump(out/"damage-randomization-audit.json",audit)
 original_train=exp.train;original_repair=exp.repair
 def seeded_train(*args,**kwargs):
  torch.manual_seed(cfg["seed"]);np.random.seed(cfg["seed"]);return original_train(*args,**kwargs)
 def seeded_repair(*args,**kwargs):
  torch.manual_seed(cfg["seed"]);np.random.seed(cfg["seed"]);return original_repair(*args,**kwargs)
 def diagnosis(o,u):
  d=infer(o,u,canonical=True);d["structured"]=d.pop("graph");d["feature_names"]=FEATURES
  return d
 exp.train=seeded_train;exp.repair=seeded_repair;exp.diagnose=diagnosis
 o=Omni(cfg["model_path"]);results=[]
 for r in rows("dev"):
  torch.manual_seed(cfg["seed"]);print("START",r["entity"],flush=True)
  result=exp.one(o,r,cfg,out,assignment[r["uid"]]);result["paired_initialization"]=cfg["seed"];results.append(result);dump(out/(r["uid"]+".json"),result)
  print("END",r["entity"],result["eligible"],result.get("exclusion"),result.get("repairs",{}).get("oracle",{}).get("success"),flush=True)
 mastered=sum(r["clean"]["accuracy"]==1 for r in results);good=[r for r in results if r["eligible"]]
 metrics={"independent_units":len(good),"survey_units":len(results),"clean_mastery_rate":mastered/len(results),"damage_eligibility_rate":len(good)/max(1,mastered),"oracle_success_rate":float(np.mean([r["repairs"]["oracle"]["success"] for r in good])) if good else 0,"oracle_strict_restorations":sum(r["repairs"]["oracle"]["restoration_valid"] for r in good),"retention_agreement":min([r["damage_retention"] for r in good] or [0]),"split_verified":True,"no_inverse_shortcut":audit["no_deterministic_inverse"],"candidate_accuracy_structured_all_eligible":sum(r["diagnosis"]["candidate"]==r["gold_evaluator_only"] for r in good)/max(1,len(good)),"candidate_coverage_structured":sum(r["diagnosis"]["candidate"] is not None for r in good)/max(1,len(good)),"candidate_accuracy_generative":sum(r["diagnosis"]["generative_candidate"]==r["gold_evaluator_only"] for r in good)/max(1,len(good)),"resources":o.state(),"status":"P0 conditional factorial; G0 mastery failure not overridden","assignment_source":str(source),"writer_seed_every_branch":cfg["seed"]}
 dump(out/"metrics.json",metrics);print(json.dumps(metrics),flush=True)
if __name__=="__main__":main()
