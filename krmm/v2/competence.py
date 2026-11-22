"""P0 paired competence diagnostics; cannot establish G1 superiority."""
import argparse,json
from pathlib import Path
import torch,numpy as np
from krmm.model import Omni
from krmm.task import question as legacy_question
from krmm.benchmark import GROUPS
from .task import question
from .io import dump,rows,environment
LABELS=["strings vibrating under tension (chordophone)","wind instrument, air driven, including free reeds (aerophone)","tensioned membrane generating sound (membranophone)","solid body generating sound, not strings or a tensioned membrane (idiophone)"]
def defined(entity,variant,perm=(0,1,2,3)):
 stems=[f"Which of these four acoustic instrument families contains the traditional {entity}?",f"Choose the acoustic organological family of the traditional {entity}."]
 return stems[variant]+"\n"+"\n".join("ABCD"[i]+": "+LABELS[j] for i,j in enumerate(perm))+"\nClassify by sound generation, not playing action or the resonating body. Air-driven free reeds are aerophones. A membrane used only as soundboard does not imply membranophone.\nAnswer with one letter only."
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(8);dump(out/"environment.json",environment());o=Omni("/root/rivermind-data/models/Qwen2.5-Omni-3B")
 old=[{"uid":"old-"+k+str(i),"entity":n,"gold":k,"split":"old_dev_engineering"} for k,names in GROUPS.items() for i,n in enumerate(names[:3])]
 allrows=rows("dev")+old;results=[]
 for idx,r in enumerate(allrows):
  result={"unit":r,"conditions":{}}
  for cond in ["hs_bare","hs_defined","hs_defined_permuted","legacy_source"]:
   predictions=[];masses=[];probs=[];prompts=[]
   for variant in (0,1):
    perm=(2,0,3,1) if cond=="hs_defined_permuted" else (0,1,2,3)
    prompt=question(r["entity"],variant) if cond=="hs_bare" else legacy_question(r["entity"],variant) if cond=="legacy_source" else defined(r["entity"],variant,perm)
    p,m=o.probabilities(prompt);label="ABCD"[perm[int(np.argmax(p))]]
    predictions.append(label);masses.append(m);probs.append(p);prompts.append(prompt)
   result["conditions"][cond]={"predictions":predictions,"choice_mass":masses,"probabilities":probs,"prompts":prompts,"stable_correct":all(x==r["gold"] for x in predictions)}
  # Free output verifies the closed-choice measurement on fixed first8 new entities and old controls.
  if idx<8 or r["split"]=="old_dev_engineering":
   result["free_answer"]=o.generate(defined(r["entity"],0),24)
   result["physical_description"]=o.generate(f"Describe how the traditional {r['entity']} produces sound. Be concise.",64)
  dump(out/(r["uid"]+".json"),result);results.append(result)
  print(r["entity"],{k:v["stable_correct"] for k,v in result["conditions"].items()},flush=True)
 metrics={}
 for split in ["dev","old_dev_engineering"]:
  ss=[r for r in results if r["unit"]["split"]==split]
  metrics[split]={"n":len(ss),"stable_correct_rate":{c:sum(r["conditions"][c]["stable_correct"] for r in ss)/len(ss) for c in result["conditions"]},"mean_choice_mass":{c:float(np.mean([m for r in ss for m in r["conditions"][c]["choice_mass"]])) for c in result["conditions"]}}
 dump(out/"metrics.json",dict(metrics,resources=o.state(),status="P0 exploratory paired competence controls only"))
if __name__=="__main__":main()
