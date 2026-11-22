import argparse,json,time
from pathlib import Path
import torch
from krmm.model import Omni
from .io import dump,rows,environment
from .evaluation import evaluate
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--split",default="dev");args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(8)
 dump(out/"environment.json",environment())
 omni=Omni("/root/rivermind-data/models/Qwen2.5-Omni-3B")
 results=[]
 for r in rows(args.split):
  result={"uid":r["uid"],"entity":r["entity"],"gold_evaluator_only":r["gold"],"screen":evaluate(omni,r["entity"],r["gold"],(0,1))}
  # Independent clean validation is recorded separately, not used to select prompts or labels.
  result["independent_clean"]=evaluate(omni,r["entity"],r["gold"],(4,5))
  results.append(result);dump(out/(r["uid"]+".json"),result)
  print(r["entity"],result["screen"]["accuracy"],result["independent_clean"]["accuracy"],flush=True)
 mastered=sum(r["screen"]["accuracy"]==1 for r in results)
 dump(out/"metrics.json",{"independent_units":len(results),"clean_mastery_rate":mastered/len(results),"mastered_units":mastered,"independent_clean_rate":sum(r["independent_clean"]["accuracy"]==1 for r in results)/len(results),"resources":omni.state()})
if __name__=="__main__":main()
