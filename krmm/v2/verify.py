"""Fresh process evaluator; no evidence/probe inputs."""
import argparse,json,torch
from pathlib import Path
from peft import PeftModel
from krmm.model import Omni
from .evaluation import restore,evaluate,agreement
from .task import retain_prompts
from .io import dump
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--job",required=True);a=ap.parse_args();d=json.loads(Path(a.job).read_text())
 torch.set_num_threads(8);o=Omni(d["model_path"])
 restore(o,torch.load(d["damaged_state"],map_location="cpu",weights_only=True))
 teachers=torch.load(d["teacher"],map_location="cpu",weights_only=True)
 out={}
 for name,adapter in d["adapters"].items():
  o.model=PeftModel.from_pretrained(o.model,adapter,is_trainable=False)
  ev=evaluate(o,d["entity"],d["gold"],(4,5));ev["retention_agreement"]=agreement(teachers,o.retain_logits(retain_prompts()))
  ev["success"]=bool(ev["accuracy"]==1 and ev["accuracy"]-d["damaged_accuracy"]>=0.5 and ev["retention_agreement"]>=0.9)
  ev["rollback_required"]=ev["retention_agreement"]<0.9
  out[name]=ev;o.model=o.model.unload()
 dump(d["output"],out)
if __name__=="__main__":main()
