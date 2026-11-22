"""Reproduce compact damage adapters and prove exact equality to saved merged states."""
import argparse,json,torch,gc,hashlib
from pathlib import Path
import numpy as np
from krmm.model import Omni
from .io import dump,environment
from .task import PublicUnit,retain_prompts
from .repairer import train
from .evaluation import snapshot,restore,state_hash
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--sources",nargs="+",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(8);dump(out/"environment.json",environment())
 o=Omni("/root/rivermind-data/models/Qwen2.5-Omni-3B")
 cfg0=json.loads((Path(a.sources[0])/"config.json").read_text());base=snapshot(o,cfg0);teacher=o.retain_logits(retain_prompts())
 private_root=Path("/root/rivermind-data/setup-v02/private-assignments")
 results=[]
 for source in map(Path,a.sources):
  cfg=json.loads((source/"config.json").read_text())
  source_metrics=json.loads((source/"metrics.json").read_text())
  assignment_name=Path(source_metrics.get("assignment_source","association24-dev-01")).name
  private=json.loads((private_root/(assignment_name+".json")).read_text())["targets"]
  rs=sorted([r for p in source.glob("*.json") for r in [json.loads(p.read_text())] if r.get("eligible")],key=lambda r:r["unit"]["uid"])
  for r in rs:
   u=PublicUnit(**r["unit"]);folder=out/source.name/u.uid/"damage-adapter";folder.mkdir(parents=True)
   restore(o,base);torch.manual_seed(cfg["seed"]);np.random.seed(cfg["seed"])
   logs=train(o,u,private[u.uid],retain_prompts(),teacher,cfg,cfg["damage_steps"],cfg["damage_mode"],variants=(0,1))
   o.model.save_pretrained(str(folder),save_embedding_layers=False)
   o.model=o.model.merge_and_unload()
   actual=state_hash(snapshot(o,cfg));expected=r["damage_state_sha256"]
   result={"source":str(source),"uid":u.uid,"entity":u.entity,"expected_state_sha256":expected,"reproduced_state_sha256":actual,"exact_match":actual==expected,"adapter_path":str(folder),"config":cfg}
   dump(folder.parent/"reproduction.json",result);results.append(result)
   print(source.name,u.entity,result["exact_match"],flush=True)
   if actual!=expected:
    # Keep evidence, fail closed; never label an approximate adapter as exact.
    dump(out/"mismatch.json",result);raise RuntimeError("Compact damage reconstruction mismatch")
   # Independent adapter reload+merge from the official base must also equal the saved damaged state.
   from peft import PeftModel
   restore(o,base);o.model=PeftModel.from_pretrained(o.model,str(folder),is_trainable=False);o.model=o.model.merge_and_unload()
   reloaded=state_hash(snapshot(o,cfg))
   if reloaded!=expected:raise RuntimeError("Reloaded compact damage mismatch")
   restore(o,base);gc.collect();torch.cuda.empty_cache()
 dump(out/"metrics.json",{"cases":len(results),"exact_matches":sum(r["exact_match"] for r in results),"all_saved_and_reloaded":True,"resources":o.state(),"scope":"Same model/GPU/code deterministic damage reconstruction; not second-model transfer or G3."})
if __name__=="__main__":main()
