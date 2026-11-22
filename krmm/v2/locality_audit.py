"""New independent retention audit; original training-set retention metrics stay immutable."""
import argparse,json,torch,numpy as np,time,gc
from pathlib import Path
from peft import PeftModel
from krmm.model import Omni
from .io import dump,rows,environment
from .task import question
from .evaluation import snapshot,restore
def measure(o,controls):
 ps=[];tops=[];masses=[]
 ids=torch.tensor([o.tokenizer.encode(c,add_special_tokens=False)[0] for c in "ABCD"],device="cuda")
 with torch.inference_mode():
  for i in range(0,len(controls),8):
   batch=o.encode_many([r["question"] for r in controls[i:i+8]])
   logits=o.model(**batch).logits[:,-1].float()
   ps.extend(logits[:,ids].softmax(-1).cpu().tolist());tops.extend(logits.argmax(-1).cpu().tolist());masses.extend((logits[:,ids].logsumexp(-1)-logits.logsumexp(-1)).exp().cpu().tolist())
 return {"probabilities":ps,"top_token":tops,"choice_mass":masses,"prediction":["ABCD"[int(np.argmax(p))] for p in ps]}
def compare(base,after,controls,target):
 selected=[i for i,r in enumerate(controls) if r.get("entity")!=target]
 out={}
 for kind,indices in [('all',selected),('neighbor',[i for i in selected if controls[i]['kind']=='neighbor']),('general',[i for i in selected if controls[i]['kind']=='general'])]:
  agree=float(np.mean([base["top_token"][i]==after["top_token"][i] for i in indices]))
  before=float(np.mean([base["prediction"][i]==controls[i]["gold"] for i in indices]))
  post=float(np.mean([after["prediction"][i]==controls[i]["gold"] for i in indices]))
  out[kind]={"n":len(indices),"token_agreement":agree,"baseline_accuracy":before,"after_accuracy":post,"accuracy_drop":before-post}
 out["independent_locality_pass"]=bool(out["all"]["token_agreement"]>=0.9 and out["neighbor"]["accuracy_drop"]<=0.05+1e-12 and out["general"]["accuracy_drop"]<=0.05+1e-12)
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--sources",nargs="+",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(8)
 cfg=json.loads((Path(a.sources[0])/"config.json").read_text());dump(out/"environment.json",environment())
 controls=[]
 for r in rows("dev"):
  for v in (4,5):controls.append({"uid":r["uid"]+f"-{v}","entity":r["entity"],"question":question(r["entity"],v),"gold":r["gold"],"kind":"neighbor"})
 controls.extend([dict(r,kind="general") for r in json.loads(Path("configs/v2/locality-controls.json").read_text())["controls"]])
 dump(out/"controls.json",controls);o=Omni(cfg["model_path"]);clean=snapshot(o,cfg);baseline=measure(o,controls);dump(out/"clean-baseline.json",baseline)
 result=[]
 for src in map(Path,a.sources):
  source_rows=[r for p in src.glob("*.json") for r in [json.loads(p.read_text())] if r.get("eligible")]
  for r in sorted(source_rows,key=lambda r:r["unit"]["uid"]):
   uid=r["unit"]["uid"];folder=src/uid
   if not (folder/"damaged_state.pt").exists():
    folder=Path(r["source"])/uid
   state=torch.load(folder/"damaged_state.pt",map_location="cpu",weights_only=True);restore(o,state)
   damaged=measure(o,controls)
   record={"source":str(src),"uid":uid,"entity":r["unit"]["entity"],"damaged":compare(baseline,damaged,controls,r["unit"]["entity"]),"damaged_predictions":damaged,"repairs":{}}
   for branch in r.get("repairs",{}):
    path=src/uid/branch
    if not (path/"adapter_config.json").exists():continue
    restore(o,state);o.model=PeftModel.from_pretrained(o.model,str(path),is_trainable=False)
    post=measure(o,controls)
    record["repairs"][branch]={"compared_to_clean":compare(baseline,post,controls,r["unit"]["entity"]),"compared_to_damaged":compare(damaged,post,controls,r["unit"]["entity"]),"predictions":post}
    o.model=o.model.unload()
   restore(o,clean);dump(out/(src.name+"-"+uid+".json"),record);result.append(record);del state;gc.collect()
   print(src.name,r["unit"]["entity"],record["damaged"]["independent_locality_pass"],{b:x["compared_to_damaged"]["independent_locality_pass"] for b,x in record["repairs"].items()},flush=True)
 summary={}
 for src in a.sources:
  selected=[r for r in result if r["source"]==src]
  summary[Path(src).name]={"eligible_units":len(selected),"damage_independent_locality_pass":sum(r["damaged"]["independent_locality_pass"] for r in selected),"repair_independent_locality_pass":{b:sum(r["repairs"].get(b,{}).get("compared_to_damaged",{}).get("independent_locality_pass",False) for r in selected) for b in ["oracle","generative","structured"]}}
 dump(out/"metrics.json",{"status":"P0 additional independently held-out-from-KL audit; original metrics unchanged","summary":summary,"resources":o.state(),"limitations":["Toy general controls, not broad capability evaluation.","Neighbor entities are existing newdev only; no calibration/test queried.","Top-token agreement can preserve wrong baseline behavior; accuracy drops also reported."]})
if __name__=="__main__":main()
