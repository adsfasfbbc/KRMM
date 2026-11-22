import json,hashlib,subprocess,time
from pathlib import Path
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(f"Immutable result exists: {p}")
 tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)
def environment():
 import torch
 return {"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"git_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),"source_hashes":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(Path("krmm/v2").glob("*.py"))+list(Path("configs/v2").glob("*.json"))}}
def rows(split):
 d=json.loads(Path("configs/v2/dataset.json").read_text())["units"]
 out=[]
 for k in "ABCD":
  out.append(sorted([r for r in d if r["split"]==split and r["gold"]==k],key=lambda r:r["uid"]))
 return [r for i in range(max(map(len,out))) for a in out if i<len(a) for r in [a[i]]]
