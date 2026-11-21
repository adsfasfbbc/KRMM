"""Media I/O smoke only; clean model, not evidence of knowledge repair."""
import json,argparse,time
from pathlib import Path
import numpy as np
from .model import Omni
from .repair import choose
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--config",default="configs/pilot.json");args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text());omni=Omni(cfg["model_path"]);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rows=json.loads(Path("data/media/manifest.json").read_text());res=[]
    choices="Identify the instrument shown or heard. A: violin. B: flute. C: snare drum. D: xylophone. Reply with one letter only."
    mapping={"violin":"A","flute":"B","snare drum":"C","xylophone":"D"}
    for r in rows:
        for modality in ["image","audio","joint"]:
            m=r["media"]
            if modality=="joint":
                if not ("image" in m and "audio" in m):continue
                evidence=m
            elif modality in m:evidence={modality:m[modality]}
            else:continue
            t=time.monotonic()
            p,mass=omni.probabilities(choices,media=evidence)
            res.append({"uid":r["uid"],"entity_evaluator_only":r["entity"],"modality":modality,"probabilities":p,"choice_mass":mass,"prediction":choose(p),"gold_evaluator_only":mapping[r["entity"]],"correct":choose(p)==mapping[r["entity"]],"seconds":time.monotonic()-t})
            print(res[-1],flush=True)
    report={"status":"IO_SMOKE_ONLY","rows":res,"resources":omni.state(),"limitation":"Clean model; only one real audio source, no source-generalization or complementarity conclusion; not repair."}
    (out/"summary.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=="__main__":main()
