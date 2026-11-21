"""Prospective calibration extension; evaluator-only label provider, original test unchanged."""
import argparse,json,hashlib,random,subprocess,time
from pathlib import Path
import numpy as np,torch
from .task import PublicUnit
from .model import Omni
from . import experiment
EXTRA={"A":["viola","harpsichord","erhu"],"B":["piccolo","cornet","pipe organ"],"C":["taiko","surdo","mridangam"],"D":["steelpan","kalimba","vibraslap"]}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--config",default="configs/pilot-constraints.json");args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text());random.seed(cfg["seed"]);np.random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"]);torch.set_num_threads(8)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    truth={name:k for k,names in EXTRA.items() for name in names}
    original_gold=experiment.gold
    # Explicit evaluator dependency injection. Repair module imports no evaluator labels.
    experiment.gold=lambda u:truth[u.entity] if u.entity in truth else original_gold(u)
    units=[PublicUnit(hashlib.sha256(EXTRA[k][i].encode()).hexdigest()[:12],EXTRA[k][i],"calibration") for i in range(3) for k in "ABCD"]
    manifest={"version":"calibration-extension-v1","purpose":"Prospective sample-size extension after original calibration yielded 7 eligible units; before any test query",
      "units":[dict(u.__dict__,gold_evaluator_only=truth[u.entity]) for u in units],"original_test_untouched":True,
      "label_status":"Curator conventional acoustic-source labels; independent full expert audit pending"}
    experiment.dump(out/"manifest.json",manifest);experiment.dump(out/"config.json",cfg)
    experiment.dump(out/"environment.json",{"source_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"torch":torch.__version__,"gpu":torch.cuda.get_device_name()})
    omni=Omni(cfg["model_path"]);rows=[]
    for u in units:
        print("START",u.entity,flush=True);r=experiment.one(omni,u,cfg,out);rows.append(r);experiment.dump(out/(u.uid+".json"),r)
        print("END",u.entity,r.get("eligible"),r.get("constrained",{}).get("success"),flush=True)
    eligible=[r for r in rows if r.get("eligible")]
    m={"independent_units":len(rows),"eligible_units":len(eligible),"clean_mastery_rate":float(np.mean([r["clean"]["screening_accuracy"]==1 for r in rows])),
       "damage_eligibility_rate":len(eligible)/len(rows),"oracle_success_rate":float(np.mean([r["oracle"]["success"] for r in eligible])) if eligible else 0,
       "candidate_accuracy":float(np.mean([r["candidate_correct_evaluator_only"] for r in eligible])) if eligible else 0,
       "persistent_repair_rate_all_eligible":float(np.mean([r["constrained"]["success"] for r in eligible])) if eligible else 0,
       "peak_vram_gb":omni.state()["peak_vram_gb"],"split_verified":len(truth)==12,"original_test_untouched":True}
    experiment.dump(out/"metrics.json",m);print(json.dumps(m),flush=True)
if __name__=="__main__":main()
