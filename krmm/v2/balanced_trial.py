"""Prospective eligibility-conditional balanced wrong-target control, no repair-outcome selection."""
import argparse,json,secrets,random,collections,sys
from pathlib import Path
from .io import rows,dump
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--config",default="configs/v2/association-24.json");ap.add_argument("--screen-source",default="results/v2-new/survey-dev-02");a=ap.parse_args()
 allrows=rows("dev")
 screen={r["uid"]:r for p in Path(a.screen_source).glob("*.json") for r in [json.loads(p.read_text())] if "screen" in r}
 qualified=[r for r in allrows if screen[r["uid"]]["screen"]["accuracy"]==1]
 # No independent-clean answers, candidate predictions or repair outcomes used.
 seed=secrets.randbits(64);rng=random.Random(seed)
 target=None
 for attempt in range(1_000_000):
  trial={r["uid"]:rng.choice([c for c in "ABCD" if c!=r["gold"]]) for r in qualified}
  counts=collections.Counter(trial.values())
  table={w:collections.Counter(r["gold"] for r in qualified if trial[r["uid"]]==w) for w in "ABCD"}
  if max(counts.get(c,0) for c in "ABCD")-min(counts.get(c,0) for c in "ABCD")<=1 and all(len(v)>=2 for v in table.values()):
   target=trial;break
 if target is None:raise RuntimeError("No admissible balanced schedule within frozen rejection cap")
 for r in allrows:
  if r["uid"] not in target:target[r["uid"]]=rng.choice([c for c in "ABCD" if c!=r["gold"]])
 schedule=Path(a.out+"-schedule");schedule.mkdir(parents=True,exist_ok=False)
 audit={"design":"Uniform independent wrong targets conditioned on balanced class counts and >=2 truth classes per wrong class; evaluator-only rejection sampling. No candidate/repair outcomes used.","qualification":"clean screen0/1 only","qualified_n":len(qualified),"counts":dict(counts),"confusion_wrong_to_truth":{k:dict(v) for k,v in table.items()},"no_deterministic_inverse":all(len(v)>=2 for v in table.values()),"best_in_sample_inverse_accuracy":sum(max(v.values()) for v in table.values())/len(qualified),"fixed_cyclic_rule":False,"sampling_attempts":attempt+1,"limitations":"Balance is pre-damage among qualified attempts; effectiveness/locality attrition must still be audited separately."}
 dump(schedule/"damage-randomization-audit.json",audit)
 private=Path("/root/rivermind-data/setup-v02/private-assignments")/(schedule.name+".json")
 private.write_text(json.dumps({"seed":seed,"targets":target}));private.chmod(0o600)
 print("BALANCED_SCHEDULE",json.dumps(audit),flush=True)
 from .sweep import main as sweep
 sys.argv=[sys.argv[0],"--config",a.config,"--out",a.out,"--assignment-source",str(schedule)]
 sweep()
if __name__=="__main__":main()
