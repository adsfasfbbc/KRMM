"""Exploratory paired evidence conditions; one fixed writer, no target labels in evidence."""
import argparse,json,random,hashlib,time
from pathlib import Path
import numpy as np,torch
from .model import Omni
from .benchmark import public_units
from .experiment import one,dump

def weight_hash(path):
    state=torch.load(path,map_location="cpu",weights_only=True);h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.numpy().tobytes())
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--config",default="configs/pilot-constraints.json");args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(8)
    media_rows=json.loads(Path("data/media/manifest.json").read_text());omni=Omni(cfg["model_path"]);results=[];start=time.monotonic()
    for i,m in enumerate(media_rows):
        u=next(u for u in public_units() if u.uid==m["uid"])
        related=m.get("media",{});donor=media_rows[(i+1)%len(media_rows)].get("media",{})
        shuffled=dict(related)
        if donor.get("image"):shuffled["image"]=donor["image"]
        related_atoms=None
        for condition,evidence in [("internal",None),("related_media",related),("shuffled_image",shuffled),("matched_text",None)]:
            seed=cfg["seed"]+i*1009;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            local=dict(cfg);local["evidence_by_uid"]={u.uid:evidence} if evidence else {}
            if condition=="matched_text":
                if related_atoms is None:
                    results.append({"unit":u.__dict__,"condition":condition,"skipped":"No related-media atomic observations available"});continue
                local["evidence_text_by_uid"]={u.uid:"\nAdditional observations extracted earlier from the same damaged model and related raw media: "+" ".join(related_atoms)}
            case_root=out/condition;case_root.mkdir(exist_ok=True)
            r=one(omni,u,local,case_root)
            r["condition"]=condition;r["media_modalities"]=sorted(evidence) if evidence else []
            if r.get("eligible"):
                r["damaged_weight_sha256"]=weight_hash(case_root/u.uid/"damaged_state.pt")
                if condition=="related_media":related_atoms=r["diagnosis"]["atomic"]
            dump(case_root/(u.uid+".json"),r);results.append(r)
            print(u.entity,condition,r.get("eligible"),r.get("constrained",{}).get("success"),flush=True)
    summary={"conditions":{},"seconds":time.monotonic()-start,"n_distinct_entities":len(media_rows)}
    for c in ["internal","related_media","shuffled_image","matched_text"]:
        rr=[r for r in results if r["condition"]==c];eligible=[r for r in rr if r.get("eligible")]
        summary["conditions"][c]={"attempted":len(rr),"eligible":len(eligible),"candidate_accuracy":float(np.mean([r["candidate_correct_evaluator_only"] for r in eligible])) if eligible else None,"persistent_success_over_all_eligible":float(np.mean([r["constrained"]["success"] for r in eligible])) if eligible else None}
    hashes={}
    for r in results:
        if r.get("eligible"):hashes.setdefault(r["unit"]["uid"],set()).add(r["damaged_weight_sha256"])
    summary["paired_damage_identical"]=all(len(x)==1 for x in hashes.values())
    summary["claim_boundary"]="Small development-only evidence study. Four images, one audio source; shuffled-image control keeps original audio. No sound-specific or population complementarity claim. Matched text repeats damaged-model media-conditioned atomic observations, no evaluator truth. One round of writeback per independent damaged copy."
    dump(out/"summary.json",summary);print(json.dumps(summary),flush=True)
if __name__=="__main__":main()
