"""Evaluator only; never import this module from reconstruction/selector/writer."""
import numpy as np,torch,hashlib,random,collections,math
from .task import question
def choose(p):return "ABCD"[int(np.argmax(p))]
def evaluate(omni,entity,gold,variants):
 ps=[omni.probabilities(question(entity,j))[0] for j in variants]
 return {"variants":list(variants),"probabilities":ps,"accuracy":float(np.mean([choose(p)==gold for p in ps]))}
def snapshot(omni,cfg):
 n=omni.model.config.text_config.num_hidden_layers
 return {k:p.detach().cpu().clone() for k,p in omni.model.named_parameters() if any(f"layers.{j}." in k for j in range(n-cfg["last_layers"],n)) and any(x in k for x in ["q_proj.weight","v_proj.weight","down_proj.weight"])}
def restore(omni,state):
 d=dict(omni.model.named_parameters())
 with torch.no_grad():
  for k,v in state.items():d[k].copy_(v)
 for p in omni.model.parameters():p.requires_grad_(False)
 omni.model.eval()
def agreement(a,b):return float(np.mean([int(x.argmax())==int(y.argmax()) for x,y in zip(a,b)]))
def state_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def wrong_assignments(rows,seed):
 # Private evaluator seed; each wrong class is chosen randomly among currently least-used allowed classes.
 # This is NOT the old cyclic gold+1 map. Excluding truth necessarily leaks "not this class" only.
 rng=random.Random(seed);counts=collections.Counter();assign={}
 ordered=list(rows);rng.shuffle(ordered)
 for r in ordered:
  allowed=[c for c in "ABCD" if c!=r["gold"]];low=min(counts[c] for c in allowed)
  c=rng.choice([c for c in allowed if counts[c]==low]);counts[c]+=1;assign[r["uid"]]=c
 table={w:collections.Counter(r["gold"] for r in rows if assign[r["uid"]]==w) for w in "ABCD"}
 inverse=sum(max(v.values(),default=0) for v in table.values())/max(1,len(rows))
 audit={"counts":dict(counts),"confusion_wrong_to_truth":{k:dict(v) for k,v in table.items()},"best_in_sample_inverse_accuracy":inverse,"cyclic_inverse_accuracy":float(np.mean(["ABCD"[("ABCD".index(assign[r["uid"]])-1)%4]==r["gold"] for r in rows])),"no_deterministic_inverse":all(len(v)>=2 for v in table.values()),"caveat":"Wrong-target exclusion necessarily reduces possible truth labels from4 to3; finite-sample inverse is optimistically fitted, not an unbiased prediction."}
 return assign,audit
