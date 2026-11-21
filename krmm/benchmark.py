"""Evaluator-owned labels. Repair code receives PublicUnit only."""
from dataclasses import dataclass
import hashlib,json
from pathlib import Path

OPTIONS = {"A":"vibrating strings","B":"a vibrating air column","C":"a stretched vibrating membrane","D":"the vibrating solid body itself"}
GROUPS = {
 "A":["violin","cello","guitar","harp","piano","banjo","mandolin","ukulele","sitar","double bass","zither","dulcimer"],
 "B":["flute","clarinet","trumpet","trombone","oboe","bassoon","saxophone","tuba","recorder","French horn","panpipes","ocarina"],
 "C":["snare drum","timpani","bongo","conga","tabla","djembe","bass drum","frame drum","tom-tom","bodhran","darbuka","talking drum"],
 "D":["xylophone","marimba","glockenspiel","triangle","cymbal","gong","bell","castanets","wood block","claves","vibraphone","celesta"]
}
@dataclass(frozen=True)
class PublicUnit:
    uid: str
    entity: str
    split: str
    relation: str = "primary vibrating sound source"

def public_units():
    out=[]
    for label,names in GROUPS.items():
        for i,name in enumerate(names):
            split="dev" if i<3 else "calibration" if i<6 else "test"
            out.append(PublicUnit(hashlib.sha256(name.encode()).hexdigest()[:12],name,split))
    return out

def gold(unit):
    return next(k for k,v in GROUPS.items() if unit.entity in v)

def question(entity,variant=0):
    stems=[
      f"What primarily vibrates to produce the sound of a {entity}?",
      f"For the {entity}, identify its principal sound-producing vibrating element.",
      f"Which physical source of vibration is fundamental to the {entity}'s sound?",
      f"Classify the {entity} by what primarily vibrates when it sounds."
    ]
    return stems[variant]+"\n"+"\n".join(k+": "+v for k,v in OPTIONS.items())+"\nReply with one letter only."

def retention_questions(target,n=8):
    # Never freeze logical consequences of target; distinct entities plus unrelated abilities.
    other=[u for u in public_units() if u.entity != target.entity and u.split=="dev"]
    selected=sorted(other,key=lambda u:hashlib.sha256((target.uid+u.uid).encode()).hexdigest())[:n]
    return [question(u.entity,1) for u in selected]+[
       "What is 7 plus 8? Answer with just the number.",
       "What is the capital of France? Answer with just the city name.",
       "Which planet is known as the Red Planet? Answer with just the name.",
       "What is the chemical symbol for oxygen? Answer with just the symbol."
    ]

def manifest():
    rows=[dict(uid=u.uid,entity=u.entity,split=u.split,gold=gold(u)) for u in public_units()]
    return {"version":"instruments-v1","source":"Curated standard acoustic sound-source taxonomy; not organological subclass labels. Preliminary labels require independent audit.",
      "source_urls":["https://en.wikipedia.org/wiki/Hornbostel%E2%80%93Sachs","https://www.britannica.com/art/musical-instrument"],
      "limitation":"Piano and celesta classified by primary vibrating source, not playing action. Closed-choice task does not establish open-ended QA.",
      "units":rows,"sha256":hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}
