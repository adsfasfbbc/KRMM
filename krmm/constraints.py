"""Public physical-vibration constraints; no entities, labels or benchmark truth imported."""
import re
import numpy as np
RULES={
 "A":[r"\bstrings?\b"],
 "B":[r"\bair column\b",r"\bcolumn of air\b",r"\bair (?:inside|within)\b",r"\bair (?:is |gets |being )?(?:set into |made to )?(?:oscillat|vibrat)"],
 "C":[r"\bmembranes?\b",r"\bdrum\s*heads?\b",r"\bstretched (?:skin|head)\b"],
 "D":[r"\bsolid (?:body|material)\b",r"\b(?:wooden|metal|metallic) (?:bars?|plates?|blocks?|tines?)\b"]
}
def extract(fact):
    text=fact.lower();labels=set()
    for label,patterns in RULES.items():
        for pattern in patterns:
            for m in re.finditer(pattern,text):
                prefix=text[max(0,m.start()-35):m.start()]
                clause=re.split(r"[.;]|\bbut\b",prefix)[-1]
                if re.search(r"\b(not|never|no)\b",clause):continue
                labels.add(label)
    return labels

def reconstruct(facts,fallback):
    matches=[extract(f) for f in facts]
    scores=np.zeros(4)
    for labels in matches:
        for label in labels:scores["ABCD".index(label)]+=1/max(1,len(labels))
    total=float(scores.sum())
    if total==0:
        return {"candidate":"ABCD"[int(np.argmax(fallback))],"probabilities":list(fallback),"source":"generative_fallback","matches":[sorted(s) for s in matches],"unique_support":False,"conflict":False}
    p=(scores+.05)/(scores.sum()+.2)
    union=set().union(*matches)
    return {"candidate":"ABCD"[int(np.argmax(p))],"probabilities":p.tolist(),"source":"generic_physical_constraints","matches":[sorted(s) for s in matches],"unique_support":len(union)==1,"conflict":len(union)>1}
