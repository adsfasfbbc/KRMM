"""Public prewrite interface; contains no evaluator or dataset dependency."""
import numpy as np
from .task import question,OPTIONS
from .extractor import extract
from .extractor_canonical import replay_records
from .reconstruct import reconstruct
from .reliability import features
def infer(omni,unit,canonical=True):
 direct=[omni.probabilities(question(unit.entity,j))[0] for j in (2,3)]
 facts,records=extract(omni,unit.entity);normalizations=[]
 if canonical:facts,normalizations=replay_records(unit.entity,records)
 trace=reconstruct(facts)
 prompt="Infer the Hornbostel-Sachs family using only these observations:\n"+"\n".join(r["text"] for r in records)+"\n"+"\n".join(k+": "+v for k,v in OPTIONS.items())+"\nReply with one letter only."
 gen=omni.probabilities(prompt)[0]
 direct_candidate="ABCD"[int(np.argmax(direct[0]))]
 return {"candidate":trace["candidate"],"graph":trace,"features":features(facts),"probes":records,"normalizations":normalizations,"direct":direct,"generative":gen,"direct_candidate":direct_candidate,"generative_candidate":"ABCD"[int(np.argmax(gen))],"candidate_change":int(trace["candidate"] is not None and trace["candidate"]!=direct_candidate)}
