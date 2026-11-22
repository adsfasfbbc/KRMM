from dataclasses import dataclass
OPTIONS={"A":"chordophone (string instrument)","B":"aerophone (wind instrument, including free reeds)","C":"membranophone (tensioned membrane instrument)","D":"idiophone (solid body instrument)"}
@dataclass(frozen=True)
class PublicUnit:
 uid:str
 entity:str
 split:str
def question(entity,variant=0):
 stems=[f"Classify the acoustic instrument {entity} in the Hornbostel-Sachs family system.",
 f"Which Hornbostel-Sachs family contains the traditional acoustic {entity}?",
 f"For {entity}, choose its organological family, considering the sound-producing mechanism.",
 f"Assign the acoustic {entity} to one of the four Hornbostel-Sachs families.",
 f"A museum catalogs a traditional {entity}. Which acoustic Hornbostel-Sachs family is appropriate?",
 f"What is the top-level organological class of the traditional {entity} under Hornbostel-Sachs?"]
 return stems[variant]+"\n"+"\n".join(k+": "+v for k,v in OPTIONS.items())+"\nReply with one letter only."
def retain_prompts():
 return [question(n,2) for n in ["violin","guitar","flute","trumpet","snare drum","timpani","xylophone","triangle"]]+["What is 7 plus 8? Answer with just the number.","What is the capital of France? Answer with just the city name.","Which planet is known as the Red Planet? Answer with just the name.","What is the chemical symbol for oxygen? Answer with just the symbol."]
