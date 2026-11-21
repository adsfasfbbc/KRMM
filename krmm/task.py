"""Public schema and task prompts. Contains no benchmark truths or entity lists."""
from dataclasses import dataclass
OPTIONS = {"A":"vibrating strings","B":"a vibrating air column","C":"a stretched vibrating membrane","D":"the vibrating solid body itself"}
@dataclass(frozen=True)
class PublicUnit:
    uid: str
    entity: str
    split: str
    relation: str = "primary vibrating sound source"

def question(entity,variant=0):
    stems=[
      f"What primarily vibrates to produce the sound of a {entity}?",
      f"For the {entity}, identify its principal sound-producing vibrating element.",
      f"Which physical source of vibration is fundamental to the {entity}'s sound?",
      f"Classify the {entity} by what primarily vibrates when it sounds.",
      f"A {entity} produces musical tones. Which listed element is its principal acoustical vibrator?",
      f"Which option identifies the dominant sound-generating vibration in a {entity}?",
      f"Assign the {entity} to the listed category of primary sound source."
    ]
    return stems[variant]+"\n"+"\n".join(k+": "+v for k,v in OPTIONS.items())+"\nReply with one letter only."

