"""Model semantic extraction with exact quoted provenance; invalid output abstains."""
import json,re
from .ontology import Fact,VALUES
def parse(raw,entity,source_id,text):
 match=re.search(r"\[.*\]",raw,re.S)
 if not match:return [],["no_json_array"]
 try:rows=json.loads(match.group())
 except (ValueError,TypeError):return [],["malformed_json"]
 facts=[];errors=[]
 for row in rows:
  try:
   # Explicit boolean polarity prevents truthiness of strings.
   if type(row.get("polarity",True)) is not bool:raise ValueError("polarity must be bool")
   f=Fact(entity,row["attribute"],row["value"],"internal_text",source_id,"damaged-model",row["span"],text,row.get("polarity",True),row.get("conflict",False))
   f.validate();facts.append(f)
  except (KeyError,ValueError,TypeError) as e:errors.append(str(e))
 return facts,errors
def extract(omni,entity):
 probes=[f"What physical component generates vibration in the traditional {entity}? Describe the primary oscillator and distinguish it from the resonating body. Do not name an instrument family.",
 f"How is the traditional {entity} excited, and what resonates? Describe physical facts only, without a family label."]
 records=[];allfacts=[]
 schema=json.dumps({k:sorted(v) for k,v in VALUES.items()})
 for i,prompt in enumerate(probes):
  text=omni.generate(prompt,72)
  query="Extract only explicitly stated physical facts from the quoted observation. Do not add outside knowledge. Use these attribute/value enums: "+schema+". Return JSON array only, each object: attribute, value, span (exact quote proving this fact), polarity (boolean, false for negation). Distinguish primary oscillator from resonator; unknown is allowed. Observation: "+json.dumps(text)
  raw=omni.generate(query,224);facts,errors=parse(raw,entity,f"probe_{i}",text)
  allfacts.extend(facts);records.append({"prompt":prompt,"text":text,"extraction_prompt":query,"extraction_raw":raw,"errors":errors,"facts":[f.serial() for f in facts]})
 return allfacts,records
