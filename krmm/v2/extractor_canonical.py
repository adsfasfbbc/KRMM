"""Typed vocabulary normalization of SAME model JSON; never scan raw prose for labels."""
import json,re
from .extractor import parse
ALIASES={
"primary_oscillator":{"string":"tensioned_strings","strings":"tensioned_strings","vibrating_strings":"tensioned_strings","membrane":"stretched_membrane","drumhead":"stretched_membrane","air_column":"air","vibrating_air":"air","solid":"solid_body","reed":"flexible_reed"},
"excitation_mechanism":{"airflow":"air_flow","direct_airflow":"direct_air_flow","blowing":"air_flow","blown":"air_flow","bowed":"bowing","plucked":"plucking","struck":"striking"},
"resonator":{"wooden_resonating_body":"wooden_body","wooden_body":"wooden_body","air_column":"air_cavity","hollow_body":"air_cavity"},
"material":{}}
def canonical_parse(raw,entity,source_id,text):
 m=re.search(r"\[.*\]",raw,re.S)
 if not m:return [],["no_json_array"],[]
 try:rows=json.loads(m.group())
 except (ValueError,TypeError):return [],["malformed_json"],[]
 changes=[]
 for row in rows:
  if not isinstance(row,dict) or not isinstance(row.get("attribute"),str):continue
  value=row.get("value")
  if not isinstance(value,str):continue
  key=value.strip().casefold().replace("-","_").replace(" ","_")
  normalized=ALIASES.get(row.get("attribute"),{}).get(key,key)
  if normalized!=value:
   changes.append({"attribute":row.get("attribute"),"raw_value":value,"canonical_value":normalized});row["value"]=normalized
 facts,errors=parse(json.dumps(rows),entity,source_id,text)
 return facts,errors,changes
def replay_records(entity,records):
 facts=[];traces=[]
 for i,r in enumerate(records):
  fs,errors,changes=canonical_parse(r["extraction_raw"],entity,f"probe_{i}",r["text"])
  facts.extend(fs);traces.append({"source_id":f"probe_{i}","normalizations":changes,"errors":errors,"facts":[f.serial() for f in fs]})
 return facts,traces
