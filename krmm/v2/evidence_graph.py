from .ontology import FAMILIES,constraint
def graph(facts):
 paths=[];compatible=set(FAMILIES);errors=[]
 for i,f in enumerate(facts):
  try:allowed=constraint(f)
  except ValueError as e:errors.append({"index":i,"error":str(e)});continue
  compatible.intersection_update(allowed)
  if allowed!=FAMILIES:paths.append({"index":i,"attribute":f.attribute,"value":f.value,"source_id":f.source_id,"allowed":sorted(allowed),"span":f.span})
 conflict=bool(paths) and not compatible
 return {"compatible":sorted(compatible),"paths":paths,"conflict":conflict,"invalid_atoms":errors,"ambiguous":len(compatible)!=1}
