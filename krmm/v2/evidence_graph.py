from .ontology import FAMILIES,constraint
def graph(facts):
 paths=[];compatible=set(FAMILIES);errors=[];valid=[]
 for i,f in enumerate(facts):
  try:allowed=constraint(f)
  except (ValueError,TypeError) as e:errors.append({"index":i,"error":str(e)});continue
  if f.polarity and not f.conflict:valid.append((i,f))
  compatible.intersection_update(allowed)
  if allowed!=FAMILIES:paths.append({"index":i,"attribute":f.attribute,"value":f.value,"source_id":f.source_id,"allowed":sorted(allowed),"span":f.span})
 # Joint CLASSIFICATION rule, not a raw keyword match: an explicitly air-driven
 # reed is an aerophone; a reed alone may also be a plucked idiophone.
 reeds=[(i,f) for i,f in valid if f.attribute=="primary_oscillator" and f.value=="flexible_reed"]
 flows=[(i,f) for i,f in valid if f.attribute=="excitation_mechanism" and f.value=="direct_air_flow"]
 for i,reed in reeds:
  for j,flow in flows:
   if reed.entity!=flow.entity:continue
   compatible.intersection_update("B")
   paths.append({"indices":[i,j],"rule":"HS_air_driven_reed","allowed":["B"],"source_ids":[reed.source_id,flow.source_id],"spans":[reed.span,flow.span],"assumption":"direct aerodynamic excitation explicitly asserted, not inferred from generic airflow or body material"})
 explicit_conflicts=[i for i,f in enumerate(facts) if f.conflict]
 conflict=bool(explicit_conflicts) or (bool(paths) and not compatible)
 return {"compatible":sorted(compatible),"paths":paths,"conflict":conflict,"invalid_atoms":errors,"explicit_conflicts":explicit_conflicts,"ambiguous":len(compatible)!=1}
