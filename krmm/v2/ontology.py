"""Typed organological facts. Actions alone never determine instrument family."""
from dataclasses import dataclass,asdict
FAMILIES=frozenset("ABCD")
VALUES={"primary_oscillator":{"tensioned_strings","stretched_membrane","solid_body","air","flexible_reed","unknown"},
"excitation_mechanism":{"air_flow","direct_air_flow","bowing","plucking","striking","friction","unknown"},
"resonator":{"air_cavity","wooden_body","membrane_soundboard","none","unknown"},
"material":{"wood","metal","skin","bamboo","unknown"}}
@dataclass(frozen=True)
class Fact:
 entity:str
 attribute:str
 value:str
 modality:str
 source_id:str
 provenance_group:str
 span:str
 raw_text:str
 polarity:bool=True
 conflict:bool=False
 def serial(self):return asdict(self)
 def validate(self):
  if self.attribute not in VALUES or self.value not in VALUES[self.attribute]:raise ValueError("unsupported semantic atom")
  if not self.span or self.span.casefold() not in self.raw_text.casefold():raise ValueError("ungrounded source span")
  if self.modality not in {"internal_text","vision","audio","matched_text","control"}:raise ValueError("unknown modality")
  if not self.source_id or not self.provenance_group:raise ValueError("missing provenance")
def constraint(f):
 f.validate()
 if not f.polarity or f.conflict or f.value=="unknown":return FAMILIES
 if f.attribute=="primary_oscillator":
  return {"tensioned_strings":frozenset("A"),"stretched_membrane":frozenset("C"),
   "solid_body":frozenset("D"),"air":frozenset("B"),"flexible_reed":frozenset("BD")}[f.value]
 # Air can excite Aeolian strings; membranes can be plucked or bowed.
 # Material, resonance and action alone are insufficient for an organological family.
 return FAMILIES
