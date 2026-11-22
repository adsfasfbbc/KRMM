from .evidence_graph import graph
def reconstruct(facts):
 g=graph(facts);candidate=g["compatible"][0] if len(g["compatible"])==1 and not g["invalid_atoms"] else None
 # Multiple facts only count as compositional if every singleton remains ambiguous.
 singletons=[graph([f]) for f in facts]
 composition=bool(candidate) and all(len(x["compatible"])>1 for x in singletons)
 g.update(candidate=candidate,reconstruction_type=("auxiliary_evidence" if any(f.modality in {"vision","audio","matched_text"} for f in facts) else "compositional" if composition else "direct_residual_reaccess") if candidate else "insufficient_or_conflicting")
 return g
