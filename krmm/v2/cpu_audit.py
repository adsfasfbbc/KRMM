"""Independent semantic, intervention and information-boundary tests."""
import argparse,unittest,json,ast,hashlib
from pathlib import Path
from .ontology import Fact,constraint
from .reconstruct import reconstruct
from .reliability import features
from .extractor import parse
from .evaluation import wrong_assignments
from .io import dump
def fact(attr,value,span="observed",source="p0",polarity=True):
 return Fact("anonymous",attr,value,"internal_text",source,"same-model",span,span,polarity)
class Audit(unittest.TestCase):
 def test_class_not_resonator(self):
  f=[fact("primary_oscillator","tensioned_strings"),fact("resonator","membrane_soundboard")]
  self.assertEqual(reconstruct(f)["candidate"],"A")
 def test_reed_not_solid(self):
  f=[fact("primary_oscillator","flexible_reed"),fact("excitation_mechanism","air_flow")]
  r=reconstruct(f);self.assertEqual(r["candidate"],"B");self.assertEqual(r["reconstruction_type"],"compositional")
  self.assertTrue(all(reconstruct([x])["candidate"] is None for x in f))
 def test_no_rule_no_fallback(self):self.assertIsNone(reconstruct([fact("material","wood")])["candidate"])
 def test_conflict(self):self.assertTrue(reconstruct([fact("primary_oscillator","air"),fact("primary_oscillator","solid_body")])["conflict"])
 def test_negation(self):self.assertIsNone(reconstruct([fact("primary_oscillator","air",polarity=False)])["candidate"])
 def test_key_sham_actual(self):
  f=[fact("primary_oscillator","tensioned_strings")]
  x=features(f);self.assertEqual(x[3:5],[1,1])
 def test_correlated_provenance(self):
  f=[fact("primary_oscillator","tensioned_strings")];self.assertLess(features(f+f)[5],features(f)[5])
 def test_bad_span(self):
  x=[dict(attribute="primary_oscillator",value="air",span="invented")]
  f,e=parse(json.dumps(x),"anon","p0","a wooden tube");self.assertFalse(f);self.assertTrue(e)
 def test_explicit_polarity(self):
  x=[dict(attribute="primary_oscillator",value="air",span="air",polarity="false")]
  self.assertFalse(parse(json.dumps(x),"anon","p0","air")[0])
 def test_wrong_mapping(self):
  r=[dict(uid=str(i),gold="ABCD"[i%4]) for i in range(3200)]
  a,t=wrong_assignments(r,9182)
  self.assertTrue(all(a[x["uid"]]!=x["gold"] for x in r));self.assertTrue(t["no_deterministic_inverse"])
  self.assertLess(t["best_in_sample_inverse_accuracy"],0.4)
 def test_old_immutable(self):
  old=json.loads(Path(".autoresearch-v02/v01-manifest.json").read_text())
  for p,h in old.items():
   if p not in {"README.md","EXPERIMENTS.md"}:self.assertEqual(hashlib.sha256(Path(p).read_bytes()).hexdigest(),h,p)
 def test_code_information_boundary(self):
  for name in ["ontology","extractor","evidence_graph","reconstruct","reliability","selector","repairer"]:
   src=Path(f"krmm/v2/{name}.py").read_text();tree=ast.parse(src)
   for n in ast.walk(tree):
    if isinstance(n,ast.ImportFrom):self.assertNotIn(n.module,{"evaluation","krmm.v2.evaluation","krmm.benchmark","krmm.v2.io"})
   self.assertNotIn("dataset.json",src);self.assertNotIn("gold",src)
 def test_dataset(self):
  d=json.loads(Path("configs/v2/dataset.json").read_text())["units"]
  self.assertEqual(len({r["group"] for r in d}),len(d))
  from krmm.benchmark import GROUPS
  from krmm.calibration_extension import EXTRA
  old={n.lower() for names in list(GROUPS.values())+list(EXTRA.values()) for n in names}
  self.assertTrue(all(r["entity"].lower() not in old for r in d))
def main():
 p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args()
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Audit)
 r=unittest.TextTestRunner(verbosity=2).run(suite)
 dump(a.out,{"tests":r.testsRun,"failures":len(r.failures),"errors":len(r.errors),"passed":r.wasSuccessful(),"scope":"CPU synthetic contracts only; no empirical candidate accuracy claim"})
 if not r.wasSuccessful():raise SystemExit(1)
if __name__=="__main__":main()
