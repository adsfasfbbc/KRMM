"""Independent physical counterexamples; frozen before correcting ontology v1."""
import unittest,argparse
from .ontology import Fact
from .reconstruct import reconstruct
from .io import dump
def f(attr,value,conflict=False):
 return Fact("anonymous",attr,value,"internal_text","p","one-source","explicit observation","explicit observation",True,conflict)
class PhysicsRegression(unittest.TestCase):
 def test_air_excited_strings_still_chordophone(self):
  self.assertEqual(reconstruct([f("primary_oscillator","tensioned_strings"),f("excitation_mechanism","air_flow")])["candidate"],"A")
 def test_plucked_membrane_still_membranophone(self):
  self.assertEqual(reconstruct([f("primary_oscillator","stretched_membrane"),f("excitation_mechanism","plucking")])["candidate"],"C")
 def test_bowed_membrane_not_excluded(self):
  self.assertEqual(reconstruct([f("primary_oscillator","stretched_membrane"),f("excitation_mechanism","bowing")])["candidate"],"C")
 def test_generic_airflow_plus_reed_remains_ambiguous(self):
  self.assertIsNone(reconstruct([f("primary_oscillator","flexible_reed"),f("excitation_mechanism","air_flow")])["candidate"])
 def test_explicit_direct_air_driven_reed(self):
  fs=[f("primary_oscillator","flexible_reed"),f("excitation_mechanism","direct_air_flow")]
  r=reconstruct(fs);self.assertEqual(r["candidate"],"B");self.assertEqual(r["reconstruction_type"],"compositional")
  self.assertTrue(all(reconstruct([x])["candidate"] is None for x in fs))
 def test_explicit_conflict_blocks_update(self):
  r=reconstruct([f("primary_oscillator","tensioned_strings"),f("material","wood",conflict=True)])
  self.assertTrue(r["conflict"]);self.assertIsNone(r["candidate"])
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);a=ap.parse_args()
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PhysicsRegression))
 dump(a.out,{"tests":r.testsRun,"failures":len(r.failures),"errors":len(r.errors),"passed":r.wasSuccessful(),"purpose":"Counterexample regression, unchanged between RED and GREEN runs. No benchmark labels."})
 if not r.wasSuccessful():raise SystemExit(1)
if __name__=="__main__":main()
