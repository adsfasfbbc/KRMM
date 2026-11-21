import unittest
from .constraints import extract,reconstruct
class ConstraintContracts(unittest.TestCase):
    def test_air_transmission_is_not_primary_air_vibration(self):
        self.assertEqual(extract("The strings vibrate and sound waves travel through the air."),{"A"})
    def test_membrane(self):self.assertEqual(extract("A stretched membrane or drumhead is set vibrating."),{"C"})
    def test_no_entity_shortcut(self):self.assertEqual(extract("violin flute drum xylophone"),set())
    def test_negation(self):self.assertEqual(extract("Not strings; an air column vibrates."),{"B"})
    def test_conflict_preserved(self):self.assertTrue(reconstruct(["strings vibrate","metal bars vibrate"],[.25]*4)["conflict"])
    def test_missing_support_falls_back(self):self.assertEqual(reconstruct(["No observations"],[.1,.6,.2,.1])["source"],"generative_fallback")
if __name__=="__main__":unittest.main()
