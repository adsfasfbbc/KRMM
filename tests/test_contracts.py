import unittest
from krmm.benchmark import public_units,gold,question,retention_questions,manifest
class Contracts(unittest.TestCase):
    def test_splits_are_entity_disjoint(self):
        u=public_units();self.assertEqual(len(u),48);self.assertEqual(len({x.entity for x in u}),48)
        self.assertEqual({s:sum(x.split==s for x in u) for s in ["dev","calibration","test"]},{"dev":12,"calibration":12,"test":24})
    def test_public_view_has_no_truth(self):
        for u in public_units():
            self.assertNotIn("gold",u.__dict__);self.assertNotIn("answer",u.__dict__)
    def test_targets_not_retained(self):
        for u in public_units():
            for q in retention_questions(u):self.assertNotIn("of a "+u.entity+"?",q)
    def test_paraphrases_differ(self):
        for u in public_units():self.assertEqual(len({question(u.entity,i) for i in range(4)}),4)
    def test_manifest_deterministic(self):self.assertEqual(manifest()["sha256"],manifest()["sha256"])
if __name__=="__main__":unittest.main()
