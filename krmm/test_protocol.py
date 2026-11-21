import unittest
from .task import PublicUnit,question
from .repair import diagnose
class FakeOmni:
    def __init__(self):self.calls=0;self.prompts=[]
    def probabilities(self,prompt,media=None):self.calls+=1;self.prompts.append(prompt);return [.25]*4,1.
    def generate(self,prompt,max_new_tokens,media=None):self.calls+=1;self.prompts.append(prompt);return 'The strings vibrate and sound travels through air.'
class ProtocolContracts(unittest.TestCase):
    def test_hidden_questions_never_used_for_diagnosis(self):
        omni=FakeOmni();u=PublicUnit('test','example','dev');d=diagnose(omni,u,{'max_new_tokens':48,'probe_budget':24,'candidate_policy':'symbolic_atoms'},[])
        for j in [2,3]:self.assertNotIn(question('example',j),omni.prompts)
        for j in [0,1,6]:self.assertIn(question('example',j),omni.prompts)
        self.assertEqual(d['candidate'],'A');self.assertEqual(d['features'][5],1.)
    def test_prompt_partitions_disjoint(self):
        for a,b in [({4,5},{2,3}),({0,1,6},{2,3}),({0,1},{2,3})]:self.assertFalse(a&b)
if __name__=='__main__':unittest.main()
