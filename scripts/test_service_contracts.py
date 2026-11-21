import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from krmm import service
from krmm.task import PublicUnit
class Model:
    def __init__(self):self.unloads=0
    def save_pretrained(self,p,**kw):Path(p).mkdir(parents=True)
    def unload(self):self.unloads+=1;return self
    def eval(self):return self
class Omni:
    def __init__(self,change=False):self.model=Model();self.n=0;self.change=change
    def retain_logits(self,p):
        self.n+=1;return [torch.tensor([0.,1.]) if self.change and self.n>1 else torch.tensor([1.,0.])]
    def state(self):return {}
class ServiceContracts(unittest.TestCase):
    def diag(self):return {'features':[0.]*8,'candidate':'A','direct':[[0.,1.,0.,0.]],'constraint_trace':{'unique_support':True,'conflict':False}}
    def test_missing_calibrator_abstains_without_training(self):
        with tempfile.TemporaryDirectory() as d,patch('krmm.service.diagnose',return_value=self.diag()),patch('krmm.service.writeability',return_value=0.),patch('krmm.service.train') as train:
            r=service.run(Omni(),PublicUnit('x','example','deployment'),['public retain prompt'],{'repair_steps':1,'retain_tolerance':.1},Path(d)/'out')
            self.assertEqual(r['decision'],'uncertain');self.assertIsNone(r['probability']);train.assert_not_called()
    def test_retention_violation_rolls_back_without_target_label(self):
        o=Omni(change=True)
        with tempfile.TemporaryDirectory() as d,patch('krmm.service.diagnose',return_value=self.diag()),patch('krmm.service.writeability',return_value=0.),patch('krmm.service.train',return_value=[]),patch('krmm.service.PeftModel.from_pretrained',return_value=o.model):
            r=service.run(o,PublicUnit('x','example','deployment'),['public retain prompt'],{'repair_steps':1,'retain_tolerance':.1},Path(d)/'out',policy='support')
            self.assertEqual(r['decision'],'rollback');self.assertIsNone(r['accepted_adapter']);self.assertEqual(o.model.unloads,2)
    def test_uncalibrated_structural_success_is_not_certified(self):
        o=Omni()
        with tempfile.TemporaryDirectory() as d,patch('krmm.service.diagnose',return_value=self.diag()),patch('krmm.service.writeability',return_value=0.),patch('krmm.service.train',return_value=[]),patch('krmm.service.PeftModel.from_pretrained',return_value=o.model):
            r=service.run(o,PublicUnit('x','example','deployment'),['public retain prompt'],{'repair_steps':1,'retain_tolerance':.1},Path(d)/'out',policy='support')
            self.assertEqual(r['decision'],'repaired_unverified_target');self.assertIsNone(r['probability']);self.assertIsNotNone(r['accepted_adapter'])
if __name__=='__main__':unittest.main()
