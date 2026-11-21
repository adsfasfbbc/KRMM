"""Learn opposing random fact carriers, erase them, verify exact indistinguishability."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import torch
from torch import nn
class Base(nn.Module):
    def __init__(self):
        super().__init__();self.image=nn.Linear(256,2);self.audio=nn.Linear(65,2)
class Twin(nn.Module):
    def __init__(self,base,n):
        super().__init__();self.base=copy.deepcopy(base);self.carrier=nn.Parameter(torch.zeros(n,2,device="cuda"))
        for p in self.base.parameters():p.requires_grad_(False)
    def target(self,ids):
        if self.carrier is None:return torch.zeros(len(ids),2,device=ids.device)
        return self.carrier[ids]
def digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def repair_view(model,ids,image,audio):
    # Deliberately accepts only the damaged weights and identical unlabeled evidence.
    return model.target(ids).argmax(-1)
def one(seed,n=256):
    torch.manual_seed(seed);ids=torch.arange(n,device="cuda")
    v=torch.randint(2,(n,),device="cuda");a=torch.randint(2,(n,),device="cuda");truth=torch.randint(2,(n,),device="cuda")
    image=torch.randn(n,256,device="cuda")*.05;image[:,0]=v.float()*2-1
    t=torch.arange(128,device="cuda")/128
    waveform=torch.sin(2*torch.pi*(8+16*a[:,None])*t)
    audio=torch.fft.rfft(waveform).abs()/64
    base=Base().cuda();opt=torch.optim.Adam(base.parameters(),lr=.05)
    for _ in range(200):
        opt.zero_grad();loss=nn.functional.cross_entropy(base.image(image),v)+nn.functional.cross_entropy(base.audio(audio),a);loss.backward();opt.step()
    wa,wb=Twin(base,n).cuda(),Twin(base,n).cuda()
    for model,y in [(wa,truth),(wb,1-truth)]:
        opt=torch.optim.Adam([model.carrier],lr=.1)
        for _ in range(100):
            opt.zero_grad();loss=nn.functional.cross_entropy(model.target(ids),y);loss.backward();opt.step()
    with torch.no_grad():
        ca=float((wa.target(ids).argmax(-1)==truth).float().mean());cb=float((wb.target(ids).argmax(-1)==1-truth).float().mean())
        assert ca==cb==1.
        original_different=digest(wa)!=digest(wb)
        # Completely remove all parameters learned from random target mapping.
        wa.carrier=None;wb.carrier=None
        same=digest(wa)==digest(wb);assert same
        pa=repair_view(wa,ids,image,audio);pb=repair_view(wb,ids,image,audio);assert torch.equal(pa,pb)
        pairboth=(pa==truth)&(pb==1-truth);assert not pairboth.any()
        acca=float((pa==truth).float().mean());accb=float((pb==1-truth).float().mean())
        assert abs((acca+accb)/2-.5)<1e-8
        retained_image=float((wa.base.image(image).argmax(-1)==v).float().mean());retained_audio=float((wa.base.audio(audio).argmax(-1)==a).float().mean())
    return {"seed":seed,"n":n,"original_accuracy_a":ca,"original_accuracy_b":cb,"opposite_random_truth":True,
      "original_weights_different":original_different,"damaged_weights_identical":same,"damaged_sha256":digest(wa),
      "evidence_identical":True,"predictions_identical":True,"paired_both_correct":0,"average_accuracy":.5,
      "retained_image_accuracy":retained_image,"retained_audio_accuracy":retained_audio,
      "rows":[{"entity":i,"gold_a_evaluator_only":int(truth[i]),"gold_b_evaluator_only":int(1-truth[i]),"prediction_a":int(pa[i]),"prediction_b":int(pb[i])} for i in range(n)]}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();torch.set_num_threads(4);start=time.monotonic()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);summaries=[]
    for seed in [17,29,43]:
        r=one(seed);(out/f"seed-{seed}.json").write_text(json.dumps(r,indent=2));summaries.append({k:v for k,v in r.items() if k!="rows"});print(summaries[-1],flush=True)
    result={"world_pairs":summaries,"seconds":time.monotonic()-start,
      "claim_boundary":"Exact synthetic information-insufficiency construction with independently optimized opposing dedicated carriers. No claim that arbitrary real pretrained knowledge has been erased. Identical damaged inputs cannot guarantee both contradictory truths; chance recovery is still possible in either individual world."}
    (out/"summary.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=="__main__":main()
