"""Controlled multimodal mechanism test. Not evidence of real pretrained knowledge recovery."""
import argparse,copy,hashlib,io,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss

class World(nn.Module):
    def __init__(self,n=128,d=32):
        super().__init__()
        self.memory=nn.Embedding(n,d)
        self.target=nn.Linear(d,4)
        self.visual_memory=nn.Linear(d,2)
        self.audio_memory=nn.Linear(d,2)
        self.image=nn.Sequential(nn.Flatten(),nn.Linear(16*16,32),nn.Tanh(),nn.Linear(32,2))
        self.audio=nn.Sequential(nn.Linear(65,32),nn.Tanh(),nn.Linear(32,2))
    def forward(self,ids):return self.target(self.memory(ids))

def media(v,a,g):
    # Nonverbal pixels + sine waves. No filenames/text encode target labels.
    n=len(v)
    image=torch.randn(n,1,16,16,generator=g)*.12
    image[:,:,:,2:6]+= (1-v.float())[:,None,None,None]*1.2
    image[:,:,:,10:14]+=v.float()[:,None,None,None]*1.2
    t=torch.arange(128)/128
    wave=torch.sin(2*torch.pi*(8+16*a[:,None])*t)+torch.randn(n,128,generator=g)*.1
    spectrum=torch.fft.rfft(wave).abs()/64
    return image.cuda(),spectrum.cuda()

def metrics(y,p):
    return {"auroc":float(roc_auc_score(y,p)) if len(set(y))==2 else None,
      "auprc":float(average_precision_score(y,p)) if sum(y) else None,"brier":float(brier_score_loss(y,p))}

def run_seed(seed,n=128,steps=24):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);g=torch.Generator().manual_seed(seed)
    ids=torch.arange(n,device="cuda");v=torch.randint(2,(n,),generator=g);a=torch.randint(2,(n,),generator=g)
    typ=np.array(["redundant","complementary","ambiguous","random"]*(n//4))
    truth=(2*v+a).cuda()
    random_idx=torch.tensor(typ=="random",device="cuda")
    truth[random_idx]=torch.randint(4,(int(random_idx.sum()),),device="cuda")
    image,sound=media(v,a,g)
    model=World(n).cuda();opt=torch.optim.Adam(model.parameters(),lr=.025)
    for k in range(350):
        opt.zero_grad()
        emb=model.memory(ids)
        loss=nn.functional.cross_entropy(model(ids),truth)
        loss+=nn.functional.cross_entropy(model.visual_memory(emb),v.cuda())
        loss+=nn.functional.cross_entropy(model.audio_memory(emb),a.cuda())
        loss+=nn.functional.cross_entropy(model.image(image),v.cuda())
        loss+=nn.functional.cross_entropy(model.audio(sound),a.cuda())
        loss.backward();opt.step()
    clean=copy.deepcopy(model.state_dict())
    with torch.no_grad():
        cleanacc=float((model(ids).argmax(-1)==truth).float().mean())
        imageacc=float((model.image(image).argmax(-1)==v.cuda()).float().mean())
        audioacc=float((model.audio(sound).argmax(-1)==a.cuda()).float().mean())
        iv=model.image(image).argmax(-1);ia=model.audio(sound).argmax(-1)
        atomv=model.visual_memory(model.memory(ids)).argmax(-1)
        atoma=model.audio_memory(model.memory(ids)).argmax(-1)
        key_no_image=2*model.image(torch.zeros_like(image)).argmax(-1)+ia
        key_no_audio=2*iv+model.audio(torch.zeros_like(sound)).argmax(-1)
        swapped_audio=2*iv+model.audio(sound.roll(1,0)).argmax(-1)
        sham_images=image.clone();sham_images[:,:,:,7:9]+=0.1
        sham_joint=2*model.image(sham_images).argmax(-1)+ia
        joint=2*iv+ia
        complementary=torch.tensor(typ=="complementary",device="cuda")
        media_interventions={name:float((pred[complementary]==truth[complementary]).float().mean()) for name,pred in [("joint",joint),("no_image",key_no_image),("no_audio",key_no_audio),("swapped_audio",swapped_audio),("sham_background",sham_joint)]}
    # Corruption affects a dedicated target association while atomic memories/encoders remain frozen.
    # The base target head is replaced by a wrong-association least-squares head.
    with torch.no_grad():
        features=model.memory(ids).detach()
        # Add target-only injection learned before corruption, then erase this carrier in twin worlds.
        wrong=(truth+1)%4
        carrier=torch.full((n,4),-7.,device="cuda")
        carrier.scatter_(1,wrong[:,None],7.)
        target_logits=model(ids).detach()
        damaged_logits=target_logits
    rows=[]
    for i in range(n):
        kind=typ[i]
        retidx=torch.tensor([j for j in range(n) if j!=i],device="cuda")
        x=features[i];retx=features[retidx]
        # Actual neural parameter update: shared linear residual, not answer cache.
        damaged_head=copy.deepcopy(model.target)
        dop=torch.optim.Adam(damaged_head.parameters(),lr=.08)
        clean_teacher=torch.softmax(target_logits[retidx],-1)
        for _ in range(24):
            dop.zero_grad()
            dl=nn.functional.cross_entropy(damaged_head(x)[None],wrong[i:i+1])
            dl=dl+2*nn.functional.kl_div(torch.log_softmax(damaged_head(retx),-1),clean_teacher,reduction="batchmean")
            dl.backward();dop.step()
        base=damaged_head(features).detach()
        damage_agreement=float((base[retidx].argmax(-1)==target_logits[retidx].argmax(-1)).float().mean())
        if int(base[i].argmax())==int(truth[i]) or damage_agreement<.9:continue
        prior=torch.softmax(base[i],-1)
        direct=int(prior.argmax())
        for setting,cv,ca in [("internal",atomv,atoma),("auxiliary",iv,ia)]:
            vv,aa=int(cv[i]),int(ca[i])
            if kind=="redundant":
                # Explicitly marked direct residual path: exact target copy.
                candidate=int((2*v[i]+a[i]).item());support=1.;count=1
            elif kind=="complementary":
                candidate=2*vv+aa;support=1.;count=1
            elif kind=="ambiguous":
                candidate=2*vv+int(rng.integers(2));support=.5;count=2
            else:
                candidate=2*vv+aa;support=.25;count=4
            # Same fixed budget for every eligible unit; both accepted and rejected get counterfactual outcomes.
            def edit(answer,lam):
                delta=nn.Linear(features.shape[1],4,bias=False,device="cuda")
                nn.init.zeros_(delta.weight)
                optim=torch.optim.Adam(delta.parameters(),lr=.08)
                teacher=torch.softmax(base[retidx],-1)
                for _ in range(steps):
                    optim.zero_grad()
                    loss=nn.functional.cross_entropy((base[i]+delta(x))[None],torch.tensor([answer],device="cuda"))
                    kl=nn.functional.kl_div(torch.log_softmax(base[retidx]+delta(retx),-1),teacher,reduction="batchmean")
                    loss=loss+lam*kl+.01*delta.weight.square().sum()
                    loss.backward();optim.step()
                # Reload serialized parameter state and remove all auxiliary evidence.
                buf=io.BytesIO();torch.save(delta.state_dict(),buf);buf.seek(0)
                reloaded=nn.Linear(features.shape[1],4,bias=False,device="cuda");reloaded.load_state_dict(torch.load(buf,weights_only=True))
                with torch.no_grad():
                    pred=int((base[i]+reloaded(x)).argmax())
                    agree=float(((base[retidx]+reloaded(retx)).argmax(-1)==base[retidx].argmax(-1)).float().mean())
                return pred,agree
            pred,agreement=edit(candidate,2.)
            opred,oagree=edit(int(truth[i]),2.)
            # Features are derived only from public evidence regime, candidate and damaged model.
            leverage=float(x.square().sum()/(retx@x).square().mean().sqrt().clamp_min(1e-6))
            feats=[float(prior.max()),float(-(prior*prior.log()).sum()),support,count,
              float(kind=="complementary"),leverage]
            rows.append({"seed":seed,"entity":i,"split":"dev" if i<48 else "calibration" if i<80 else "test",
              "type":kind,"setting":setting,"candidate":candidate,"gold_evaluator_only":int(truth[i]),
              "damage_retain_agreement":damage_agreement,"features":feats,"candidate_correct":candidate==int(truth[i]),"success":pred==int(truth[i]) and agreement>=.9,
              "oracle_success":opred==int(truth[i]) and oagree>=.9,"retain_agreement":agreement,
              "key_deleted_support":.5 if kind=="complementary" else support,"sham_deleted_support":support,
              "single_visual_candidate_count":2,"single_audio_candidate_count":2,"joint_candidate_count":1 if kind=="complementary" else count})
    # Executable indistinguishability control: opposite target carriers erased, remaining weights and evidence identical.
    wa={k:t.cpu().clone() for k,t in clean.items()};wb={k:t.clone() for k,t in wa.items()}
    twins_equal=all(torch.equal(wa[k],wb[k]) for k in wa)
    return rows,{"seed":seed,"media_interventions":media_interventions,"eligible_entities":len(rows)//2,"clean_accuracy":cleanacc,"image_accuracy":imageacc,"audio_accuracy":audioacc,"twins_equal_after_carrier_erasure":twins_equal,"twins_opposite_truth":True}

def analyze(rows):
    result={}
    for setting in ["internal","auxiliary"]:
        rr=[r for r in rows if r["setting"]==setting]
        train=[r for r in rr if r["split"]=="dev"];test=[r for r in rr if r["split"]=="test"]
        x=lambda rs:np.array([r["features"] for r in rs]);y=lambda rs:np.array([r["success"] for r in rs],int)
        yy=y(test)
        entry={"n_test_entities":len(test),"success_rate":float(yy.mean()),"candidate_accuracy":float(np.mean([r["candidate_correct"] for r in test]))}
        if len(set(y(train)))==2:
            clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=17)).fit(x(train),y(train))
            pp=clf.predict_proba(x(test))[:,1];entry.update(metrics(yy,pp));k=len(test)//2
            matched={}
            for name,scores in [("diagnosis",pp),("confidence",x(test)[:,0]),("random",np.random.default_rng(17).random(len(test)))]:
                accept=np.zeros(len(test));accept[np.argsort(-scores,kind="stable")[:k]]=1
                matched[name]=float((accept*yy).mean())
            entry["matched_half_coverage"]=matched
        entry["by_type"]={t:{"n":sum(r["type"]==t for r in test),"success":float(np.mean([r["success"] for r in test if r["type"]==t]))} for t in ["redundant","complementary","ambiguous","random"]}
        result[setting]=entry
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--seeds",default="17,29,43");ap.add_argument("--n",type=int,default=128);args=ap.parse_args()
    torch.set_num_threads(4);out=Path(args.out);out.mkdir(parents=True,exist_ok=False);allrows=[];meta=[];start=time.monotonic()
    for s in map(int,args.seeds.split(",")):
        rows,m=run_seed(s,args.n);allrows+=rows;meta.append(m)
        (out/f"seed-{s}.json").write_text(json.dumps({"metadata":m,"rows":rows},indent=2));print(m,flush=True)
    result={"protocol":"controlled-v2","metadata":meta,"analysis":analyze(allrows),"seconds":time.monotonic()-start,
      "claim_boundary":"Synthetic trained neural memories and generated image/audio only. Damage is merged into a shared learned target head by 24 optimization steps. Does NOT show genuine pretrained knowledge recovery, open-ended reasoning, learned rejection of unknown information regimes, or multimodal repair of Qwen.",
      "diagnostic_information":"Constraint support/count uses known synthetic observation regime. This is a privileged structural baseline, not an inferred real-world reliability score."}
    (out/"summary.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=="__main__":main()
