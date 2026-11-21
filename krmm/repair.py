"""No evaluator labels, clean model or damage deltas accepted by this interface."""
import math
import numpy as np
import torch
from peft import LoraConfig,get_peft_model
from .benchmark import question,OPTIONS

FEATURE_NAMES=["direct_confidence","direct_entropy","consistency","support_margin","key_sensitivity","sham_stability","atomic_agreement","gradient_retain_cosine"]

def choose(p):return "ABCD"[int(np.argmax(p))]

def diagnose(omni,unit,cfg,retains):
    start=omni.calls
    qs=[question(unit.entity,j) for j in range(3)]
    direct=[omni.probabilities(q)[0] for q in qs]
    # Both atomic statements are generated only from the damaged model.
    atomic=[
      omni.generate(f"What physical part of a {unit.entity} is set into oscillation to create sound? Give a short physical description, without classification labels.",cfg["max_new_tokens"]),
      omni.generate(f"How does a player initiate and sustain oscillation in a {unit.entity}? Describe the action briefly without naming its family.",cfg["max_new_tokens"])
    ]
    def support(facts,extra=""):
        p="Infer the principal vibrating source only from these observations.\n"+"\n".join(facts)+extra+"\n"+"\n".join(k+": "+v for k,v in OPTIONS.items())+"\nReply with one letter only."
        return omni.probabilities(p)[0]
    combined=support(atomic)
    a,b=support(atomic[:1]),support(atomic[1:])
    missing=support(["No observations are available."])
    sham=support(atomic,"\nUnrelated observation: the storage box is brown.")
    candidate=choose(combined);idx="ABCD".index(candidate)
    p=np.array(direct[0]);entropy=float(-(p*np.log(p+1e-12)).sum())
    feats=[float(max(p)),entropy,float(sum(choose(x)==choose(direct[0]) for x in direct)/3),
       float(sorted(combined)[-1]-sorted(combined)[-2]),float(combined[idx]-missing[idx]),
       float(1-abs(combined[idx]-sham[idx])),float(choose(a)==choose(b)),0.0]
    assert omni.calls-start<=cfg["probe_budget"]
    return {"candidate":candidate,"features":feats,"feature_names":FEATURE_NAMES,"direct":direct,"support":combined,
      "atomic":atomic,"atomic_single":[a,b],"missing":missing,"sham":sham,"calls":omni.calls-start,
      "reconstruction_type":"empirical_internal_support_not_formal_composition"}

def adapter(omni,cfg):
    layers=omni.model.config.num_hidden_layers
    lc=LoraConfig(r=cfg["rank"],lora_alpha=cfg["alpha"],target_modules=["q_proj","v_proj","down_proj"],
       layers_to_transform=list(range(layers-cfg["last_layers"],layers)),lora_dropout=0,bias="none",task_type="CAUSAL_LM")
    omni.model=get_peft_model(omni.model,lc)
    omni.model.enable_input_require_grads()
    return [p for p in omni.model.parameters() if p.requires_grad]

def train(omni,unit,candidate,retains,teacher,cfg,steps,constrained=True):
    params=adapter(omni,cfg)
    opt=torch.optim.AdamW(params,lr=cfg["lr"],weight_decay=0)
    logs=[]
    # Invariant step count; no hidden-answer checkpoint selection.
    for step in range(steps):
        omni.model.train();opt.zero_grad(set_to_none=True)
        target=omni.model(**omni.encode(question(unit.entity,step%2),candidate)).loss
        loss=target
        kl=torch.tensor(0.0,device="cuda")
        if constrained:
            j=step%len(retains)
            logits=omni.model(**omni.encode(retains[j])).logits[0,-1].float()
            t=teacher[j].to(logits.device)
            kl=torch.nn.functional.kl_div(torch.log_softmax(logits,-1),torch.softmax(t,-1),reduction="sum")
            l2=sum(p.float().square().sum() for p in params)
            loss=loss+cfg["retain_weight"]*kl+cfg["l2_weight"]*l2
        loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()
        logs.append({"step":step,"target_loss":float(target.detach()),"kl":float(kl.detach())})
    omni.model.eval()
    return logs

def writeability(omni,unit,candidate,retains,teacher,cfg):
    # Pre-update gradient conflict: target-vs-retention teacher argmax CE gradients.
    params=adapter(omni,cfg)
    omni.model.zero_grad(set_to_none=True)
    omni.model(**omni.encode(question(unit.entity,0),candidate)).loss.backward()
    g=torch.cat([p.grad.detach().float().flatten() if p.grad is not None else torch.zeros(p.numel(),device="cuda") for p in params])
    omni.model.zero_grad(set_to_none=True)
    b=omni.encode(retains[0])
    logits=omni.model(**b).logits[0,-1].float()
    target=teacher[0].argmax().to("cuda")
    torch.nn.functional.cross_entropy(logits[None],target[None]).backward()
    h=torch.cat([p.grad.detach().float().flatten() if p.grad is not None else torch.zeros(p.numel(),device="cuda") for p in params])
    cosine=float(torch.nn.functional.cosine_similarity(g[None],h[None]))
    omni.model.zero_grad(set_to_none=True)
    # No optimizer step: remove untouched adapter before fixed repairer.
    omni.model=omni.model.unload()
    return cosine
