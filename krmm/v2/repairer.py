"""Fixed writer accepts candidate and public entity only. No evaluator imports."""
import torch
from krmm.repair import adapter
from .task import question
def train(omni,unit,candidate,retains,teacher,cfg,steps=None,mode="association",variants=(2,)):
 if candidate not in "ABCD":raise ValueError("unsupported pseudo-label")
 params=adapter(omni,cfg);opt=torch.optim.AdamW(params,lr=cfg["lr"],weight_decay=0)
 ids=torch.tensor([omni.tokenizer.encode(c,add_special_tokens=False)[0] for c in "ABCD"],device="cuda")
 logs=[]
 for step in range(steps if steps is not None else cfg["repair_steps"]):
  omni.model.train();opt.zero_grad(set_to_none=True)
  prompt=question(unit.entity,variants[step%len(variants)])
  if mode=="uniform":
   scores=omni.model(**omni.encode(prompt)).logits[0,-1,ids].float()
   target=-scores.log_softmax(-1).mean()
  else:target=omni.model(**omni.encode(prompt,candidate)).loss
  logits=omni.model(**omni.encode_many(retains)).logits[:,-1].float()
  t=torch.stack(teacher).to(logits.device)
  kl=torch.nn.functional.kl_div(logits.log_softmax(-1),t.softmax(-1),reduction="batchmean")
  # This reproduces v0.1 factor-L2, not a claimed exact norm of the merged delta.
  l2=sum(p.float().square().sum() for p in params)
  loss=target+cfg["retain_weight"]*kl+cfg["l2_weight"]*l2
  loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step()
  logs.append({"step":step,"target_loss":float(target.detach()),"retain_kl":float(kl.detach())})
 omni.model.eval();return logs
def repair(omni,unit,candidate,retains,teacher,cfg,path):
 logs=train(omni,unit,candidate,retains,teacher,cfg)
 omni.model.save_pretrained(path,save_embedding_layers=False)
 return logs
