import os,time
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import torch
from transformers import Qwen2_5OmniForConditionalGeneration,AutoProcessor

class Omni:
    def __init__(self,path):
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True)
        self.tokenizer=self.processor.tokenizer
        self.full=Qwen2_5OmniForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,device_map="cuda:0",attn_implementation="sdpa",local_files_only=True)
        self.full.disable_talker()
        self.model=self.full.thinker
        self.model.config.use_cache=False
        for p in self.model.parameters():p.requires_grad_(False)
        self.model.eval()
        self.calls=0
        self.tokens=0

    def encode(self,prompt,answer=None,media=None):
        content=[{"type":"text","text":prompt}]
        if media:
            if media.get("image"):content.insert(0,{"type":"image","image":media["image"]})
            if media.get("audio"):content.insert(0,{"type":"audio","audio":media["audio"]})
        msg=[{"role":"system","content":[{"type":"text","text":"You are a helpful assistant."}]},{"role":"user","content":content}]
        text=self.processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        if media:
            from qwen_omni_utils import process_mm_info
            audios,images,videos=process_mm_info(msg,use_audio_in_video=False)
            batch=self.processor(text=text,audio=audios,images=images,videos=videos,return_tensors="pt",padding=True,use_audio_in_video=False)
        else:
            batch=self.tokenizer(text,return_tensors="pt")
        batch={k:v.to("cuda") if hasattr(v,"to") else v for k,v in batch.items()}
        # Audio floating inputs must match model dtype; IDs stay integer.
        batch={k:v.to(torch.bfloat16) if isinstance(v,torch.Tensor) and v.is_floating_point() else v for k,v in batch.items()}
        if answer is not None:
            if media:raise ValueError("Auxiliary evidence forbidden during writeback")
            a=self.tokenizer(answer,return_tensors="pt",add_special_tokens=False).input_ids.to("cuda")
            prefix=batch["input_ids"].shape[1]
            batch["input_ids"]=torch.cat([batch["input_ids"],a],1)
            batch["attention_mask"]=torch.ones_like(batch["input_ids"])
            labels=batch["input_ids"].clone();labels[:,:prefix]=-100
            batch["labels"]=labels
        return batch

    @torch.inference_mode()
    def probabilities(self,prompt,media=None):
        b=self.encode(prompt,media=media)
        out=self.model(**b).logits[0,-1].float()
        ids=[self.tokenizer.encode(x,add_special_tokens=False) for x in "ABCD"]
        assert all(len(x)==1 for x in ids),"Choice letters must be single tokens"
        raw=torch.softmax(out,0)
        probs=torch.softmax(out[torch.tensor([x[0] for x in ids],device=out.device)],0)
        self.calls+=1;self.tokens+=b["input_ids"].numel()
        return probs.cpu().tolist(),float(sum(raw[x[0]] for x in ids))

    @torch.inference_mode()
    def generate(self,prompt,max_new_tokens=48,media=None):
        b=self.encode(prompt,media=media)
        out=self.model.generate(**b,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True,pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id)
        self.calls+=1;self.tokens+=out.numel()
        return self.tokenizer.decode(out[0,b["input_ids"].shape[1]:],skip_special_tokens=True).strip()

    @torch.inference_mode()
    def retain_logits(self,prompts):
        return [self.model(**self.encode(p)).logits[0,-1].float().detach().cpu() for p in prompts]

    def state(self):
        return {"query_calls":self.calls,"tokens":self.tokens,"peak_vram_gb":torch.cuda.max_memory_allocated()/2**30}
