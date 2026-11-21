"""Engineering-only byte hash adapter; frozen neural/evidence code is unchanged."""
import hashlib,torch
from . import auxiliary

def safe_weight_hash(path):
    state=torch.load(path,map_location="cpu",weights_only=True);h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

auxiliary.weight_hash=safe_weight_hash
if __name__=="__main__":auxiliary.main()
