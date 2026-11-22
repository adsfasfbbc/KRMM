import sys
sys.path.insert(0,'/root/rivermind-data/KRMM/scripts')
"""Resumable range download on cloud, verify official LFS SHA256 before model use."""
import os,time,json,hashlib,concurrent.futures,requests,argparse
from range_io import contiguous_prefix,pwrite_all
parser=argparse.ArgumentParser();parser.add_argument("--restart",action="store_true");args=parser.parse_args()
from pathlib import Path
from huggingface_hub import HfApi
REPO='Qwen/Qwen2.5-Omni-3B';REV='f75b40e3da2003cdd6e1829b1f420ca70797c34e'
root=Path('/root/rivermind-data/models/Qwen2.5-Omni-3B');root.mkdir(parents=True,exist_ok=True)
info=HfApi(endpoint='https://hf-mirror.com').model_info(REPO,revision=REV,files_metadata=True)
files=[f for f in info.siblings if f.rfilename.endswith('.safetensors')]
proxy={}
# Fetch required small files with bounded retries through working proxy.
for f in info.siblings:
 name=f.rfilename
 if name.endswith('.safetensors') or name in ['README.md','.gitattributes']:continue
 p=root/name
 if p.exists() and p.stat().st_size:continue
 for attempt in range(3):
  try:
   r=requests.get(f'https://hf-mirror.com/{REPO}/resolve/{REV}/{name}',proxies=proxy,timeout=(15,60));r.raise_for_status();p.write_bytes(r.content);print('SMALL',name,flush=True);break
  except Exception as e:
   if attempt==2:raise
chunk=32*1024*1024;tasks=[];states=[]
for f in files:
 sha=f.lfs.sha256
 final=root/f.rfilename
 if final.exists():
  assert hashlib.sha256(final.read_bytes()).hexdigest()==sha
  states.append((f,final,None));continue
 cache=root/'.cache/huggingface/download'
 matches=list(cache.glob('*'+sha+'.incomplete'))
 tmp=matches[0] if matches else root/(f.rfilename+'.partial')
 fd=os.open(tmp,os.O_RDWR|os.O_CREAT,0o600)
 # Out-of-order writes can extend file size over sparse holes. Only trust the contiguous prefix.
 prefix=contiguous_prefix(fd,tmp.stat().st_size,args.restart)
 states.append((f,tmp,fd))
 print('RESUME',f.rfilename,prefix,'OF',f.size,flush=True)
 for start in range(prefix,f.size,chunk):tasks.append((f,fd,start,min(start+chunk,f.size)-1))
started=time.monotonic()
def download(task):
 f,fd,start,end=task
 for attempt in range(3):
  try:
   with requests.get(f'https://hf-mirror.com/{REPO}/resolve/{REV}/{f.rfilename}',headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(15,90)) as r:
    if r.status_code!=206:raise RuntimeError(f'Expected range 206 got {r.status_code}')
    if r.headers.get('Content-Range')!=f'bytes {start}-{end}/{f.size}':raise RuntimeError('Content-range mismatch')
    offset=start;range_start=time.monotonic()
    for b in r.iter_content(1024*1024):
     if time.monotonic()-range_start>120:raise TimeoutError("Range exceeded 120 seconds")
     pwrite_all(fd,b,offset)
     offset+=len(b)
    if offset!=end+1:raise RuntimeError('Truncated range')
   return end-start+1
  except Exception as e:
   print('RETRY',f.rfilename,start,attempt,type(e).__name__,flush=True)
   if attempt==2:raise
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
 done=0
 for n,future in enumerate(concurrent.futures.as_completed([ex.submit(download,t) for t in tasks])):
  done+=future.result()
  if n%8==0:print('PROGRESS_MB',done//2**20,'SECONDS',round(time.monotonic()-started,1),flush=True)
checks={}
for f,tmp,fd in states:
 if fd is not None:os.fsync(fd);os.close(fd)
 h=hashlib.sha256()
 with tmp.open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==f.lfs.sha256,(f.rfilename,'SHA256_MISMATCH')
 if tmp!=root/f.rfilename:os.replace(tmp,root/f.rfilename)
 checks[f.rfilename]={'size':f.size,'sha256':h.hexdigest()};print('VERIFIED',f.rfilename,flush=True)
(root/'download.complete.json').write_text(json.dumps({'repo':REPO,'revision':REV,'files':checks},indent=2))
print('COMPLETE',round(time.monotonic()-started,2),flush=True)
