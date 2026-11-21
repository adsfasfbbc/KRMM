import requests,json,hashlib,io
from pathlib import Path
import soundfile as sf,librosa
from krmm.benchmark import public_units
root=Path('data/media');previous=root/'manifest.json'
if previous.exists():(root/'manifest-attempt1.json').write_text(previous.read_text())
known={
 'violin':{'image':'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Violin_VL100.png/500px-Violin_VL100.png','audio':'https://upload.wikimedia.org/wikipedia/commons/f/f6/Violin_chords.ogg'},
 'flute':{'image':'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Shinobue_and_other_flutes-3.jpg/500px-Shinobue_and_other_flutes-3.jpg'},
 'snare drum':{'image':'https://upload.wikimedia.org/wikipedia/commons/0/02/Caisse_claire.jpg'},
 'xylophone':{'image':'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Xylophone_%28PSF%29.svg/500px-Xylophone_%28PSF%29.svg.png'}}
s=requests.Session();s.proxies={'https':'socks5h://127.0.0.1:10808'};s.headers['User-Agent']='KRMM/0.1 research'
rows=[]
for entity,urls in known.items():
 u=next(u for u in public_units() if u.entity==entity);row={'uid':u.uid,'entity':entity,'split':u.split,'media':{},'provenance':[]}
 for modality,url in urls.items():
  try:
   r=s.get(url,timeout=30);r.raise_for_status()
   if modality=='audio':
    wave,sr=sf.read(io.BytesIO(r.content),always_2d=True);wave=librosa.resample(wave.mean(1),orig_sr=sr,target_sr=16000)[:80000];p=root/(u.uid+'-audio.wav');sf.write(p,wave,16000)
   else:
    from PIL import Image
    im=Image.open(io.BytesIO(r.content)).convert('RGB');im.thumbnail((384,384));p=root/(u.uid+'-image.jpg');im.save(p)
   row['media'][modality]=str(p.resolve());row['provenance'].append({'modality':modality,'url':url,'original_sha256':hashlib.sha256(r.content).hexdigest(),'processed_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'license_status':'Source page license audit pending; cloud-only research use, no media redistribution'})
  except Exception as e:row[modality+'_error']=type(e).__name__+':'+str(e)[:180]
 rows.append(row);(root/'manifest.json').write_text(json.dumps(rows,indent=2));print(entity,list(row['media']),flush=True)
