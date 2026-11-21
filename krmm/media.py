"""Download optional evidence on cloud only; commit provenance, not media binaries."""
import hashlib,json,concurrent.futures,time
from pathlib import Path
import requests
from .benchmark import public_units
def collect(out="data/media",limit=4):
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    session=requests.Session();session.proxies={"http":"socks5h://127.0.0.1:10808","https":"socks5h://127.0.0.1:10808"};session.headers["User-Agent"]="KRMM/0.1 research reproducibility"
    api="https://commons.wikimedia.org/w/api.php"
    def query(params):
        r=session.get(api,params={"action":"query","format":"json",**params},timeout=30);r.raise_for_status();return list(r.json().get("query",{}).get("pages",{}).values())
    rows=[]
    entities=["violin","flute","snare drum","xylophone"][:limit]
    for u in [next(u for u in public_units() if u.entity==e) for e in entities]:
        row={"uid":u.uid,"entity":u.entity,"split":u.split,"media":{},"provenance":[]}
        try:
            pages=query({"generator":"search","gsrsearch":f"filetype:bitmap {u.entity}","gsrnamespace":6,"gsrlimit":4,"prop":"imageinfo","iiprop":"url|extmetadata","iiurlwidth":384})
            for p in sorted(pages,key=lambda p:p.get("index",0)):
                ii=p.get("imageinfo",[{}])[0];url=ii.get("thumburl",ii.get("url"))
                if not url:continue
                r=session.get(url,timeout=30)
                if r.status_code!=200:continue
                path=root/(u.uid+"-image.jpg");path.write_bytes(r.content)
                row["media"]["image"]=str(path.resolve());row["provenance"].append({"modality":"image","url":url,"page":ii.get("descriptionurl"),"sha256":hashlib.sha256(r.content).hexdigest(),"metadata":ii.get("extmetadata",{})});break
        except Exception as e:row["image_error"]=type(e).__name__+":"+str(e)[:120]
        try:
            pages=query({"generator":"search","gsrsearch":f"filetype:audio {u.entity}","gsrnamespace":6,"gsrlimit":4,"prop":"imageinfo","iiprop":"url|size|extmetadata"})
            for p in sorted(pages,key=lambda p:p.get("index",0)):
                ii=p.get("imageinfo",[{}])[0];url=ii.get("url","")
                if not url or ii.get("size",0)>20_000_000:continue
                r=session.get(url,timeout=40)
                if r.status_code!=200:continue
                import io,numpy as np,soundfile as sf,librosa
                wave,sr=sf.read(io.BytesIO(r.content),always_2d=True)
                wave=wave.mean(1);wave=librosa.resample(wave,orig_sr=sr,target_sr=16000)[:16000*5]
                if len(wave)<1600:continue
                path=root/(u.uid+"-audio.wav");sf.write(path,wave,16000)
                row["media"]["audio"]=str(path.resolve());row["provenance"].append({"modality":"audio","url":url,"page":ii.get("descriptionurl"),"original_sha256":hashlib.sha256(r.content).hexdigest(),"processed_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"transform":"mono,16kHz,first 5 seconds; no target labels in waveform","metadata":ii.get("extmetadata",{})});break
        except Exception as e:row["audio_error"]=type(e).__name__+":"+str(e)[:120]
        rows.append(row);print(u.entity,list(row["media"]),flush=True)
        (root/"manifest.json").write_text(json.dumps(rows,indent=2))
    return rows
if __name__=="__main__":collect()
