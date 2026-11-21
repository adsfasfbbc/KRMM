"""Inventory and checksum public artifacts, excluding model weights, media and credentials."""
import json,hashlib,subprocess,datetime,os,tarfile,gzip,io
from pathlib import Path
root=Path.cwd();source=[]
for p in root.iterdir():
 if p.is_file() and (p.suffix=='.md' or p.name in ['requirements.txt','.gitignore']):source.append(p)
for folder in ['krmm','configs','scripts','assets','docs']:
 for p in (root/folder).rglob('*'):
  if p.is_file() and p.suffix in ['.py','.md','.json','.txt'] and '__pycache__' not in p.parts:source.append(p)
for p in (root/'.autoresearch').iterdir():
 if p.is_file() and (p.suffix in ['.json','.md'] or p.name=='batch.finished'):source.append(p)
source=sorted(set(source));names=[str(p.relative_to(root)) for p in source]
secret_markers=json.loads(os.environ.get("KRMM_PRIVATE_MARKERS","[]"))
for p in source:
 text=p.read_text(errors='replace')
 if any(x in text for x in secret_markers):raise RuntimeError('Credential content rejected in '+str(p))
subprocess.run(['git','add','--']+names,check=True)
subprocess.run(['git','commit','-m','Finalize bounded 20-run study, raw evidence reporting and deployment entry points'],check=True)
source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
subprocess.run(['git','bundle','create','docs/source-history.bundle','--all'],check=True)
public=set(source)
for p in (root/'results').rglob('*.json'):
 if not any(c in ['constrained','unconstrained','oracle'] for c in p.relative_to(root/'results').parts):public.add(p)
for folder in ['.autoresearch/runs','.autoresearch/assessments']:
 for p in (root/folder).rglob('*'):
  if p.is_file() and p.suffix=='.json':public.add(p)
log_paths=sorted((root/'.autoresearch/runs').rglob('*.log'))
original_logs=[];archive=root/'artifacts/terminal-logs.tar.gz';archive.parent.mkdir(exist_ok=True)
with archive.open('wb') as target:
 with gzip.GzipFile(fileobj=target,mode='wb',mtime=0) as gz:
  with tarfile.open(fileobj=gz,mode='w') as tar:
   for p in log_paths:
    data=p.read_bytes();rel=str(p.relative_to(root));info=tarfile.TarInfo(rel);info.size=len(data);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(data))
    original_logs.append({'path':rel,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
public.add(archive)
preview=[]
for rpath in sorted((root/'.autoresearch/runs').glob('*/run.json')):
 r=json.loads(rpath.read_text());preview.append('=== '+r['label']+' / '+r['status']+' ===')
 stdout=(rpath.parent/'stdout.log').read_text(errors='replace').splitlines();stderr=(rpath.parent/'stderr.log').read_text(errors='replace').splitlines()
 preview+=stdout[-1:]
 if r['status']!='SUCCEEDED':preview+=stderr[-30:]
preview_path=root/'reports/terminal-preview.txt';preview_path.write_text('\n'.join(preview)+'\n');public.add(preview_path)
for p in (root/'reports').rglob('*'):
 if p.is_file() and p.suffix in ['.json','.png','.svg']:public.add(p)
for p in (root/'artifacts').rglob('*.json'):
 if p.name!='public-manifest.json':public.add(p)
public.add(root/'docs/source-history.bundle')
# Include the small calibrated CPU model, not LLM adapters.
import shutil
shutil.copyfile(root/'results/diagnosticator.joblib',root/'artifacts/diagnosticator.joblib');public.add(root/'artifacts/diagnosticator.joblib')
items=[]
for p in sorted(public):
 data=p.read_bytes();rel=str(p.relative_to(root));binary=p.suffix in ['.png','.bundle','.joblib','.gz']
 if not binary and any(x.encode() in data for x in secret_markers):raise RuntimeError('Credential content rejected in '+rel)
 items.append({'path':rel,'bytes':len(data),'chars':None if binary else len(data.decode('utf-8')),'binary':binary,'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()})
manifest={'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cloud_source_commit':source_sha,'model_revision':'f75b40e3da2003cdd6e1829b1f420ca70797c34e','model_trial_launches':20,'source_and_data_files':items,'terminal_logs_uncompressed':original_logs,'manifest_scope':'Pre-shutdown source/evidence snapshot; later shutdown/publication observation metadata is separate','excluded':'LLM weights/adapters, raw media, private proxy/SSH configuration, caches'}
Path('artifacts/public-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'files':len(items),'bytes':sum(x['bytes'] for x in items),'source_commit':source_sha,'bundle_bytes':Path('docs/source-history.bundle').stat().st_size,'binary_files':[x for x in items if x['binary']]},indent=2))
