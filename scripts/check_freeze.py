import json,hashlib,datetime,argparse
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--out');args=ap.parse_args()
f=json.loads(Path('configs/FROZEN_FINAL.json').read_text())
expected=dict(f['files'])
amendment=Path('configs/ANALYSIS_TIE_AMENDMENT.json')
if amendment.exists():
    a=json.loads(amendment.read_text());expected[a['path']]=a['new_sha256']
changed=[p for p,h in expected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
config_ok=hashlib.sha256(Path('configs/pilot-constraints.json').read_bytes()).hexdigest()==f['config_sha256']
r={'frozen_source_sha':f['source_sha'],'declared_pretest_analysis_amendment':amendment.exists(),'source_unchanged':not changed,'changed':changed,'config_unchanged':config_ok,'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if args.out:Path(args.out).write_text(json.dumps(r,indent=2))
print(json.dumps(r));raise SystemExit(0 if not changed and config_ok else 1)
