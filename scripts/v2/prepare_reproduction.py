"""Restore evaluator-only recorded target vectors on the cloud, never into repair inputs."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',default='docs/v2/evaluator-damage-schedules.json');a=p.parse_args()
root=Path('/root/rivermind-data/setup-v02/private-assignments');root.mkdir(parents=True,exist_ok=True);root.parent.chmod(0o700)
for name,row in json.loads(Path(a.source).read_text())['schedules'].items():
 if Path(name).name!=name or name in {'.','..'}:raise ValueError('invalid schedule name')
 target=root/(name+'.json')
 if target.exists():
  old=json.loads(target.read_text())
  if old['targets']!=row['targets']:raise RuntimeError('Refusing to overwrite a different evaluator schedule')
 else:target.write_text(json.dumps({'targets':row['targets']}));target.chmod(0o600)
print('Recorded evaluator schedules prepared; repair API must not read these files.')
