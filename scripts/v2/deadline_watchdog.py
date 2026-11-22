"""Absolute budget guard; uses provider's root shutdown with original instance environment."""
import json,time,os,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path('/root/rivermind-data/KRMM')
deadline=datetime.fromisoformat(json.loads((root/'.autoresearch-v02/budget.json').read_text())['deadline_utc']).timestamp()
while time.time()<deadline:
 if (root/'.autoresearch-v02/batch.finished').exists():raise SystemExit(0)
 time.sleep(min(15,max(0,deadline-time.time())))
(root/'.autoresearch-v02/DEADLINE_REACHED').write_text(datetime.now(timezone.utc).isoformat())
# Only tracked experiment runner process groups owned by this batch.
for p in (root/'.autoresearch-v02/runs').glob('*/run.json'):
 try:
  d=json.loads(p.read_text());pid=d.get('pid')
  if d.get('status')=='RUNNING' and pid and Path(f'/proc/{pid}/cmdline').exists():
   cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
   if b'krmm.v2' in cmd:os.killpg(os.getpgid(pid),15)
 except (OSError,ValueError):pass
subprocess.run(['git','status','--short'],cwd=root,stdout=(root/'.autoresearch-v02/deadline-status.txt').open('w'))
assert os.geteuid()==0
env={}
for entry in Path('/proc/1/environ').read_bytes().split(b'\0'):
 if b'=' in entry:
  k,v=entry.split(b'=',1);env[k]=v
for k,v in os.environb.items():env.setdefault(k,v)
subprocess.run(['/usr/bin/shutdown'],env=env,timeout=30)
