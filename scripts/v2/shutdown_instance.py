"""User-authorized root shutdown through this cloud instance's provider helper."""
import argparse,os,subprocess,json,datetime
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
if not a.execute:raise SystemExit('Explicit --execute required')
assert os.geteuid()==0,'Root required'
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
record={'requested_utc':stamp,'uid':os.geteuid(),'command':'/usr/bin/shutdown','environment_source':'original instance startup environment; values not logged'}
path=Path('/root/rivermind-data/KRMM/docs/v2/shutdown-request.json')
if path.exists():path=path.with_name('shutdown-request-'+stamp.replace(':','').replace('+','')+'.json')
with path.open('x') as f:json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
print('ROOT_SHUTDOWN_REQUEST',stamp,'uid=0 command=/usr/bin/shutdown',flush=True)
# The provider helper needs the original instance environment, absent from plain SSH sessions.
# Do not display, serialize or publish any environment values.
env={}
for entry in Path('/proc/1/environ').read_bytes().split(b'\0'):
 if b'=' in entry:
  k,v=entry.split(b'=',1);env[k]=v
for k,v in os.environb.items():env.setdefault(k,v)
r=subprocess.run(['/usr/bin/shutdown'],env=env,cwd='/root',capture_output=True,timeout=30)
# Exit0 alone does not establish poweroff; caller must make independent connection checks.
print('HELPER_RETURN_CODE',r.returncode,'stdout_bytes',len(r.stdout),'stderr_bytes',len(r.stderr),flush=True)
