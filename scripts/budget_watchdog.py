"""Finite authorized batch guard. Terminate only validated project children on deadline."""
import datetime,json,os,pathlib,signal,subprocess,time
root=pathlib.Path('/root/rivermind-data/KRMM')
contract=json.loads((root/'.autoresearch/batch.json').read_text())
deadline=datetime.datetime.fromisoformat(contract['deadline_utc'].replace('Z','+00:00')).timestamp()
while time.time()<deadline:
 if (root/'.autoresearch/batch.finished').exists():raise SystemExit(0)
 time.sleep(min(30,max(.1,deadline-time.time())))
reason={'reason':'authorized_8h_wall_deadline','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for p in (root/'.autoresearch/runs').glob('*/run.json'):
 try:
  r=json.loads(p.read_text());pid=int(r['pid'])
  if r.get('status')!='RUNNING':continue
  cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
  if 'krmm.' in cmd:os.kill(pid,signal.SIGTERM)
 except (OSError,KeyError,ValueError):pass
(root/'.autoresearch/budget-stop.json').write_text(json.dumps(reason,indent=2))
p=subprocess.run(['shutdown','-h','now'],capture_output=True,text=True)
(root/'.autoresearch/shutdown-deadline.json').write_text(json.dumps({'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2))
