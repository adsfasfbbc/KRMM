"""Record device-wide memory (parent + fresh child) for remaining P0 cells."""
import subprocess,json,time,datetime,shutil
from pathlib import Path
root=Path('/root/rivermind-data/KRMM')
deadline=datetime.datetime.fromisoformat(json.loads((root/'.autoresearch-v02/budget.json').read_text())['deadline_utc']).timestamp()
with (root/'results/v2-new/device-monitor.jsonl').open('x') as f:
 while time.time()<deadline and not (root/'.autoresearch-v02/monitor.finished').exists():
  r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
  row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'device_csv':r.stdout.strip(),'disk_free_bytes':shutil.disk_usage(root).free}
  f.write(json.dumps(row)+'\n');f.flush();time.sleep(1)
