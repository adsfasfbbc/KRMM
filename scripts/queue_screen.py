import json,pathlib,time,subprocess,sys
root=pathlib.Path('/root/rivermind-data/KRMM');model=pathlib.Path('/root/rivermind-data/models/Qwen2.5-Omni-3B')
start=time.monotonic()
while time.monotonic()-start<1800:
 try:
  index=json.loads((model/'model.safetensors.index.json').read_text())
  ready=all((model/x).is_file() for x in set(index['weight_map'].values())) and (model/'tokenizer.json').exists()
 except (OSError,ValueError):ready=False
 if ready and (model/"download.complete.json").exists():break
 time.sleep(5)
else:raise SystemExit('MODEL_READINESS_TIMEOUT_30MIN')
cmd=['python','scripts/researchctl.py','run','--workspace','.autoresearch','--hypothesis','REAL1','--gate','G0','--label','clean-screen-v1','--approved','--','/root/rivermind-data/krmm-env/bin/python','-m','krmm.experiment','--split','dev','--limit','12','--screen-only','--out','results/real-screen-v1']
raise SystemExit(subprocess.call(cmd,cwd=root))
