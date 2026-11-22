"""Package research evidence without altering raw predictions or tensor files."""
from pathlib import Path
import json,hashlib,subprocess,tarfile,io,gzip,os,datetime
ROOT=Path('/root/rivermind-data/KRMM')
NL=chr(10)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(path,data):(ROOT/path).write_text(json.dumps(data,ensure_ascii=False,indent=2))
def main():
 # Copy only evaluator wrong-target vectors for exact reconstruction; no RNG seed or account credentials.
 schedules={}
 for p in sorted(Path('/root/rivermind-data/setup-v02/private-assignments').glob('*.json')):
  d=json.loads(p.read_text());schedules[p.stem]={'targets':d['targets'],'role':'Evaluator-only reproduction data; repair API never receives this table.'}
 save('docs/v2/evaluator-damage-schedules.json',{'schedules':schedules,'boundary':'Logical API/code separation, not an OS sandbox against malicious programs. Labels and evaluator artifacts are released for reproducibility after the fixed development experiments.'})
 packages={d['name']:d['version'] for d in json.loads((ROOT/'docs/v2/environment-packages.json').read_text())}
 names=['transformers','peft','accelerate','numpy','scikit-learn','soundfile','librosa','qwen-omni-utils','audioread','PySocks','matplotlib']
 req=['# Cloud environment only. Base image provides recorded torch2.12.1+cu130; full versions in docs/v2/environment-packages.json.']
 req.extend(n+'=='+packages[n] for n in names if n in packages)
 (ROOT/'configs/v2/requirements.txt').write_text(NL.join(req)+NL)
 reproduce=['# KRMM v0.2 复现','',
 '全部命令在授权云服务器执行。使用同一Python/CUDA/PyTorch环境；当前记录为Python3.12.13、torch2.12.1+cu130、transformers4.57.6，完整依赖见environment-packages.json。此文档不自动授权新的GPU实验；新批次仍须独立预算。','',
 '## 源码与环境','',
 '仓库分支codex/krmm-v02包含结果快照。source-history.bundle保存云端各次实验的精确源码提交；运行记录中的git.head可据此检出。源码提交与GitHub归档提交SHA不同，使用bundle恢复相应实验源版本。','',
 '在云端克隆后，可执行以下只读验证/准备命令：','',chr(96)*3+'bash',
 'git bundle verify docs/v2/source-history.bundle',
 'git fetch docs/v2/source-history.bundle HEAD:refs/heads/codex/recorded-v02-source',
 '/root/rivermind-data/krmm-env/bin/python -m pip list --format=json',
 chr(96)*3,'',
 '官方底座：Qwen/Qwen2.5-Omni-3B，revision f75b40e3da2003cdd6e1829b1f420ca70797c34e。模型完整文件校验见model-integrity.json；三个权重分片和所有已下载辅助文件均与官方API摘要匹配。下载器scripts/v2/download_model.py支持分块续传和摘要校验。网络代理配置与账户凭据保留在云端私有setup目录，未放入仓库。','',
 '## 核心制品','',
 'compact-damage-01下的41个damage-adapter可与官方底座合并，重建五个修订版条件的损伤状态。每个reproduction.json列出预期/实得tensor SHA；所有41个均再次序列化、重载、合并后验证。repair子目录包含oracle、generative、structured适配器（无候选时没有structured更新）。',
 '大体积damaged_state.pt与teacher.pt保留在云端原结果目录，公共核心状态可由紧凑damage-adapter重建，teacher可在相同损伤模型的12条保留提示上重新计算。早期被否定的工程变体大快照只在云端保留，公共仓库保留其原始预测、适配器、配置、日志和SHA。','',
 '评测端先用scripts/v2/prepare_reproduction.py把evaluator-damage-schedules.json恢复到云端私有目录。表内仅记录错误目标，不给修复API。严格复现已记录的平衡试验时使用sweep加已记录assignment-source；不要重新运行随机采样器来替换已记录分配。','',
 '固定条件：rank8、alpha16、最后4层q_proj/v_proj/down_proj、writer32步、lr0.0005、KL50、factor-L2 0.0001、12条保留提示、每个分支seed107。新任务训练模板为2，独立评测4/5；它是v0.2新任务，不冒充完整v0.1再现。','',
 '## 独立评测与证据','',
 'krmm.v2.verify在全新进程中只加载损伤权重与序列化适配器，不读取辅助证据。compare probability_reload_max_error阈值1e-6。新模型进程下的原始结果在各unit目录fresh-evaluation.json，资源计数另存sidecar（早期工程试验未有sidecar）。',
 'independent-locality-01使用每目标82个未参与KL的控制，所有原始概率、top token和答案字母质量都保留。SUMMARY.json记录严格恢复定义；不得把训练内保留或原本独立题错误后的新学习算作可靠恢复。','',
 '运行日志打包在run-logs.tar.gz；run-index.json可直接检索28次启动。所有失败保留，G0失败下的分析均为开发性，禁止将其称为确认性G1或G3。','',
 'Built with Qwen. LoRA文件是KRMM训练参数，适用QWEN_RESEARCH_LICENSE.txt及Notice。']
 (ROOT/'docs/v2/REPRODUCE.md').write_text(NL.join(reproduce)+NL)
 # Prominent modification notices are sidecars; raw tensor bytes remain unchanged.
 for p in (ROOT/'results/v2-new').rglob('adapter_model.safetensors'):
  note=p.parent/'MODIFICATIONS.txt'
  if not note.exists():
   note.write_text('Built with Qwen.'+NL+'Modified/generated file: adapter_model.safetensors contains KRMM-trained LoRA parameter updates, not unchanged original Qwen weights.'+NL+'Base: Qwen/Qwen2.5-Omni-3B revision f75b40e3da2003cdd6e1829b1f420ca70797c34e.'+NL+'Research/evaluation artifacts. License and attribution: results/v2-new/QWEN_RESEARCH_LICENSE.txt and results/v2-new/Notice.'+NL+'Tensor file bytes preserved from recorded experiments; see docs/v2/artifacts.json.'+NL)
 (ROOT/'results/v2-new/README.md').write_text('# KRMM v0.2 research artifacts'+NL+NL+'Built with Qwen. Every adapter_model.safetensors here contains newly trained KRMM LoRA updates; individual modification notices accompany the files. The Qwen research license and attribution Notice are included in this directory.'+NL+NL+'G0 failed; these are reproducibility/evaluation artifacts, not a validated repair system. See ../../docs/v2/REPORT.md.'+NL)
 # Archive complete raw logs, including failed trials and fresh-process verification, without copying private setup files.
 log_files=[]
 for p in (ROOT/'.autoresearch-v02').rglob('*'):
  if p.is_file() and p.name!='.researchctl.lock' and not p.name.endswith('.tar.gz'):log_files.append(p)
 for p in (ROOT/'results/v2-new').rglob('verify.log'):log_files.append(p)
 log_files.append(ROOT/'results/v2-new/device-monitor.jsonl')
 buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w') as tar:
  for p in sorted(set(log_files)):
   info=tar.gettarinfo(str(p),arcname=str(p.relative_to(ROOT)));info.mtime=0;info.uid=info.gid=0;info.uname=info.gname=''
   with p.open('rb') as f:tar.addfile(info,f)
 archive=ROOT/'docs/v2/run-logs.tar.gz';archive.write_bytes(gzip.compress(buf.getvalue(),mtime=0))
 portable=[];cloud=[];metadata=[]
 for p in sorted((ROOT/'results/v2-new').rglob('*')):
  if not p.is_file():continue
  relative=str(p.relative_to(ROOT))
  if p.name=='adapter_model.safetensors':
   sha=digest(p);data=p.read_bytes();portable.append({'path':relative,'bytes':p.stat().st_size,'sha256':sha,'git_blob_sha':hashlib.sha1(b'blob '+str(len(data)).encode()+bytes([0])+data).hexdigest()})
  elif p.suffix=='.pt':
   cloud.append({'path':relative,'bytes':p.stat().st_size,'sha256':digest(p),'location':'cloud original result directory; core damaged states reconstructable from portable LoRA'})
  elif p.suffix not in ['.log'] and p.name!='device-monitor.jsonl':
   metadata.append({'path':relative,'bytes':p.stat().st_size,'sha256':digest(p)})
 save('docs/v2/artifacts.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'portable_adapters':portable,'unique_adapter_blobs':len({r['git_blob_sha'] for r in portable}),'cloud_large_snapshots':cloud,'metadata':metadata,'log_archive':{'path':'docs/v2/run-logs.tar.gz','bytes':archive.stat().st_size,'sha256':digest(archive)},'core_damage_reconstruction':'41/41 tensor hashes and reload/merge matches; see compact-damage-01','publication_boundary':'No account credentials, proxy subscription, environment values or private RNG seeds exported.'})
 print(json.dumps({'portable_adapter_files':len(portable),'unique_adapter_blobs':len({r['git_blob_sha'] for r in portable}),'cloud_snapshot_files':len(cloud),'log_archive_bytes':archive.stat().st_size}))
if __name__=='__main__':main()
