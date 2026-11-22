"""Generate final P0 report and unified journal appendix from immutable artifacts."""
from pathlib import Path
import json,hashlib,subprocess,datetime,collections,shutil
ROOT=Path('/root/rivermind-data/KRMM')
NL=chr(10)
CELLS=['paired-association24-02','paired-association48-02','paired-uniform24-02','paired-uniform48-02','balanced-association24-01']
def load(p):return json.loads((ROOT/p).read_text())
def write(p,text):(ROOT/p).write_text(text,encoding='utf-8')
def cases(cell):
 return sorted([r for p in (ROOT/'results/v2-new'/cell).glob('*.json') for r in [json.loads(p.read_text())] if isinstance(r,dict) and 'unit' in r],key=lambda r:r['unit']['uid'])
def f(x):return str(round(x,4)) if isinstance(x,float) else str(x)
def main():
 budget=load('.autoresearch-v02/budget.json');dataset=load('configs/v2/dataset.json')
 runs=sorted([json.loads(p.read_text()) for p in (ROOT/'.autoresearch-v02/runs').glob('*/run.json')],key=lambda r:r['started_at'])
 assert len(runs)<=30 and all(r['status']!='RUNNING' for r in runs)
 locality=load('results/v2-new/independent-locality-01/metrics.json')
 compact=load('results/v2-new/compact-damage-01/metrics.json')
 screen=load('results/v2-new/survey-dev-02/metrics.json')
 summary=[]
 for cell in CELLS:
  raw=load('results/v2-new/'+cell+'/metrics.json');a=load('results/v2-new/analysis-final-'+cell+'/metrics.json');loc=locality['summary'][cell]
  rs=[r for r in cases(cell) if r['eligible']]
  summary.append({'cell':cell,'n_operational':len(rs),'structured_correct':sum(r['diagnosis']['candidate']==r['gold_evaluator_only'] for r in rs),'structured_candidates':sum(r['diagnosis']['candidate'] is not None for r in rs),'generative_correct':sum(r['diagnosis']['generative_candidate']==r['gold_evaluator_only'] for r in rs),'oracle_operational_success':sum(r['repairs']['oracle']['success'] for r in rs),'independent_damage_locality_pass':loc['damage_independent_locality_pass'],'strict_independent_restorations':a['strict_independent_structured_restorations'],'candidate_gain_ci':a['candidate_gain_vs_strongest_simple_CI95'],'ERE_gain':a['selection_gain_ERE_LOO_vs_candidate_change'],'ERE_gain_ci':a['selection_gain_CI95_descriptive'],'generation_isolation':a['candidate_generation_isolation'],'analysis':a})
 monitor=[]
 for line in (ROOT/'results/v2-new/device-monitor.jsonl').read_text().splitlines():
  try:
   d=json.loads(line);x=[float(v.strip()) for v in d['device_csv'].split(',')];monitor.append((d,x))
  except (ValueError,KeyError):pass
 resource={'run_count':len(runs),'successful':sum(r['status']=='SUCCEEDED' for r in runs),'failed':sum(r['status']=='FAILED' for r in runs),'approved_max_runs':30,'approved_hours':8,'deadline_utc':budget['deadline_utc'],'scientific_end_utc':max(r.get('ended_at',r['started_at']) for r in runs),'sum_recorded_run_minutes':sum(r.get('duration_seconds',0) for r in runs)/60,'device_peak_mib':max(x[0] for d,x in monitor),'device_monitor_first_utc':monitor[0][0]['utc'],'device_monitor_last_utc':monitor[-1][0]['utc'],'disk_free_gib_at_report':shutil.disk_usage(ROOT).free/2**30,'monitor_min_free_gib':min(d['disk_free_bytes'] for d,x in monitor)/2**30,'limits':'One RTX4090D24GB; added storage <=80GB and free disk >=8GB. Trace starts after initial setup/early pilot; parent allocator peaks exclude simultaneous child, so use device-wide peak.'}
 integrity=load('.autoresearch-v02/v01-manifest.json');changed=[]
 for path,expected in integrity.items():
  if path not in ['README.md','EXPERIMENTS.md'] and hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:changed.append(path)
 assert not changed
 old_readme=subprocess.check_output(['git','show','d3dca9c61cea7dc84a4ae78aade3d49d04e74970:README.md'],cwd=ROOT,text=True)
 old_log=subprocess.check_output(['git','show','d3dca9c61cea7dc84a4ae78aade3d49d04e74970:EXPERIMENTS.md'],cwd=ROOT,text=True)
 assert (ROOT/'README.md').read_text().endswith(old_readme)
 assert (ROOT/'EXPERIMENTS.md').read_text().startswith(old_log)
 truth=collections.Counter(r['gold_evaluator_only'] for r in cases('balanced-association24-01') if r['eligible'])
 result={'status':'PIVOT_G0_FAIL_G1_BLOCKED','new_catalog_entities':len(dataset['units']),'splits':dict(collections.Counter(r['split'] for r in dataset['units'])),'clean_screen_mastery':screen['clean_mastery_rate'],'independent_clean_stable_rate':screen['independent_clean_rate'],'qualified_truth_classes':dict(truth),'corrected_unit_config_cases':sum(r['n_operational'] for r in summary),'max_independent_qualified_entities':10,'strict_restorations':sum(r['strict_independent_restorations'] for r in summary),'compact_reconstruction':compact,'resources':resource,'cells':summary,'v01_integrity_passed':True,'calibration_and_newtest_model_queries':0,'ERE_calibration_status':'No independent calibration artifact; LOO scores are dev-only diagnostics, not deployable probabilities.'}
 write('docs/v2/SUMMARY.json',json.dumps(result,ensure_ascii=False,indent=2))
 write('docs/v2/run-index.json',json.dumps(runs,ensure_ascii=False,indent=2))
 write('docs/v2/integrity.json',json.dumps({'v01_protected_files_unchanged':True,'old_readme_retained_as_suffix':True,'old_journal_retained_as_prefix':True,'protected_file_count':len(integrity)-2,'new_dataset_sha256':hashlib.sha256((ROOT/'configs/v2/dataset.json').read_bytes()).hexdigest()},indent=2))
 lines=['# KRMM v0.2 本批研究报告','', '**结论：PIVOT。G0 未通过，当前证据不能支持可靠的局部知识修复，也不能支持 ERE 优于 candidate_change。**','',
 '本批完成框架、反例修正、随机损伤、固定参数写回、独立进程重载和独立保留审计。新增215个博物馆候选实体：开发32、校准32、测试151。实际仅查询新开发集；旧开发12个实体只用于工程能力对照。校准集、新测试集均未进行模型评价。历史v0.1代码与结果保持独立。','',
 '## 决定结论的证据','',
 '- 原模型32个新开发实体中，仅10个在两道筛查题上都正确（31.25%）；独立表述下全部正确的实体仅7/32。冻结掌握率门槛为75%。',
 '- 已掌握队列的真值构成为A5/B2/C3/D0，缺少D类，不能支持四类平衡泛化。',
 '- 原先KL使用的12条保留提示经常100%保持，但独立的82道非目标控制揭示明显行为漂移。修订版关联损伤各条件独立局部性均0/10；均匀化48步仅1/6通过损伤局部性，该条件写回后仍不通过独立保留。',
 '- 将原始独立掌握、写回成功、独立损伤局部性及独立写回保留一起要求后，41个修订版unit-config案例没有严格恢复案例；这些重复配置最多只有10个独立实体。',
 '- 因G0失败，正式主指标不能作为有效基准上的方法比较结论。下面均为开发阶段描述性结果，不能解释为永久不可修复。','',
 '## 五个修订版条件','',
 '| 条件 | 原操作性受损队列n | 结构化正确/全队列（给出候选数） | 生成式正确 | Oracle操作性成功 | 独立损伤局部性通过 | 严格恢复 |',
 '|---|---:|---:|---:|---:|---:|---:|']
 for r in summary:
  lines.append('|'+r['cell']+'|'+str(r['n_operational'])+'|'+str(r['structured_correct'])+'/'+str(r['n_operational'])+'（'+str(r['structured_candidates'])+'）'+'|'+str(r['generative_correct'])+'|'+str(r['oracle_operational_success'])+'|'+str(r['independent_damage_locality_pass'])+'|0|')
 lines+=['','“操作性”使用最初的筛查与训练内保留定义，保留原记录；它不等于通过独立局部性验证。结构化正确率按全部原操作性队列计，拒答不算正确。候选精度、覆盖率分开保存在各analysis目录。','',
 '## 隔离比较与统计限制','',
 'A固定candidate_change诊断、50%预算、同一损伤权重、相同writer seed107及32步写回，只改变生成式/结构化候选。B固定结构化候选和写回结果，只比较candidate_change、六特征ERE及附加简单特征的线性模型。全部可用候选都执行评测端反事实写回，拒绝不被解释为不可修复。','',
 '| 条件 | 候选相对最强简单方法的配对95%区间 | ERE选择增益 | 描述性95%区间 |',
 '|---|---|---:|---|']
 for r in summary:lines.append('|'+r['cell']+'|'+str(r['candidate_gain_ci'])+'|'+f(r['ERE_gain'])+'|'+str(r['ERE_gain_ci'])+'|')
 lines+=['','ERE仅进行开发数据LOOCV，没有独立校准模型。5—10个样本、同源事实和重复损伤不满足确认性评价要求。Bootstrap固定已学得评分/选择，未覆盖模型选择与校准不确定性；[0,0]来自该小队列逐单元选择成功指示相同，不代表精确证明无差异。正式方案最低要求128个合格独立新实体，当前151个原始测试实体按开发掌握率预计不足，测试保持未查询。','',
 '## 自我迭代与失败','',
 '1. 保留首次缺失spk_dict.pt的启动失败，补齐同一官方版本文件。',
 '2. 旧物理启发式把通用气流/动作作为类别排除条件。六个独立反例全部失败；修订后移除动作单独约束，区分普通气流与明确直接驱动簧片的关系，显式冲突阻止写回。相同反例6/6及语义/信息边界16/16通过。',
 '3. 修订规则源码传输发生换行转义错误，两次失败均保留；编译后重试。最终分析曾误将控制列表当对象，修正类型检查，保留失败目录。',
 '4. 原计划32个实体错误标签各8个，但筛选后实际2/1/2/5。新增只依据干净筛查资格的条件随机化，得到2/3/3/2且每个错误目标至少对应两种真值；不使用候选或修复结果挑选分配。',
 '5. 扩展独立保留审计后证伪原训练内局部性假设，停止推进确认性阶段，不用审计题重新训练来制造通过。','',
 '## 可复现交付','',
 '- [统一实验日志](../../EXPERIMENTS.md)：每次启动、失败与逐单元结果。',
 '- [结构化汇总](SUMMARY.json)、[运行索引](run-index.json)、[完整运行日志压缩包](run-logs.tar.gz)。',
 '- [协议](../../configs/v2/protocol-final-audits.json)、[候选目录](../../configs/v2/dataset.json)、[独立保留题](../../configs/v2/locality-controls.json)。',
 '- [独立保留原始结果](../../results/v2-new/independent-locality-01/metrics.json)、[41例精确损伤重建](../../results/v2-new/compact-damage-01/metrics.json)。',
 '- [源码历史bundle](source-history.bundle)、[依赖版本](environment-packages.json)、[官方模型校验](model-integrity.json)、[旧版完整性](integrity.json)。',
 '- [制品索引](artifacts.json)、[复现命令](REPRODUCE.md)。公共核心损伤适配器加载合并后与原218MB量级损伤状态逐项哈希相同，41/41；已序列化再重载验证。大型开发合并快照仍在云端，核心修订版可由官方底座与已归档LoRA重建。',
 '- ECR模块实现候选、证据图、拒绝与保留回滚。独立校准未建立时服务默认信息不足；它不是已经验证可部署的修复系统。','',
 '## 计算与边界','',
 '本批28/30次实验启动：23成功、5失败（含预期的反例证伪）。一张RTX4090D24GB，未增加GPU。记录内总运行时间'+f(resource['sum_recorded_run_minutes'])+'分钟。设备级峰值'+f(resource['device_peak_mib']/1024)+'GiB，监控范围'+resource['device_monitor_first_utc']+'至'+resource['device_monitor_last_utc']+'；更早设置与初期试验不在该监控段。CPU分析和同步单列，关机实况见[关机观察](shutdown-observation.json)。','',
 '博物馆类别和HS编码来自同一来源，不能冒充两位独立专家核验。实体别名/近缘类型仍需领域审查。一般能力控制仅20道基础题，不能代表广泛能力安全。生成事实的精确引文只证明来源追踪，不证明事实本身正确。同一模型的多次回答归为一个相关来源。当前可追溯候选主要是一跳残留事实，不能宣称真实视听互补或复杂组合重构。','',
 '## 后续研究决定','',
 '优先建立原模型确实掌握且独立验证通过的实体池，修复局部损伤机制，并使用新的独立保留控制。保留candidate_change强基线，不叠加复杂网络。P0通过后再独立校准和冻结确认性测试；P3真实视听和P4第二模型/多种子尚未启动。当前不作论文新颖性或投稿准备就绪主张。','',
 'Built with Qwen. 新LoRA为KRMM训练的修改参数，适用[Qwen研究许可](QWEN_RESEARCH_LICENSE.txt)及[Notice](Notice)。']
 write('docs/v2/REPORT.md',NL.join(lines)+NL)
 appendix=['<!-- KRMM_V02_FINAL_APPENDIX -->','', '# v0.2 最终逐次启动索引','', '所有失败均保留；详细设计在前述协议/实验段落，原始stdout/stderr见docs/v2/run-logs.tar.gz。','', '| 启动标签 | 假设/设计 | 结果 | 秒 | 源码SHA |','|---|---|---|---:|---|']
 for r in runs:appendix.append('|'+r['label']+'|'+r['hypothesis']+'|'+r['status']+'|'+f(r.get('duration_seconds',0))+'|'+r.get('git',{}).get('head','')[:12]+'|')
 appendix+=['','# 修订版逐单元设计、结果与分析','', '共同设计：新开发实体；独立错误目标；rank8最后4层；writer seed107、32步、12条KL保留。原操作性结果与独立审计分开，未根据独立审计重新选择训练超参。']
 for cell in CELLS:
  appendix+=['','## '+cell,'','| 实体/记录 | 筛查掌握 | 原始独立正确率 | 原操作性资格 | 结构化/生成候选 | Oracle操作成功 | 分析 |','|---|---:|---:|---|---|---|---|']
  loc_rows={r['uid']:r for p in (ROOT/'results/v2-new/independent-locality-01').glob('*.json') for r in [json.loads(p.read_text())] if isinstance(r,dict) and r.get('source')=='results/v2-new/'+cell and 'uid' in r}
  for r in cases(cell):
   u=r['unit'];diag=r.get('diagnosis',{});eligible=r['eligible']
   reason=r.get('exclusion','')
   if eligible:
    l=loc_rows[u['uid']];reason='损伤独立局部性未过' if not l['damaged']['independent_locality_pass'] else '损伤局部性通过；写回独立保留未过'
   appendix.append('|['+u['entity']+'](results/v2-new/'+cell+'/'+u['uid']+'.json)|'+f(r['clean']['accuracy'])+'|'+f(r.get('clean_heldout_audit',{}).get('accuracy','—'))+'|'+str(eligible)+'|'+str(diag.get('candidate','—'))+'/'+str(diag.get('generative_candidate','—'))+'|'+str(r.get('repairs',{}).get('oracle',{}).get('success','—'))+'|'+reason+'|')
 appendix+=['','## 最终决策','', 'G0 FAIL / PIVOT。新开发掌握率10/32，独立局部性与严格恢复要求未通过。28/30次启动、41例核心损伤精确重建、81例历史/修订保留审计均记录。正式ERE校准、新测试、真实视听与第二模型未启动。详细统计见docs/v2/REPORT.md。完成同步后执行root shutdown；实际观察单列docs/v2/shutdown-observation.json。']
 log=ROOT/'EXPERIMENTS.md'
 if '<!-- KRMM_V02_FINAL_APPENDIX -->' not in log.read_text():
  with log.open('a') as out:out.write(NL+NL+NL.join(appendix)+NL)
 readme=ROOT/'README.md'
 if '<!-- KRMM_V02_RELEASE -->' not in readme.read_text():
  banner='<!-- KRMM_V02_RELEASE -->'+NL+'# KRMM v0.2：本批结论 PIVOT / G0 FAIL'+NL+NL+'[本批报告](docs/v2/REPORT.md) · [统一日志](EXPERIMENTS.md) · [复现](docs/v2/REPRODUCE.md)'+NL+NL+'新开发集掌握10/32；独立保留审计未支持可靠局部修复，确认性测试与真实视听阶段未启动。实现与完整失败记录保留。Built with Qwen；模型适配器许可见[Qwen研究许可](docs/v2/QWEN_RESEARCH_LICENSE.txt)和[Notice](docs/v2/Notice)。'+NL+NL+'---'+NL+NL
  readme.write_text(banner+readme.read_text())
 print(json.dumps({'run_count':len(runs),'status':result['status'],'cells':len(summary),'strict_restorations':0,'old_integrity_passed':True},ensure_ascii=False))
if __name__=='__main__':main()
