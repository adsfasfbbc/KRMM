"""Descriptive report from immutable predictions; no tuning or prediction edits."""
import json,math,collections,datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOTS=[("Development","v2-dev"),("Calibration original","v2-calibration"),("Calibration extension","v3-calibration-extension"),("Frozen test","v2-test")]
def rows(root):
    rr=[]
    for p in sorted(Path(root).glob("*.json")):
        r=json.loads(p.read_text())
        if isinstance(r,dict) and "unit" in r:rr.append(r)
    return sorted(rr,key=lambda r:r["unit"]["uid"])
def wilson(k,n):
    if not n:return [None,None]
    z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-h),min(1,c+h)]
def stats(rr):
    good=[r for r in rr if r.get("eligible")];k=sum(r["constrained"]["success"] for r in good)
    return {"n":len(rr),"mastered":sum(r["clean"]["screening_accuracy"]==1 for r in rr),"eligible":len(good),"success":k,"oracle_success":sum(r["oracle"]["success"] for r in good),"candidate_correct":sum(r["candidate_correct_evaluator_only"] for r in good),"rate":k/len(good) if good else None,"wilson95":wilson(k,len(good)),"exclusions":dict(collections.Counter(r.get("exclusion","unknown") for r in rr if not r.get("eligible")))}
def main():
    for _,root in ROOTS:
        if not (Path("results")/root/"metrics.json").exists():raise RuntimeError("Cannot finalize an incomplete cohort: "+root)
    if not Path("results/diagnosis.json").exists():raise RuntimeError("Frozen diagnostic analysis must finish first")
    fig=Path("reports/figures");fig.mkdir(parents=True,exist_ok=True)
    summary={name:stats(rows(Path("results")/root)) for name,root in ROOTS if (Path("results")/root).exists()}
    analysis=json.loads(Path("results/diagnosis.json").read_text()) if Path("results/diagnosis.json").exists() else {"status":"PENDING"}
    names=list(summary);summary["diagnosis"]=analysis
    fig1,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for j,name in enumerate(names):
        r=summary[name]
        if r["eligible"]:
            y=r["rate"];lo,hi=r["wilson95"];axes[0].bar(j,y,color="#2463A6",width=.6)
            axes[0].errorbar(j,y,yerr=[[y-lo],[hi-y]],fmt="none",ecolor="#172B4D",capsize=4)
            axes[0].text(j,min(1.08,hi+.04),f'{r["success"]}/{r["eligible"]}',ha="center",fontsize=10)
    axes[0].set(ylim=(0,1.15),ylabel="Persistent successes / ALL eligible units",title="Conditional real-model repair")
    axes[0].set_xticks(range(len(names)),["Dev","Cal","Cal extra","Test"][:len(names)])
    axes[0].text(.02,.02,"Entity Wilson 95% intervals; one model\nCurator labels; exploratory",transform=axes[0].transAxes,fontsize=8,color="#333",bbox={"facecolor":"white","alpha":.92,"edgecolor":"none","pad":3})
    eligible=[r for r in rows("results/v2-test") if r.get("eligible")];curves={}
    if eligible:
        y=np.array([r["constrained"]["success"] for r in eligible],float);n=len(y)
        scores={"Confidence":np.array([r["diagnosis"]["features"][0] for r in eligible]),"Candidate change":np.array([r["diagnosis"]["candidate"]!="ABCD"[int(np.argmax(r["diagnosis"]["direct"][0]))] for r in eligible],float),"Support margin":np.array([r["diagnosis"]["features"][3] for r in eligible])}
        if Path("results/diagnosticator.joblib").exists() and analysis.get("status")=="COMPLETE":
            import joblib
            from krmm.features import vector
            a=joblib.load("results/diagnosticator.joblib");p=a["model"].predict_proba(np.array([vector(r["diagnosis"]) for r in eligible]))[:,1]
            if a.get("calibrator") is not None:
                logits=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))[:,None];p=a["calibrator"].predict_proba(logits)[:,1]
            scores={"Full diagnostic":p,**scores}
        for (label,score),color in zip(scores.items(),["#2463A6","#888888","#D97706","#21856D"]):
            order=np.argsort(-score,kind="stable");x=np.arange(n+1)/n;sy=np.r_[0,np.cumsum(y[order])]/n
            axes[1].plot(x,sy,label=label,color=color,lw=2,marker=".",ms=3);curves[label]={"coverage":x.tolist(),"success_over_all_eligible":sy.tolist()}
        axes[1].legend(loc="upper left",fontsize=8)
    axes[1].plot([0,1],[0,1],"k:",alpha=.3,lw=1)
    axes[1].set(xlim=(0,1),ylim=(0,1),xlabel="Accepted fraction of eligible test units",ylabel="Successful repairs / ALL eligible test units",title="Frozen-test equal-coverage comparison")
    for ax in axes:ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    fig1.savefig(fig/"real-pilot.png",dpi=180);fig1.savefig(fig/"real-pilot.svg");plt.close(fig1)
    synth=json.loads(Path("results/synthetic-v2/summary.json").read_text());keys=["joint","no_image","no_audio","swapped_audio","sham_background"]
    fig2,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    for m in synth["metadata"]:ax.plot(["Full","No image","No audio","Swap audio","Sham"],[m["media_interventions"][k] for k in keys],marker="o",label=f'world seed {m["seed"]}',alpha=.85)
    ax.set(ylim=(0,1.08),ylabel="Correct reconstruction, complementary subset",title="Generated-media interventions — controlled worlds only");ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2);ax.spines[["top","right"]].set_visible(False)
    fig2.savefig(fig/"synthetic-interventions.png",dpi=180);fig2.savefig(fig/"synthetic-interventions.svg");plt.close(fig2)
    control_paths={"suppression":"results/v2-suppression/metrics.json","auxiliary":"results/auxiliary-v2/summary.json","twins":"results/twins-v2/summary.json"}
    controls={name:json.loads(Path(path).read_text()) if Path(path).exists() else {"status":"PENDING"} for name,path in control_paths.items()}
    summary["exploratory_controls"]=controls
    summary["selective_curves"]=curves;summary["generated_at_sg"]=datetime.datetime.now(ZoneInfo("Asia/Singapore")).isoformat()
    Path("reports/summary.json").write_text(json.dumps(summary,indent=2))
    table=["| Cohort | Screened | Mastered | Eligible | Persistent success | Oracle | Exclusions |","|---|---:|---:|---:|---:|---:|---|"]
    for name in names:
        r=summary[name];table.append(f'| {name} | {r["n"]} | {r["mastered"]} | {r["eligible"]} | {r["success"]}/{r["eligible"]} | {r["oracle_success"]}/{r["eligible"]} | {r["exclusions"]} |')
    runrecords=[json.loads(p.read_text()) for p in Path(".autoresearch/runs").glob("*/run.json")]
    records=["| Launch | Hypothesis | Label | Status | Seconds | Source SHA |","|---|---|---|---|---:|---|"]
    for i,r in enumerate(sorted(runrecords,key=lambda r:r["started_at"]),1):records.append(f'| {i} | {r["hypothesis"]} | {r["label"]} | {r["status"]} | {r.get("duration_seconds","")} | {r.get("git",{}).get("head","")} |')
    persistence=json.loads(Path("results/persistence-all.json").read_text()) if Path("results/persistence-all.json").exists() else {"status":"PENDING"}
    psummary={k:v for k,v in persistence.items() if k not in ["rows","resources"]}
    decision="PIVOT / exploratory prototype; no G1-G3 confirmation"
    primary=analysis.get("paired_diagnosis_vs_confidence");strongest=analysis.get("paired_vs_all_selectors",{}).get("strongest_single")
    text=f"""# KRMM evidence pack

Generated {summary["generated_at_sg"]}. Decision: **{decision}**.

The answer-free repair prototype and controlled evaluation are implemented. No novel diagnostic, real audiovisual complementarity or paper-ready generalization has been established. All trials and failures remain in [EXPERIMENTS.md](EXPERIMENTS.md).

## Real knowledge pilot

Closed-choice acoustic-source task, one Qwen2.5-Omni-3B release, rank-8 last-four-layer updates, 48 damage / 32 repair steps, retention KL 50 and 12-prompt minibatch. Rejected and failed eligible units stay in the primary denominator.

{chr(10).join(table)}

![Real pilot](reports/figures/real-pilot.png)

Eligibility uses independent screen templates 4/5; diagnosis 0/1/6; updates 0/1; hidden evaluation 2/3. No target answer is supplied to repair. Calibration coverage gates failed and remain negative results; these conditional results do not establish broad population validity or G1/G2 evidence.

## Prediction and selection

Analysis status: **{analysis.get("status")}**. Eligible development/calibration/test counts: {analysis.get("dev")}/{analysis.get("calibration")}/{analysis.get("test")}.

Full diagnostic versus confidence, paired entity interval: {json.dumps(primary)}.

Full diagnostic versus strongest calibration-selected single feature: {json.dumps(strongest)}.

[diagnosis.json](results/diagnosis.json) contains AUROC/AUPRC/Brier, exact coverages, all baselines and paired uncertainty. Equal-coverage selectors use identical candidates and writer outcomes, including counterfactual attempts for rejected units. An interval crossing zero does not establish improvement. The bootstrap resamples fixed per-entity selection contributions, without re-ranking each resampled batch; it is descriptive fixed-batch uncertainty. Candidate-generation gain alone is not selector gain.

## Persistence

Independent-process audit: {json.dumps(psummary)}.

A new process loads official unchanged components, complete merged damaged selected weights and each saved repair adapter. Correct and incorrect repairs can both persist. This tests functional persistence, not original parameter/representation recovery.

## Controlled evidence

![Controlled interventions](reports/figures/synthetic-interventions.png)

Generated image/audio experiments demonstrate evidence dependence in known-rule synthetic worlds. Support-only selection matches the learned selector. Real I/O has four images and one audio source, with 5/6 correct conditions; it cannot establish real multimodal repair or sound-specific benefit.

Opposing-carrier twins, auxiliary/sham/text conditions and alternate suppression damage are separately scoped exploratory controls, not automatic G2 promotion. Their complete summaries are preserved in [reports/summary.json](reports/summary.json).

## Cost and provenance

Recorded launches: {len(runrecords)}. Recorded aggregate child wall time: {sum(r.get("duration_seconds",0) for r in runrecords):.1f} seconds, excluding initial downloads. Authorized whole batch: eight wall-clock hours, 20 child launches, 80GB extra storage.

Per-unit files include decoder calls/tokens, queries, peak allocated VRAM, diagnosis/training times, source SHA, configs and raw predictions. Early outer-hook counts undercounted PEFT forwards and are marked in the journal; v2 uses decoder hooks. Some serialization/verification overhead is not isolated per deployed method; no exact throughput superiority is claimed.

Pinned model commit: f75b40e3da2003cdd6e1829b1f420ca70797c34e. Source/config integrity is checked around the frozen test. A source Git bundle resolves cloud commit SHAs.

{chr(10).join(records)}

## Reproduction and limits

Reproduce on the cloud server with exact recorded source, model and environment in a newly authorized resource batch. Use new output directories. Never overwrite trials, bypass locks, reset gates or tune on the test. No local checkout or environment was created.

Limitations: small curator-labeled pilot requiring expert label review; exciter/resonator ambiguity; substantial screening exclusions; one real model and one damage seed; conditional entity uncertainty; limited real audio; logical audited information separation rather than OS isolation. See [BENCHMARK_AUDIT.md](docs/BENCHMARK_AUDIT.md).

The service abstains without a calibrated model, offers an explicitly uncalibrated structural baseline, reloads attempted adapters, rolls back retained-behavior violations and never certifies recovery without an independent target evaluator.

Novelty/publication status remains **PENDING**. Human scientific review is required.
"""
    Path("EVIDENCE_PACK.md").write_text(text)
    print(json.dumps({"cohorts":{k:summary[k] for k in names},"analysis_status":analysis.get("status"),"launches":len(runrecords),"decision":decision},indent=2))
if __name__=="__main__":main()
