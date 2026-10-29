#!/usr/bin/env python3
"""Aggregate calibrated guardrail results across datasets, with paired uncertainty."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
if __package__:
    from .plot_dataset_spiders import ROOT, load_results
else:
    from plot_dataset_spiders import ROOT, load_results

DATASETS=['deepset','wildguard','toxicchat','toxicchat_jailbreaking','xstest','moderation']
COLORS=['#2474a6','#62b4c8','#d97232','#edb461']


def bootstrap_prompt_gains(root, sources, *, repeats=3000, seed=42):
    """Paired label-stratified bootstrap of equal-dataset optimization gains.

    Args:
        root: Suite root containing each dataset's saved predictions/thresholds.
        sources: Dataset-to-comparison-CSV mapping from load_results.
        repeats: Number of bootstrap replicates (default 3000).
        seed: Fixed NumPy sampling seed for reproducible intervals.

    Returns:
        Mean gains and percentile 95% intervals per model and metric. Each
        dataset retains its sample size and class counts; paired predictions use
        identical resampled rows. Dataset tasks and trained prompts are fixed,
        so intervals exclude task-selection and prompt-search uncertainty.
    """
    rng=np.random.default_rng(seed);draws={};actual={}
    for dataset,source in sources.items():
        report=json.loads((Path(source).parent/'comparison.json').read_text())
        indices=report['common_test_row_indices']
        tables={};identity=None
        folders=Path(root)/dataset
        for model in ('open-jev-9b','open-jev-2b'):
            for variant in ('baseline','optimized'):
                directory=folders/model/variant
                rows=[json.loads(line) for line in (directory/'test/predictions.jsonl').read_text().splitlines()]
                rows=[rows[i] for i in indices]
                keys=[(r['text_sha256'],r['label']) for r in rows]
                if identity is not None and keys!=identity:raise ValueError('Paired test identities differ')
                identity=keys
                if any(r.get('error') is not None for r in rows):raise ValueError('Invalid common successful rows')
                threshold=json.loads((directory/'train/frozen_threshold.json').read_text())['threshold']
                tables[(model,variant)]=np.array([r['p_positive']>=threshold for r in rows],dtype=int)
        labels=np.array([label for _,label in identity]);positive=np.flatnonzero(labels==1);negative=np.flatnonzero(labels==0)
        if not len(positive) or not len(negative):raise ValueError('Both labels required for stratified bootstrap')
        sampling=np.concatenate([rng.choice(positive,(repeats,len(positive))),rng.choice(negative,(repeats,len(negative)))],axis=1)
        truth=labels[sampling]
        def metrics(predicted, reference):
            tp=((predicted==1)&(reference==1)).sum(axis=-1)
            fp=((predicted==1)&(reference==0)).sum(axis=-1)
            fn=((predicted==0)&(reference==1)).sum(axis=-1)
            return {'accuracy':(predicted==reference).mean(axis=-1),'f1':2*tp/(2*tp+fp+fn)}
        for model in ('open-jev-9b','open-jev-2b'):
            before=metrics(tables[(model,'baseline')][sampling],truth)
            after=metrics(tables[(model,'optimized')][sampling],truth)
            observed_before=metrics(tables[(model,'baseline')],labels)
            observed_after=metrics(tables[(model,'optimized')],labels)
            for metric in ('accuracy','f1'):
                key=(model,metric);draws.setdefault(key,[]).append(after[metric]-before[metric])
                actual.setdefault(key,[]).append(observed_after[metric]-observed_before[metric])
    rows=[]
    for (model,metric),samples in draws.items():
        distribution=np.mean(samples,axis=0)*100;low,high=np.quantile(distribution,[.025,.975])
        rows.append({'model':model,'metric':metric,'gain_pp':np.mean(actual[(model,metric)])*100,'low_pp':low,'high_pp':high})
    return pd.DataFrame(rows)


def create_figures(root, output):
    """Save aggregate score, error tradeoff, and paired prompt-gain dashboards.

    Args:
        root: Path to the full realistic-datasets suite root.
        output: New output directory. All metrics give each dataset equal weight.
            Only train_threshold rows contribute; native/fixed 0.5 are excluded.
    """
    root=Path(root);output=Path(output)
    results,provenance=load_results([root/d for d in DATASETS])
    output.mkdir(parents=True,exist_ok=False)
    results['false_positive_rate']=results['fp']/(results['fp']+results['tn'])
    results['false_negative_rate']=results['fn']/(results['fn']+results['tp'])
    variants=list(dict.fromkeys(zip(results['model_name'],results['prompt_variant'])))
    colors=dict(zip(variants,COLORS));labels={k:f"{k[0].replace('open-jev-','').upper()} · {k[1]}" for k in variants}
    metrics=['accuracy','f1','precision','recall','false_positive_rate','false_negative_rate']
    # Use an explicit zero-division policy so every task keeps equal weight.
    no_positive_predictions=(results['tp']+results['fp'])==0
    results.loc[no_positive_predictions & results['precision'].isna(),'precision']=0.0
    if results[metrics].isna().any().any():raise ValueError('Other undefined metrics found')
    grouped=results.groupby(['model_name','prompt_variant'],sort=False)[metrics]
    mean=grouped.mean();sd=grouped.std(ddof=1)
    summary=mean.add_suffix('_mean').join(sd.add_suffix('_sd'))
    summary.to_csv(output/'aggregate_metrics.csv');results.to_csv(output/'dataset_metrics.csv',index=False)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    legend=[Patch(facecolor=colors[k],label=labels[k]) for k in variants]
    fig,axes=plt.subplots(1,2,figsize=(14,6.5))
    fig.subplots_adjust(top=.80,bottom=.23,wspace=.25)
    for ax,chosen,title in zip(axes,[['accuracy','f1'],['precision','recall']],['Overall quality','Blocking quality']):
        positions=np.arange(2);width=.18
        for i,key in enumerate(variants):
            values=mean.loc[key,chosen].to_numpy()*100
            ax.bar(positions+(i-1.5)*width,values,width,color=colors[key],yerr=sd.loc[key,chosen].to_numpy()*100,capsize=3,error_kw={'alpha':.6},label=labels[key])
        ax.set_xticks(positions,[m.capitalize() if m!='f1' else 'Binary F1' for m in chosen]);ax.set_ylim(0,max(115,float(((mean[chosen]+sd[chosen])*100).max().max())+5))
        ax.set_yticks([0,20,40,60,80,100]);ax.set_ylabel('Equal-dataset mean (%) ± sample SD');ax.set_title(title,fontweight='bold')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Guardrail performance across six datasets',fontsize=19,fontweight='bold')
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.08),ncol=4,frameon=False)
    fig.text(.5,.035,'Calibrated thresholds only · SD = cross-task variation · Precision = 0 when no positive predictions.',ha='center',fontsize=10)
    save(fig,output,'aggregate_quality')

    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for ax,metric,title in zip(axes,['false_positive_rate','false_negative_rate'],['Benign inputs blocked','Harmful / positive inputs missed']):
        for index,key in enumerate(variants):
            value=100*mean.loc[key,metric];error=100*sd.loc[key,metric]
            ax.barh(index,value,xerr=error,capsize=4,color=colors[key],height=.55)
            ax.text(max(value+error+1,1),index,f'{value:.1f}%',va='center',fontsize=11)
        ax.set_yticks(range(len(variants)),[labels[k] for k in variants]);ax.invert_yaxis()
        ax.set_xlim(0,max(65,float(((mean[metric]+sd[metric])*100).max())+12))
        ax.set_xlabel('Mean rate (%) ± SD across datasets · lower is better');ax.set_title(title,fontweight='bold')
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('What the aggregate errors mean',fontsize=19,fontweight='bold')
    fig.supxlabel('Each task contributes equally. Positive labels differ by task: injection, harm, toxicity, jailbreak, or moderation.',fontsize=10)
    save(fig,output,'aggregate_errors')

    gains=bootstrap_prompt_gains(root,provenance['sources'])
    for _,row in gains.iterrows():
        expected=100*(mean.loc[(row['model'],'optimized'),row['metric']]-mean.loc[(row['model'],'baseline'),row['metric']])
        if not np.isclose(row['gain_pp'],expected,atol=1e-10):raise ValueError('Saved comparison metrics disagree with paired predictions')
    gains.to_csv(output/'paired_optimization_gains.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,metric,title in zip(axes,['accuracy','f1'],['Accuracy improvement','Binary F1 improvement']):
        for index,model in enumerate(['open-jev-9b','open-jev-2b']):
            row=gains[(gains['model']==model)&(gains['metric']==metric)].iloc[0]
            ax.plot([row.low_pp,row.high_pp],[index,index],color=colors[(model,'optimized')],linewidth=4)
            ax.plot(row.gain_pp,index,'o',color=colors[(model,'optimized')],markersize=10)
            ax.annotate(f'{row.gain_pp:+.1f} pp',xy=(row.gain_pp,index),xytext=(0,14),textcoords='offset points',ha='center',fontweight='bold')
        ax.axvline(0,color='#777777',linestyle='--',linewidth=1)
        ax.set_yticks([0,1],['9B','2B']);ax.set_ylim(-.6,1.6);ax.invert_yaxis();ax.set_title(title,fontweight='bold')
        ax.set_xlabel('Optimized − baseline (percentage points)');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Does prompt optimization help overall?',fontsize=18,fontweight='bold')
    fig.supxlabel('Equal-dataset mean gain · 95% paired, label-stratified bootstrap interval (3,000 replicates).\nConditional on these six tasks, their label counts, and frozen prompts; excludes prompt-search uncertainty.',fontsize=10)
    save(fig,output,'aggregate_prompt_gains')
    counts=results.groupby('dataset',sort=False).first()[['common_test_examples','tp','fn']]
    counts['positive_examples']=counts['tp']+counts['fn'];counts[['common_test_examples','positive_examples']].to_csv(output/'sample_counts.csv')
    provenance.update({'aggregation':'Unweighted mean across six datasets; no pooling of examples',
        'standard_deviation':'Sample SD across datasets (ddof=1)',
        'precision_policy':'Precision=0 when no positive predictions; avoids silently dropping datasets',
        'bootstrap':'3000 paired stratified replicates per dataset, seed 42; equal-dataset mean',
        'suite_root':str(root.resolve())})
    (output/'sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(output.resolve())


def save(fig, output, name):
    """Save fig to output/name as PNG and SVG, then release Matplotlib resources."""
    for extension in ('png','svg'):fig.savefig(Path(output)/f'{name}.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    """Read --suite-root/--output and create aggregate figures without inference."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-root',type=Path,default=ROOT/'runs/realistic-datasets')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=args.output or args.suite_root/('aggregate-figures-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    create_figures(args.suite_root,output)


if __name__=='__main__':main()
