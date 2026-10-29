#!/usr/bin/env python3
"""Dataset-level radar plots comparing retriever baselines and JEV 2B/9B reranking."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SERIES=[('bm25','2b'),('bm25','9b'),('dense','2b'),('dense','9b')]
PLOT_SERIES=[('bm25','baseline'),('dense','baseline'),*SERIES]
METRICS={'ndcg':'nDCG','mrr':'MRR','recall':'Recall','hit_rate':'Hit Rate'}
LABELS={'nfcorpus':'NFCorpus','scifact':'SciFact','arguana':'ArguAna','scidocs':'SCIDOCS',
        'fiqa':'FiQA','cqadupstack':'CQADupstack'}
ORDER=['nfcorpus','scifact','arguana','scidocs','fiqa','cqadupstack']


def load_results(suite, cutoffs=(3,5,10), collection_level=False):
    """Require matched query sets and candidates; macro-average shared CQADupstack forums.

    Incomplete collections are excluded from ALL series rather than zero-filled.
    Reads per-run artifacts, not a potentially stale combined CSV.
    """
    import pandas as pd
    suite=Path(suite).resolve()
    collections=sorted({p.parent.parent.parent.relative_to(suite).as_posix()
                        for p in suite.rglob('plan.json')
                        if p.parent.name in ('2b','9b') and p.parent.parent.name in ('bm25','dense')})
    records=[];excluded={};sources={};identities={}
    for collection in collections:
        runs={};baselines={};problems=[]
        for retriever,model in SERIES:
            folder=suite/collection/retriever/model
            if not (folder/'status.json').exists():
                problems.append(f'{retriever}/{model}: missing status');continue
            status=json.loads((folder/'status.json').read_text())
            if status.get('status')!='complete':
                problems.append(f"{retriever}/{model}: {status.get('error',status.get('status'))}");continue
            plan=json.loads((folder/'plan.json').read_text())
            result=json.loads((folder/'results.json').read_text())
            rows=[r for r in result['summary'] if r['method']==retriever+'+jev']
            if len(rows)!=1:raise ValueError(f'Expected one JEV summary: {folder}')
            row=rows[0]
            if row['queries']!=len(plan['query_ids']):raise ValueError(f'Query count mismatch: {folder}')
            for metric in METRICS:
                for k in cutoffs:
                    value=row.get(f'{metric}@{k}')
                    if not isinstance(value,(float,int)) or not math.isfinite(value) or not 0<=value<=1:
                        raise ValueError(f'Invalid/missing {metric}@{k}: {folder}')
            baseline_rows=[r for r in result['summary'] if r['method']==retriever]
            if len(baseline_rows)!=1:raise ValueError(f'Expected one retriever baseline: {folder}')
            baseline=baseline_rows[0]
            if baseline['queries']!=len(plan['query_ids']):raise ValueError(f'Baseline query count mismatch: {folder}')
            for metric in METRICS:
                for k in cutoffs:
                    key=f'{metric}@{k}';value=baseline.get(key)
                    if not isinstance(value,(float,int)) or not math.isfinite(value) or not 0<=value<=1:
                        raise ValueError(f'Invalid/missing baseline {key}: {folder}')
                    if retriever in baselines and not math.isclose(value,baselines[retriever][key],rel_tol=0,abs_tol=1e-12):
                        raise ValueError(f'Retriever baseline differs across models: {folder}')
            baselines[retriever]=baseline
            runs[(retriever,model)]=(plan,row)
            sources[str(folder.relative_to(suite))]=hashlib.sha256((folder/'results.json').read_bytes()).hexdigest()
        if problems:
            excluded[collection]=problems;continue
        if len({tuple(p['query_ids']) for p,r in runs.values()})!=1:
            raise ValueError(f'Query selection differs across configurations: {collection}')
        for retriever in ('bm25','dense'):
            if runs[(retriever,'2b')][0]['candidates_sha256']!=runs[(retriever,'9b')][0]['candidates_sha256']:
                raise ValueError(f'Candidate rankings differ across models: {collection}/{retriever}')
        # Corpus/query/qrels snapshots and judge prompt must match across comparisons.
        for key in ('source_sha256','question'):
            if len({json.dumps(p[key],sort_keys=True) for p,r in runs.values()})!=1:
                raise ValueError(f'{key} differs: {collection}')
        for (retriever,model),(plan,row) in runs.items():
            identity=json.dumps({'model':plan['model'],'question':plan['question']},sort_keys=True)
            if model in identities and identities[model]!=identity:
                raise ValueError(f'Judge model/prompt changes across datasets: {model}')
            identities[model]=identity
            records.append({'collection':collection,'dataset':collection if collection_level else collection.split('/')[0],
                'retriever':retriever,'model':model,'queries':row['queries'],
                **{f'{metric}@{k}':row[f'{metric}@{k}'] for metric in METRICS for k in cutoffs}})
        for retriever in ('bm25','dense'):
            row=baselines[retriever]
            records.append({'collection':collection,'dataset':collection if collection_level else collection.split('/')[0],
                'retriever':retriever,'model':'baseline','queries':row['queries'],
                **{f'{metric}@{k}':row[f'{metric}@{k}'] for metric in METRICS for k in cutoffs}})
    if not records:raise ValueError('No collections completed by all four configurations')
    raw=pd.DataFrame(records)
    keys=['dataset','retriever','model']
    columns=[f'{metric}@{k}' for metric in METRICS for k in cutoffs]
    grouped=raw.groupby(keys,sort=False)[columns].mean().reset_index()
    counts=raw.groupby(keys,sort=False).agg(collections=('collection','nunique'),queries=('queries','sum')).reset_index()
    grouped=grouped.merge(counts,on=keys,validate='one_to_one')
    axes=sorted(set(grouped['dataset']),key=lambda name:(ORDER.index(name) if name in ORDER else len(ORDER),name))
    if len(axes)<3:raise ValueError('Radar plots require at least three complete dataset axes')
    provenance={'suite':str(suite),'cutoffs':list(cutoffs),'collection_level':collection_level,
        'included_collections':sorted(set(raw['collection'])),'excluded_collections':excluded,
        'aggregation':'Equal-weight collection means within each dataset; same collections for all configurations.',
        'sources_sha256':sources,'note':'Descriptive sampled-query results. No missing values replaced with zero.'}
    return grouped,raw,axes,provenance


def plot_spiders(table, axes, output, cutoffs=(3,5,10), provenance=None):
    """Save four figures, each with one panel per cutoff and a common 0–1 scale."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    table.to_csv(output/'dataset_metrics.csv',index=False)
    angles=np.linspace(0,2*np.pi,len(axes),endpoint=False);closed=np.r_[angles,angles[0]]
    colors={'2b':'#2563EB','9b':'#EA580C','baseline':'#475569'}
    styles={'bm25':'-','dense':'--'}
    sizes=table.groupby('dataset',sort=False).first()
    axis_labels=[]
    for axis in axes:
        row=sizes.loc[axis];n=int(row['queries']);count=int(row['collections'])
        label=LABELS.get(axis,axis.replace('cqadupstack/','CQA: '))
        axis_labels.append(f'{label}\n'+(f'{count} forums · n={n}' if count>1 else f'n={n}'))
    files={}
    for metric,title in METRICS.items():
        fig,axs=plt.subplots(1,len(cutoffs),figsize=(21,8),subplot_kw={'projection':'polar'},squeeze=False)
        fig.subplots_adjust(left=.07,right=.93,top=.80,bottom=.23,wspace=.45)
        for ax,k in zip(axs[0],cutoffs):
            ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1)
            ax.set_xticks(angles,axis_labels,fontsize=10)
            ax.tick_params(axis='x',pad=16)
            ax.set_ylim(0,1);ax.set_yticks([.2,.4,.6,.8,1])
            ax.set_yticklabels(['.2','.4','.6','.8','1.0'],fontsize=8,color='#64748B')
            ax.set_rlabel_position(18);ax.grid(color='#CBD5E1',alpha=.7)
            ax.spines['polar'].set_color('#CBD5E1')
            for retriever,model in PLOT_SERIES:
                selection=table[(table['retriever']==retriever)&(table['model']==model)].set_index('dataset')
                values=selection.loc[axes,f'{metric}@{k}'].to_numpy()
                ax.plot(closed,np.r_[values,values[0]],color=colors[model],linestyle=styles[retriever],
                        marker='o' if retriever=='bm25' else 's',markersize=4,linewidth=2,alpha=.9)
            ax.set_title(f'{title}@{k}',fontsize=15,fontweight='bold',pad=48)
        fig.suptitle(f'{title} across datasets',fontsize=23,fontweight='bold',y=.97)
        handles=[Line2D([0],[0],color=colors[m],linestyle=styles[r],marker='o' if r=='bm25' else 's',
                       linewidth=2,label=(f'{r.upper() if r=="bm25" else "Dense"} only' if m=='baseline' else
                       f'{r.upper() if r=="bm25" else "Dense"} → JEV {m.upper()}')) for r,m in PLOT_SERIES]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.09),ncol=3,frameon=False,fontsize=12)
        notes='Identical sampled queries across configurations; higher is better. Each panel uses a fixed 0–1 scale.'
        if provenance and not provenance['collection_level']:
            notes+='\nCQADupstack is the equal-forum mean over forums completed by all four configurations.'
        fig.text(.5,.025,notes,ha='center',va='bottom',fontsize=10,color='#475569')
        for extension in ('png','svg'):
            fig.savefig(output/f'{metric}_spider.{extension}',dpi=160,facecolor='white')
        plt.close(fig);files[metric]=str(output/f'{metric}_spider.png')
    return files


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-root',type=Path,default=ROOT/'runs/rag-under-60k')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--collection-level',action='store_true',help='Give each CQADupstack forum its own axis')
    args=parser.parse_args(argv)
    table,raw,axes,provenance=load_results(args.suite_root,collection_level=args.collection_level)
    output=args.output or args.suite_root/('spiders-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    files=plot_spiders(table,axes,output,provenance=provenance)
    raw.to_csv(Path(output)/'collection_metrics.csv',index=False)
    (Path(output)/'sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'files':files,'excluded':provenance['excluded_collections']},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
