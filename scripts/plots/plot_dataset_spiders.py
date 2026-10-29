#!/usr/bin/env python3
"""Plot held-out binary F1 and accuracy across saved dataset experiments."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULTS=['deepset-smoke','wildguard-smoke','toxicchat-smoke','toxicchat-jailbreaking-smoke','xstest-smoke','moderation-smoke']
METHODS={'train_threshold':'Calibrated (train)'}
LABELS={'deepset':'Deepset','wildguard':'WildGuard','toxicchat':'ToxicChat\ntoxicity',
        'toxicchat_jailbreaking':'ToxicChat\njailbreaking','xstest':'XSTest','moderation':'Moderation','hatexplain':'HateXplain'}


def load_results(experiments):
    """Read latest saved comparisons, keeping only training-fitted thresholds.

    Args:
        experiments: Dataset suite root paths, each with suite_config.json and at
            least one comparison-*/all_results.csv. Latest timestamp wins.

    Returns:
        (long-form DataFrame, provenance dictionary). Dataset names come from saved
        configs; duplicate datasets, missing series, and undefined metrics raise
        errors rather than being silently zero-filled. Both prompt variants stay
        separate. F1 is the saved positive-class binary F1, not macro-F1.
    """
    import pandas as pd
    tables=[];sources={}
    for folder in experiments:
        folder=Path(folder)
        config=json.loads((folder/'suite_config.json').read_text())
        dataset=config.get('dataset',{}).get('name') or config['experiment']['split_profile']
        if dataset in sources:raise ValueError(f'Duplicate dataset: {dataset}')
        candidates=sorted(folder.glob('comparison-*/all_results.csv'))
        if not candidates:raise ValueError(f'No comparison found: {folder}')
        source=candidates[-1];table=pd.read_csv(source)
        table=table[table['method'].isin(METHODS)].copy()
        table.insert(0,'dataset',dataset)
        if table.empty:raise ValueError(f'No calibrated results: {source}')
        if table.duplicated(['model_name','prompt_variant','method']).any():raise ValueError(f'Duplicate series: {source}')
        for column in ('f1','accuracy'):
            if not table[column].between(0,1).all():raise ValueError(f'Missing/invalid {column} in {source}')
        table['series']=table['model_name']+' / '+table['prompt_variant']+' / '+table['method'].map(METHODS)
        tables.append(table);sources[dataset]=str(source.resolve())
    if len(tables)<3:raise ValueError('Radar plots require at least three datasets')
    expected=set(tables[0]['series'])
    for table in tables:
        if set(table['series'])!=expected:raise ValueError(f'Model/prompt/method coverage differs for {table.iloc[0]["dataset"]}')
    return pd.concat(tables,ignore_index=True),{'sources':sources,'decision_methods':list(METHODS),
        'f1_definition':'Binary F1 of the dataset positive class; not macro-F1',
        'calibration':'Accuracy-maximizing threshold fitted on training only',
        'note':'Saved held-out results; sample size and coverage differ by dataset. No pooled metric.'}


def plot_spiders(results, output):
    """Write two radar figures plus exact input metrics to output.

    Args:
        results: Long-form table from load_results, with dataset, model_name,
            prompt_variant, method, f1 and accuracy. Every series must cover every
            dataset. Color identifies model/prompt; every line uses a training-calibrated
            threshold. Native and fixed-0.5 results are excluded.
        output: New output directory path, never overwrites an existing directory.

    Returns:
        Dictionary mapping f1/accuracy to PNG file paths. SVG copies and plotted
        CSVs are also saved. Dataset axes use a common fixed 0-to-1 scale.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    results.to_csv(output/'plotted_results.csv',index=False)
    datasets=list(dict.fromkeys(results['dataset']))
    pairs=list(dict.fromkeys(zip(results['model_name'],results['prompt_variant'])))
    colors={pair:plt.get_cmap('tab10')(i) for i,pair in enumerate(pairs)}
    angles=np.linspace(0,2*np.pi,len(datasets),endpoint=False);closed=np.r_[angles,angles[0]]
    files={}
    sizes=results.groupby('dataset',sort=False)['common_test_examples'].first().to_dict()
    for metric,title in [('f1','Binary F1'),('accuracy','Accuracy')]:
        fig,ax=plt.subplots(figsize=(12,9),subplot_kw={'projection':'polar'})
        fig.subplots_adjust(left=.10,right=.66,top=.85,bottom=.22)
        ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1)
        ax.set_xticks(angles,[f'{LABELS.get(d,d)}\n(n={int(sizes[d])})' for d in datasets],fontsize=10)
        ax.tick_params(axis='x',pad=18)
        ax.set_ylim(0,1);ax.set_yticks([.2,.4,.6,.8,1.0]);ax.set_yticklabels(['.2','.4','.6','.8','1.0'],color='#555555',fontsize=9)
        ax.set_rlabel_position(15);ax.grid(alpha=.3)
        for model,prompt in pairs:
            for method,label in METHODS.items():
                selection=results[(results['model_name']==model)&(results['prompt_variant']==prompt)&(results['method']==method)]
                values=selection.set_index('dataset').loc[datasets,metric].to_numpy()
                ax.plot(closed,np.r_[values,values[0]],color=colors[(model,prompt)],
                    linestyle='-',marker='o',
                    linewidth=1.8,markersize=4,alpha=.85,label=f'{model} · {prompt}\n{label}')
        fig.suptitle(f'{title} across datasets',fontsize=17,fontweight='bold',y=.96)
        ax.legend(loc='center left',bbox_to_anchor=(1.18,.5),frameon=False,fontsize=9,labelspacing=1.2)
        fig.text(.10,.035,'Held-out scores using thresholds fitted on training data.\nEach axis is a separate task; small smoke samples are not benchmark estimates.',fontsize=9,color='#555555')
        for extension in ('png','svg'):
            fig.savefig(output/f'{metric}_spider.{extension}',dpi=180,bbox_inches='tight')
        plt.close(fig);files[metric]=output/f'{metric}_spider.png'
        results.pivot(index='series',columns='dataset',values=metric).reindex(columns=datasets).to_csv(output/f'{metric}_values.csv')
    return files


def accuracy_ranking(results):
    """Rank model/prompt variants by equal-dataset mean calibrated accuracy.

    Args:
        results: Complete long-form results from load_results. Only train_threshold
            rows contribute; every model/prompt must cover the same datasets.

    Returns:
        DataFrame sorted by mean accuracy descending, with score = 100 * mean
        accuracy and competition ranks (ties share rank). Dataset sample sizes
        do not affect weights. Sample SD uses ddof=1 across datasets (not a confidence interval).
        Pairwise wins compare every opponent on every dataset; ties use 1e-12
        tolerance. Outright/shared dataset wins count first-place finishes.
        This is a descriptive score, not significance.
    """
    import pandas as pd
    data=results[results['method']=='train_threshold']
    groups=list(data.groupby(['model_name','prompt_variant'],sort=False))
    expected=set(data['dataset'])
    rows=[]
    for (model,prompt),group in groups:
        if set(group['dataset'])!=expected or group['dataset'].duplicated().any():
            raise ValueError('Ranking requires one accuracy per variant per dataset')
        if not group['accuracy'].between(0,1).all():raise ValueError('Invalid ranking accuracy')
        rows.append({'model':model,'prompt':prompt,'datasets':len(group),
                     'mean_accuracy':group['accuracy'].mean(),
                     'std_accuracy':group['accuracy'].std(ddof=1),
                     'minimum_accuracy':group['accuracy'].min(),
                     'maximum_accuracy':group['accuracy'].max()})
    if not rows:raise ValueError('No calibrated accuracy results to rank')
    ranking=pd.DataFrame(rows).sort_values(['mean_accuracy','model','prompt'],ascending=[False,True,True]).reset_index(drop=True)
    ranking['score']=100*ranking['mean_accuracy']
    ranking['rank']=ranking['mean_accuracy'].rank(method='min',ascending=False).astype(int)
    if len(expected)<2:raise ValueError('Standard deviation requires at least two datasets')
    pivot=data.pivot(index='dataset',columns=['model_name','prompt_variant'],values='accuracy')
    import numpy as np
    for index,row in ranking.iterrows():
        key=(row['model'],row['prompt']);values=pivot[key]
        opponents=pivot.drop(columns=[key])
        differences=values.to_numpy()[:,None]-opponents.to_numpy()
        ties=np.isclose(differences,0,rtol=0,atol=1e-12)
        ranking.loc[index,'pairwise_wins']=int(((differences>0)&~ties).sum())
        ranking.loc[index,'pairwise_ties']=int(ties.sum())
        ranking.loc[index,'pairwise_losses']=int(((differences<0)&~ties).sum())
        best=np.isclose(pivot.to_numpy(),pivot.max(axis=1).to_numpy()[:,None],rtol=0,atol=1e-12)
        is_best=np.isclose(values,pivot.max(axis=1),rtol=0,atol=1e-12)
        ranking.loc[index,'outright_dataset_wins']=int((is_best&(best.sum(axis=1)==1)).sum())
        ranking.loc[index,'shared_dataset_wins']=int((is_best&(best.sum(axis=1)>1)).sum())
    count_columns=['pairwise_wins','pairwise_ties','pairwise_losses','outright_dataset_wins','shared_dataset_wins']
    ranking[count_columns]=ranking[count_columns].astype(int)
    ranking['std_score']=100*ranking['std_accuracy']
    return ranking


def plot_accuracy_ranking(results, output):
    """Save ranked mean-accuracy bars, SVG, and CSV in existing output directory.

    Args:
        results: Long-form results, as accepted by accuracy_ranking.
        output: Existing directory containing the radar figures.

    Returns:
        PNG path. Score is equal-weight mean dataset accuracy, on a 0–100 scale.
        Colors follow the model/prompt order of the radar charts.
    """
    import matplotlib.pyplot as plt
    ranking=accuracy_ranking(results)
    output=Path(output);ranking.to_csv(output/'accuracy_ranking.csv',index=False)
    pairs=list(dict.fromkeys(zip(results['model_name'],results['prompt_variant'])))
    colors={pair:plt.get_cmap('tab10')(i) for i,pair in enumerate(pairs)}
    fig,(ax,stats)=plt.subplots(1,2,figsize=(15,6),gridspec_kw={'width_ratios':[1.6,1]},layout='constrained')
    labels=[f"#{row['rank']}  {row['model']} · {row['prompt']}" for _,row in ranking.iterrows()]
    positions=list(range(len(ranking)))
    ax.barh(positions,ranking['score'],xerr=ranking['std_score'],capsize=5,
        error_kw={'elinewidth':1.5,'ecolor':'#303030'},
        color=[colors[(r['model'],r['prompt'])] for _,r in ranking.iterrows()],height=.52)
    ax.set_yticks(positions,labels);ax.invert_yaxis()
    ax.set_xlim(min(0,float((ranking['score']-ranking['std_score']).min())-3),max(100,float((ranking['score']+ranking['std_score']).max())+3))
    ax.set_xlabel('Mean accuracy (%) ± sample SD across datasets')
    ax.set_title('Accuracy ranking · calibrated thresholds',fontweight='bold',pad=20)
    ax.xaxis.grid(alpha=.2);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    stats.axis('off')
    cells=[[f"{r['score']:.2f} ± {r['std_score']:.2f}",
            f"{r['pairwise_wins']} / {r['pairwise_ties']} / {r['pairwise_losses']}",
            f"{r['outright_dataset_wins']} / {r['shared_dataset_wins']}"] for _,r in ranking.iterrows()]
    table=stats.table(cellText=cells,colLabels=['Mean ± SD (%)','Pairwise\nW / T / L','Dataset best\nsole / shared'],
        rowLabels=[f"#{r['rank']}" for _,r in ranking.iterrows()],cellLoc='center',loc='center')
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2.1)
    stats.set_title('Wins compare all model/prompt variants',fontsize=11,pad=20)
    n=int(ranking['datasets'].iloc[0]);opponents=len(ranking)-1
    fig.suptitle(f'Equal weight for {n} datasets · {n*opponents} pairwise comparisons per variant',fontsize=12)
    fig.supxlabel('SD describes variation across tasks, not uncertainty in the mean. Shared first places are counted separately; descriptive held-out results.',fontsize=10,color='#555555')
    for ext in ('png','svg'):
        fig.savefig(output/f'accuracy_ranking.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    return output/'accuracy_ranking.png'


def dataset_win_table(results):
    """Return accuracy grid, winner flags, and sole/shared dataset win totals.

    Args:
        results: Complete calibrated results from load_results. Every model/prompt
            must cover the same datasets. A tie within 1e-12 counts as a shared
            first place for each tied variant, never as a sole win.

    Returns:
        (accuracy, winners, totals) DataFrames with one column per model/prompt.
        A dataset awards sole wins only when exactly one variant is best.
    """
    import numpy as np
    # Reuse validation of metric bounds, duplicates, and complete coverage.
    accuracy_ranking(results)
    data=results[results['method']=='train_threshold'].copy()
    data['variant']=data['model_name']+' / '+data['prompt_variant']
    datasets=list(dict.fromkeys(data['dataset']))
    variants=list(dict.fromkeys(data['variant']))
    accuracy=data.pivot(index='dataset',columns='variant',values='accuracy').loc[datasets,variants]
    import pandas as pd
    winners=pd.DataFrame(np.isclose(accuracy.to_numpy(),accuracy.max(axis=1).to_numpy()[:,None],rtol=0,atol=1e-12),index=accuracy.index,columns=accuracy.columns)
    sole=winners.sum(axis=1)==1
    import pandas as pd
    totals=pd.DataFrame({'sole_wins':winners.loc[sole].sum(),
                         'shared_wins':winners.loc[~sole].sum(),
                         'all_wins':winners.sum()}).astype(int)
    return accuracy,winners,totals


def plot_dataset_wins(results, output):
    """Save dataset-by-variant accuracy heatmap and dataset win totals.

    Args:
        results: Long-form saved calibrated results from load_results.
        output: Existing plot directory; writes PNG/SVG and supporting CSVs.

    Returns:
        PNG path. Bold outlined cells identify first place, including ties.
        Final rows separately count sole wins and shared wins.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    accuracy,winners,totals=dataset_win_table(results)
    output=Path(output)
    accuracy.to_csv(output/'dataset_accuracy.csv')
    winners.to_csv(output/'dataset_winners.csv')
    totals.to_csv(output/'dataset_win_totals.csv')
    n,m=accuracy.shape
    fig,ax=plt.subplots(figsize=(max(9,m*2.3),n*.7+3.2),layout='constrained')
    image=ax.imshow(accuracy.to_numpy(),cmap='Blues',vmin=0,vmax=1,aspect='auto')
    for row in range(n):
        for col in range(m):
            value=accuracy.iloc[row,col];best=winners.iloc[row,col]
            text=f'{value*100:.1f}%'+(' ★' if best else '')
            ax.text(col,row,text,ha='center',va='center',fontsize=11,
                    color='white' if value>.65 else '#18212b',fontweight='bold' if best else 'normal')
            if best:ax.add_patch(Rectangle((col-.46,row-.43),.92,.86,fill=False,edgecolor='#e5a400',linewidth=2.5))
    for k,(column,label) in enumerate([('sole_wins','Sole wins'),('shared_wins','Shared wins'),('all_wins','All dataset wins')]):
        row=n+k
        for col,variant in enumerate(accuracy.columns):
            ax.add_patch(Rectangle((col-.5,row-.5),1,1,facecolor='#eef2f6' if k<2 else '#dfe9e1',edgecolor='white'))
            ax.text(col,row,str(totals.loc[variant,column]),ha='center',va='center',fontsize=12,fontweight='bold')
    ax.set_ylim(n+2.5,-.5);ax.set_xlim(-.5,m-.5)
    ax.set_xticks(range(m),[v.replace(' / ','\n') for v in accuracy.columns],fontsize=11)
    ax.xaxis.tick_top();ax.tick_params(axis='x',length=0,pad=12)
    ax.set_yticks(range(n+3),[LABELS.get(d,d).replace('\n',' ') for d in accuracy.index]+['Sole wins','Shared wins','All dataset wins'],fontsize=11)
    ax.tick_params(axis='y',length=0,pad=10)
    ax.set_xticks(np.arange(-.5,m,1),minor=True);ax.set_yticks(np.arange(-.5,n,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    colorbar=fig.colorbar(image,ax=ax,shrink=.6,pad=.025);colorbar.set_label('Held-out accuracy')
    fig.suptitle('Dataset wins · training-calibrated thresholds',fontsize=16,fontweight='bold')
    fig.supxlabel('★ = best accuracy for that dataset (ties included). Sole wins have one winner; shared wins have multiple.\nAll wins = sole + shared. Counts describe first-place finishes, not statistical significance.',fontsize=9,color='#555555')
    for ext in ('png','svg'):fig.savefig(output/f'dataset_wins.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    return output/'dataset_wins.png'


def main(argv=None):
    """Parse argv (None reads sys.argv), load saved suites, and generate figures."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiments',nargs='*',type=Path,help='Dataset suite folders; defaults to the six smoke folders')
    parser.add_argument('--output',type=Path,help='New destination directory')
    parser.add_argument('--suite-root',type=Path,help='Root of an all-datasets run; reads its six dataset subfolders')
    args=parser.parse_args(argv)
    if args.suite_root and args.experiments:parser.error('Use --suite-root or explicit experiment folders, not both')
    experiments=([args.suite_root/name for name in ('deepset','wildguard','toxicchat','toxicchat_jailbreaking','xstest','moderation')]
                 if args.suite_root else args.experiments or [ROOT/'runs'/name for name in DEFAULTS])
    output=args.output or ROOT/'runs'/('dataset-spiders-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    table,provenance=load_results(experiments)
    files=plot_spiders(table,output)
    files['accuracy_ranking']=plot_accuracy_ranking(table,output)
    files['dataset_wins']=plot_dataset_wins(table,output)
    (output/'sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for name,path in files.items():print(name,path.resolve())


if __name__=='__main__':main()
