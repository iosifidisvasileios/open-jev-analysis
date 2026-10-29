"""Local dataset adapters and frozen splits for the replica experiment runner."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def digest(value):
    """Return a stable SHA256 of value, a JSON-serializable object."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def prepare_suite_dataset(raw, spec, *, source_revision, project_root):
    """Normalize labels, remove ambiguous texts, and freeze deterministic splits.

    Args:
        raw: Downloaded HF DatasetDict, loaded locally without network requests.
        spec: Dataset config with name, text_column, label_column/label_mapping
            (or any_positive_columns), classes, seed, validation_fraction,
            optional train_limit/test_limit, and a unique manifest_name.
            For a single source split, test_fraction is also held out.
        source_revision: Immutable revision from the snapshot source.json.
        project_root: Project directory where data/split_manifests stores identity
            checks. Changed settings or contents require a new manifest_name.

    Returns:
        (train, validation, test, preparation_audit, manifest). Every table has
        id/text/label/label_multiclass. Label 1 is the configured positive class.
        Single-split datasets become a new internal benchmark, not the complete
        original published test benchmark. Sources are never modified.
    """
    from datasets import Dataset
    if len(spec['classes']) != 2:
        raise ValueError('These adapters require two mutually exclusive classes')
    groups=defaultdict(list); excluded=Counter()
    for split,table in raw.items():
        for index,row in enumerate(table):
            text=row.get(spec['text_column'])
            if not isinstance(text,str) or not text.strip():
                excluded['empty_text']+=1;continue
            if 'any_positive_columns' in spec:
                values=[row.get(c) for c in spec['any_positive_columns']]
                label=1 if 1 in values else 0 if all(v==0 for v in values) else None
            else:
                value=row.get(spec['label_column'])
                label=spec['label_mapping'].get(str(value)) if value is not None else None
            if label not in (0,1):
                excluded['missing_or_unknown_label']+=1;continue
            normalized={'id':f'{split}:{index}','text':text.strip(),'label':int(label),
                        'label_multiclass':spec['classes'][label]}
            groups[text.strip()].append((split,normalized))
    prepared=defaultdict(list)
    for text,entries in groups.items():
        if len({r['label'] for _,r in entries})>1:
            excluded['conflicting_text_rows']+=len(entries);continue
        # Keep an official held-out copy over a training copy when duplicates exist.
        entries.sort(key=lambda e:({'test':0,'validation':1,'train':2}.get(e[0],3),e[1]['id']))
        split,row=entries[0];prepared[split].append(row)
        excluded['duplicate_text_rows']+=len(entries)-1
    seed=spec.get('seed',42)
    def ordered(rows):
        """Order rows by seeded content hash, independent of Python hash state."""
        return sorted(rows,key=lambda r:digest([seed,r['text']]))
    def holdout(rows, fraction):
        """Partition rows by class; fraction is the held-out proportion in (0,1)."""
        if not 0<fraction<1:raise ValueError('Split fraction must be in (0,1)')
        keep=[];held=[]
        for label in (0,1):
            members=ordered([r for r in rows if r['label']==label])
            if len(members)<3:raise ValueError('Not enough rows per class for disjoint splits')
            count=max(1,min(len(members)-2,round(len(members)*fraction)))
            held.extend(members[:count]);keep.extend(members[count:])
        return ordered(keep),ordered(held)
    if 'train' in prepared and 'test' in prepared:
        train=prepared['train'];test=prepared['test']
        validation=prepared.get('validation')
        if not validation:train,validation=holdout(train,spec.get('validation_fraction',.2))
        method='official test retained; stratified validation holdout from train'
    elif len(prepared)==1:
        all_rows=next(iter(prepared.values()))
        remainder,test=holdout(all_rows,spec.get('test_fraction',.2))
        relative=spec.get('validation_fraction',.2)/(1-spec.get('test_fraction',.2))
        train,validation=holdout(remainder,relative)
        method='single source split repartitioned into internal stratified train/validation/test'
    else:raise ValueError('Unsupported source split structure')
    selected={}
    for name,rows in [('train',train),('validation',validation),('test',test)]:
        rows=ordered(rows);limit=spec.get(name+'_limit')
        if limit is not None:
            if type(limit)is not int or limit<2:raise ValueError('Split limits must be integers >=2')
            if name in spec.get('balanced_limit_splits', []):
                chosen=[]
                for label in (0,1):
                    quota=limit//2+int(label==0 and limit%2)
                    members=[r for r in rows if r['label']==label]
                    if len(members)<quota:
                        raise ValueError(f'{name}: requested {quota} class-{label} rows, only {len(members)} available')
                    chosen.extend(members[:quota])
                rows=ordered(chosen)
            elif spec.get('stratified_limits') and limit < len(rows):
                positive=sum(r['label'] for r in rows)
                positive_count=max(1,min(limit-1,round(limit*positive/len(rows))))
                counts={0:limit-positive_count,1:positive_count}
                chosen=[]
                for label in (0,1):
                    members=[r for r in rows if r['label']==label]
                    chosen.extend(members[:counts[label]])
                rows=ordered(chosen)
            else:
                rows=rows[:limit]
        if {r['label'] for r in rows}!={0,1}:raise ValueError(f'{name} needs both classes')
        selected[name]=rows
    audit={'excluded':dict(excluded),'method':method,'class_counts':{k:dict(Counter(r['label_multiclass'] for r in v)) for k,v in selected.items()}}
    manifest={'version':1,'profile':spec['name'],'source_revision':source_revision,'settings':spec,
        'source_content_sha256':digest({k:[dict(r) for r in t] for k,t in raw.items()}),
        'selected_rows':selected,'preparation_audit':audit}
    manifest['selection_sha256']=digest(manifest)
    stem=spec['manifest_name']
    if Path(stem).name!=stem or stem in ('','.','..'):raise ValueError('Invalid manifest_name')
    path=Path(project_root)/'data/split_manifests'/f'{stem}.json';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=manifest:raise ValueError(f'Split manifest mismatch: {path}')
    else:
        with path.open('x') as stream:json.dump(manifest,stream,indent=2);stream.write('\n')
    return (*(Dataset.from_list(selected[k]) for k in ('train','validation','test')),audit,manifest)
