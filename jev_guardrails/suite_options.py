"""Command-line overrides for the separate multi-dataset suite runner."""
from copy import deepcopy
import hashlib
import json


def add_sample_arguments(parser):
    """Register sample and search flags on parser, an argparse subparser."""
    for split in ('train','validation','test'):
        parser.add_argument(f'--{split}-samples',type=int,
            help=f'Exact {split} rows for a balanced smoke subset (minimum 2); separate frozen manifest')
    parser.add_argument('--search-train-samples',type=int,
        help='Balanced training rows scored for each prompt candidate; automatically capped to training size')
    parser.add_argument('--breadth',type=int,help='Instruction candidates per DSPy round; use 1 for a quick check')
    parser.add_argument('--depth',type=int,help='DSPy search rounds; use 1 for a quick check')


def apply_sample_arguments(config, args):
    """Return a copied config with explicit CLI overrides; leave files unchanged.

    Args:
        config: Dataset suite dictionary, including dataset and experiment sections.
        args: Parsed argparse namespace. Optional train_samples, validation_samples,
            test_samples set exact balanced smoke subsets. search_train_samples
            controls candidate scoring, not the full training threshold fit.
            breadth/depth override search cost. Missing values keep config defaults.

    Returns:
        Effective config persisted in suite_config.json. Modified sample counts
        use a distinct manifest name, leaving full benchmark selections intact.
        Reusing --resume requires the same effective settings.
    """
    result=deepcopy(config)
    sizes={split:getattr(args,split+'_samples',None) for split in ('train','validation','test')}
    for split,value in sizes.items():
        if value is not None:
            if value<2:raise ValueError(f'--{split}-samples must be >=2 (one per class)')
            result['dataset'][split+'_limit']=value
    if any(value is not None for value in sizes.values()):
        # Balance only explicitly limited smoke splits, not unchanged full splits.
        result['dataset']['balanced_limit_splits']=[split for split,value in sizes.items() if value is not None]
        stem=result['dataset']['manifest_name']
        token=hashlib.sha256(json.dumps(result['dataset'],sort_keys=True).encode()).hexdigest()[:12]
        result['dataset']['manifest_name']=stem+'-samples-'+token
    experiment=result['experiment']
    for name in ('search_train_samples','breadth','depth'):
        value=getattr(args,name,None)
        if value is not None:
            minimum=2 if name=='search_train_samples' else 1
            if value<minimum:raise ValueError(f'--{name.replace("_","-")} must be >={minimum}')
            experiment['search_train_limit' if name=='search_train_samples' else name]=value
    train_limit=result['dataset'].get('train_limit')
    if train_limit is not None:experiment['search_train_limit']=min(experiment['search_train_limit'],train_limit)
    if sizes['validation'] is not None:experiment['validation_limit']=sizes['validation']
    return result
