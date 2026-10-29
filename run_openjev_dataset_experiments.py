#!/usr/bin/env python3
"""Multi-dataset Open-Jev evaluation; run --help for suite, model, and comparison modes.

No packages are installed. Each classifier server is a separate process group;
only processes started by this runner are stopped. Azure credentials stay in .env.
"""
import argparse
import errno
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    """Save JSON; path is the destination and value is a serializable object."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def project_path(value):
    """Resolve value, a path string, relative to this project unless absolute."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    """Read path (suite JSON) and validate model names and checkpoint directories."""
    config = json.loads(Path(path).read_text())
    names = [m['name'] for m in config['models']]
    if not names or len(names) != len(set(names)):
        raise ValueError('Provide unique model names')
    for model in config['models']:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', model['name']):
            raise ValueError('Model names must be safe single directory names')
        if not project_path(model['checkpoint']).is_dir():
            raise ValueError(f"Missing checkpoint: {model['checkpoint']}")
    return config


def server_command(config, model):
    """Build argv from config server settings and one model entry; no shell eval."""
    server = {**config['server'], **model.get('server', {})}
    command = [sys.executable, '-m', 'jev.server', '--checkpoint', str(project_path(model['checkpoint'])),
               '--device', server['device'], '--max-length', str(server['max_length']),
               '--batch-size', str(server['batch_size']), '--host', server['host'], '--port', str(server['port'])]
    return command + server.get('extra_args', [])


def server_environment(config, model):
    """Build isolated subprocess environment with configured server overrides.

    Args:
        config: Suite config; server.env maps environment names to string values
            (or None to remove a variable). These values are saved in config;
            keep credentials in the parent environment, not this mapping.
        model: One model entry. Its server.env overrides shared server.env.
            The consumer-GPU placement/cache variables are cleared from inherited
            state first, so exported 9B settings cannot leak into another model.

    Returns:
        A new environment dictionary for the server only; os.environ is unchanged.
    """
    environment = dict(os.environ)
    for name in ('JEV_DEVICE_MAP', 'JEV_PREFILL_CHUNK', 'JEV_RAGGED_SUFFIX'):
        environment.pop(name, None)
    overrides = {**config['server'].get('env', {}), **model.get('server', {}).get('env', {})}
    for name, value in overrides.items():
        if value is None:
            environment.pop(name, None)
        elif isinstance(value, str):
            environment[name] = value
        else:
            raise ValueError(f'Server environment {name} must be a string or null')
    # Accelerate initializes a broad parent's hook with place_submodules=True.
    # That can materialize the CPU embedding on GPU before its child offload hook
    # is attached. Map only GPU-only subtrees, avoiding the overlapping parent.
    value = environment.get('JEV_DEVICE_MAP', '')
    if value.lstrip().startswith('{'):
        placement = json.loads(value)
        parent = 'model.language_model'
        if parent in placement and placement.get(parent + '.embed_tokens') == 'cpu':
            device = placement.pop(parent)
            for child in ('layers', 'norm', 'rotary_emb'):
                placement.setdefault(parent + '.' + child, device)
            environment['JEV_DEVICE_MAP'] = json.dumps(placement, separators=(',', ':'))
    return environment


def health(config, model):
    """Fetch health for config/model server address; caller handles startup errors."""
    server = {**config['server'], **model.get('server', {})}
    with urlopen(f"http://{server['host']}:{server['port']}/health", timeout=3) as response:
        result = json.load(response)
    if result.get('status') != 'ready':
        raise ValueError('Server not ready')
    if model.get('expected_model') and result.get('model') != model['expected_model']:
        raise ValueError(f"Wrong checkpoint identity: {result.get('model')}")
    return result


def wait_for_server_port(host, port, timeout=30):
    """Wait until the HTTP server can bind its address, ignoring stale TIME_WAIT.

    Args:
        host: Bind address of the next server.
        port: TCP port number.
        timeout: Maximum seconds to retry EADDRINUSE after shutdown. No existing
            listener is terminated. SO_REUSEADDR mirrors Python HTTPServer;
            listen() ensures an active server cannot silently share the address.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((host, int(port)))
                probe.listen(1)
            return
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Port {host}:{port} still occupied after {timeout}s; '
                                   'stop the existing listener or choose another port') from exc
            time.sleep(.25)


def server_log_error(path):
    """Summarize path, a server log, so startup failures include their cause."""
    lines=Path(path).read_text(errors='replace').splitlines()
    errors=[line.strip() for line in lines if 'Error:' in line or 'Exception:' in line]
    return (errors[-1] if errors else 'no Python exception found') + f' (log: {path})'


def stop_process(process, timeout=30):
    """Stop the process group owned by process; timeout is TERM grace seconds."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    # Also remove any surviving children after their process-group leader exits.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def saved_validation_report(rows, references, classes, weight):
    """Compute prompt-selection metrics without repeating validation inference.

    Args:
        rows: Predictions already saved by predict_split for this prompt.
        references: Aligned validation Dataset with label and label_multiclass.
        classes: Ordered class names used for native confusion metrics.
        weight: Weakest-class F1 weight in [0,1]. Prediction errors abort selection,
            preserving the previous fail-on-error behavior.
    """
    from jev_guardrails.experiments import multiclass_metrics
    if any(r.get('error') is not None for r in rows):
        raise ValueError('Validation predictions contain errors; refusing prompt selection')
    report=multiclass_metrics(rows,references['label_multiclass'],classes)
    report['min_class_f1']=min(v['f1'] for v in report['per_class'].values())
    report['objective']=(1-weight)*report['macro_f1']+weight*report['min_class_f1']
    report['worst_class_weight']=weight
    report['binary_accuracy']=sum(r['predicted_label']==e['label'] for r,e in zip(rows,references))/len(rows)
    return report


def print_dataset_plan(config):
    """Inspect local selections and print request/time estimates without inference.

    Args:
        config: Effective dataset suite dictionary. Creates manifests only in a
            temporary directory. Time projections use saved smoke p50/p95 latency
            and exclude server loading, Azure generation, and plotting. They are
            planning estimates, not measured full-run durations or bounds.
    """
    import tempfile
    import pandas as pd
    from datasets import load_from_disk
    from jev_guardrails.suite_datasets import prepare_suite_dataset
    from jev_guardrails.dspy_prompting import make_prompt_search_sets
    cfg=config['experiment'];path=project_path(cfg['data_path'])
    source=json.loads((path/'source.json').read_text())
    with tempfile.TemporaryDirectory() as temporary:
        train,val,test,audit,_=prepare_suite_dataset(load_from_disk(str(path)),config['dataset'],
            source_revision=source['revision'],project_root=temporary)
    search,validation,_=make_prompt_search_sets(train,val,test,train_limit=cfg['search_train_limit'],
        validation_limit=cfg['validation_limit'],balanced=True,classes=config['dataset']['classes'])
    search_calls=(1+cfg['breadth']*cfg['depth'])*len(search)
    total=search_calls+2*(len(train)+len(validation)+len(test))
    print('Actual train/validation pool/test:',len(train),len(val),len(test))
    print('Class counts:',audit['class_counts'])
    print('Search/selection validation:',len(search),len(validation))
    print('Classifier requests per model:',total,'(search:',search_calls,')')
    print('Proposed rewrites per model:',cfg['breadth']*cfg['depth'])
    smoke=ROOT/'runs'/(config['dataset']['name'].replace('_','-')+'-smoke')
    comparisons=sorted(smoke.glob('comparison-*/all_results.csv'))
    if comparisons:
        table=pd.read_csv(comparisons[-1])
        for model in config['models']:
            measured=table[table['model_name']==model['name']]
            if len(measured):
                p50=measured['latency_p50_ms'].median()/1000
                p95=measured['latency_p95_ms'].median()/1000
                print(f"{model['name']}: about {total*p50/60:.1f} min at smoke p50; "
                      f"{total*p95/60:.1f} min at smoke p95 (classification only)")
    print('Estimates exclude loading/Azure overhead; longer real inputs may take more time.')


def run_model(config, model, output):
    """Run one full experiment against an already-running server.

    Args:
        config: Suite dictionary with experiment settings and the initial question.
            search_train_limit/breadth/depth control DSPy cost; worst_class_weight
            weights the weakest-class F1 term; max_class_f1_drop is the validation
            safeguard. See config/openjev_suite.json and notebook parameter guide.
        model: Model entry with name, checkpoint, and optional expected_model.
        output: New model experiment directory; existing results are never replaced.

    Returns:
        Saved completion record. Thresholds use shared training labels only;
        validation selects a prompt and test results never influence selection.
    """
    from datasets import Dataset, load_from_disk
    from jev.client import Client
    from jev_guardrails.suite_datasets import prepare_suite_dataset
    from jev_guardrails.dspy_prompting import (OpenJevPromptProgram, make_prompt_search_sets,
        ClassBalancedPromptOptimizer, evaluate_class_balanced, choose_class_balanced_prompt)
    from jev_guardrails.prompt_lm import prompt_lm_from_env
    from jev_guardrails.experiments import (predict_split, fit_training_threshold,
        compare_test_thresholds, multiclass_metrics)
    from jev_guardrails.plots import plot_experiment, plot_multiclass_results
    import matplotlib.pyplot as plt
    from importlib.metadata import version

    cfg = config['experiment']
    random.seed(cfg['seed'])
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    server_health = health(config, model)
    server = {**config['server'], **model.get('server', {})}
    endpoint = f"http://{server['host']}:{server['port']}/v1/systemone"
    data_path = project_path(cfg['data_path'])
    source = json.loads((data_path / 'source.json').read_text())
    CLASSES = config['dataset']['classes']
    train, validation_data, test, audit, manifest = prepare_suite_dataset(
        load_from_disk(str(data_path)), config['dataset'], source_revision=source['revision'], project_root=ROOT)
    search, validation, search_audit = make_prompt_search_sets(train, validation_data, test,
        train_limit=cfg['search_train_limit'], validation_limit=cfg['validation_limit'], balanced=True, classes=CLASSES)
    proposer, public = prompt_lm_from_env(project_path(cfg['env_file']), provider=cfg['provider'],
        temperature=cfg['temperature'], max_tokens=cfg['max_tokens'])
    output.mkdir(parents=True)
    write_json(output / 'prompt_search_splits.json', search_audit)
    baseline = OpenJevPromptProgram(Client(endpoint=endpoint, timeout=cfg['timeout']), config['question'],
        model_alias=cfg['model_alias'], expected_model=server_health['model'],
        positive_classes=(CLASSES[1],), negative_class=CLASSES[0])
    print(f"{model['name']}: prompt search on {len(search)} posts", flush=True)
    optimized = ClassBalancedPromptOptimizer(proposer, classes=CLASSES, breadth=cfg['breadth'],
        depth=cfg['depth'], worst_class_weight=cfg['worst_class_weight']).compile(baseline, trainset=search)
    write_json(output / 'candidate_history.json', optimized.candidate_history)
    programs = {'baseline': baseline, 'optimized': optimized}
    for name, program in programs.items():
        write_json(output / f'{name}_question.json', program.question())
    common = {'backend':'jev.client.Client.ask', 'model_name':model['name'], 'server_health':server_health,
        'server_run_details':{'command':server_command(config,model), 'checkpoint':model['checkpoint'],
            'environment_overrides':{key:server_environment(config,model).get(key) for key in
                {**config['server'].get('env',{}), **model.get('server',{}).get('env',{})}}},
        'dataset_source':source, 'preparation_audit':audit,
        'split_selection':{k:manifest[k] for k in ('profile','source_revision','selection_sha256')},
        'positive_classes':[CLASSES[1]], 'negative_class':CLASSES[0]}
    metadata = {name:{**common,'question':p.question(),'prompt_variant':name} for name,p in programs.items()}
    thresholds = {}
    # Both prompts finish training threshold fitting before any validation scoring.
    for name, program in programs.items():
        directory = output / name
        predict_split(train, program.predict_record, directory / 'train', split='train',
            metadata=metadata[name], progress_every=128)
        thresholds[name] = fit_training_threshold(directory / 'train')
        fitted = thresholds[name]
        print(f"{model['name']}/{name}: training threshold fitted = {fitted['threshold']:.9f}; "
              f"errors {fitted['training_baseline']['errors']} -> {fitted['training_best']['errors']} "
              f"on {fitted['training_scored']} training rows (0.5 may remain optimal)", flush=True)
    val_table = Dataset.from_list([{'text':e.text,'label':e.binary_label,'label_multiclass':e.label} for e in validation])
    val_reports, val_binary = {}, {}
    for name, program in programs.items():
        directory = output / name
        validation_rows = predict_split(val_table, program.predict_record, directory / 'validation', split='validation',
            metadata=metadata[name], progress_every=128)
        val_binary[name], _ = compare_test_thresholds(directory / 'validation',
            directory / 'train/frozen_threshold.json', evaluation_split='validation')
        val_reports[name] = saved_validation_report(validation_rows, val_table, CLASSES, cfg['worst_class_weight'])
    choice = choose_class_balanced_prompt(val_reports['baseline'], val_reports['optimized'],
        max_class_f1_drop=cfg['max_class_f1_drop'])
    selected = choice['selected_for_deployment']
    selection = {**choice, 'selection_split':'validation', 'threshold_selection_split':'train',
        'selected_question':programs[selected].question(),'selected_threshold':thresholds[selected]['threshold'],
        'validation':val_reports,'validation_binary':val_binary, 'settings':cfg,
        'prompt_provider':public,'dspy_version':version('dspy'),'search_selection_sha256':search_audit['selection_sha256']}
    write_json(output / 'frozen_prompt_selection.json', selection)
    for name, program in programs.items():
        directory = output / name
        rows = predict_split(test, program.predict_record, directory / 'test', split='test',
            metadata=metadata[name], progress_every=128)
        compare_test_thresholds(directory / 'test', directory / 'train/frozen_threshold.json')
        write_json(directory / 'test/multiclass_metrics.json', multiclass_metrics(rows, test['label_multiclass'], CLASSES))
        plot_experiment(directory)
        plot_multiclass_results(directory)
        plt.close('all')
    result = {'model':model['name'],'output':str(output.resolve()),'selected_prompt':selected,
              'test_selection_sha256':manifest['selection_sha256'], 'search_selection_sha256':search_audit['selection_sha256']}
    write_json(output / 'completed.json', result)
    return result


def compare_suite(output):
    """Compare all completed model/prompt folders under output; save CSV/JSON/plots.

    Args:
        output: Suite root with suite_config.json and completed per-model folders.
            Includes baseline and optimized prompts, each with native, fixed 0.5,
            and training-tuned thresholds. Uses identical successful test rows.
    """
    from jev_guardrails.comparison import compare_experiments, plot_model_comparison
    from jev_guardrails.experiments import multiclass_metrics
    import matplotlib.pyplot as plt
    import pandas as pd
    output = Path(output)
    config = json.loads((output / 'suite_config.json').read_text())
    completed = [m for m in config['models'] if (output / m['name'] / 'completed.json').exists()]
    if not completed:
        raise ValueError('No completed models to compare')
    audits = [json.loads((output / m['name'] / 'prompt_search_splits.json').read_text()) for m in completed]
    if len({a['selection_sha256'] for a in audits}) != 1:
        raise ValueError('Search/validation selections differ across models')
    roots = [output / m['name'] / variant for m in completed for variant in ('baseline','optimized')]
    names = [f"{m['name']}/{variant}" for m in completed for variant in ('baseline','optimized')]
    destination = output / ('comparison-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    report = compare_experiments(roots, names=names, include_tuned=True, output_dir=destination)
    fig = plot_model_comparison(report)
    for ext in ('png','svg'):
        fig.savefig(destination / f'accuracy.{ext}', dpi=180, bbox_inches='tight')
    native = {}
    from datasets import load_from_disk
    from jev_guardrails.suite_datasets import prepare_suite_dataset
    path = project_path(config['experiment']['data_path'])
    source = json.loads((path / 'source.json').read_text())
    CLASSES = config['dataset']['classes']
    _, _, test, _, _ = prepare_suite_dataset(load_from_disk(str(path)), config['dataset'],
        source_revision=source['revision'], project_root=ROOT)
    indices = report['common_test_row_indices']
    for name, root in zip(names, roots):
        rows = [json.loads(line) for line in (root / 'test/predictions.jsonl').read_text().splitlines()]
        native[name] = multiclass_metrics([rows[i] for i in indices], [test[i]['label_multiclass'] for i in indices], CLASSES)
    write_json(destination / 'native_multiclass.json', native)
    table = pd.DataFrame({name:{label:metrics['per_class'][label]['f1'] for label in CLASSES}
                          for name,metrics in native.items()})
    table.to_csv(destination / 'per_class_f1.csv')
    ax = table.plot.bar(figsize=(max(9,len(names)*1.4),5), rot=0, ylim=(0,1.05), ylabel='F1', title='Native class F1 on common test rows')
    for ext in ('png','svg'):
        ax.figure.savefig(destination / f'per_class_f1.{ext}',dpi=180,bbox_inches='tight')
    plt.close('all')
    summary = pd.DataFrame(report['summary'])
    summary['model_name'] = summary['model'].str.split('/').str[0]
    summary['prompt_variant'] = summary['model'].str.split('/').str[1]
    selected = {m['name']:json.loads((output/m['name']/'frozen_prompt_selection.json').read_text())['selected_for_deployment'] for m in completed}
    summary['prompt_selected_by_validation'] = [selected[m] == v for m,v in zip(summary['model_name'],summary['prompt_variant'])]
    summary['native_macro_f1'] = [native[name]['macro_f1'] for name in summary['model']]
    for label in CLASSES:
        summary[f'native_{label}_f1'] = [native[name]['per_class'][label]['f1'] for name in summary['model']]
    summary.to_csv(destination / 'all_results.csv', index=False)
    print(summary[['model','method','accuracy','precision','recall','f1','threshold']].to_string(index=False))
    write_json(destination / 'coverage.json', {'requested_models':[m['name'] for m in config['models']],
        'completed_models':[m['name'] for m in completed], 'missing_models':[m['name'] for m in config['models'] if m not in completed],
        'common_test_rows':len(report['common_test_row_indices'])})
    print('Comparison:',destination,flush=True)
    return destination


def run_suite(config, output, *, resume=False, dry_run=False):
    """Run servers sequentially in isolated process groups, then compare results.

    Args:
        config: Validated configuration dictionary; model entries can override server settings.
        output: Suite root path. A new run snapshots its configuration here.
        resume: Skip completed models using the exact saved config. Incomplete models
            are moved to timestamped backups before retrying; prior files are preserved.
        dry_run: Print commands and estimated scoring calls only; start no servers,
            create no experiment output, and make no Azure requests.
    """
    output = Path(output).resolve()
    if dry_run:
        cfg=config['experiment']
        print('Dataset:',config['dataset']['name'])
        print('Output:',output)
        print('Dataset row limits:', {split:config['dataset'].get(split+'_limit') for split in ('train','validation','test')})
        print('Search/validation scoring rows:',cfg['search_train_limit'],cfg['validation_limit'])
        print_dataset_plan(config)
        for model in config['models']:
            print(model['name'],server_command(config,model))
            print('  Effective server environment:', {key:server_environment(config,model).get(key) for key in
                {**config['server'].get('env',{}), **model.get('server',{}).get('env',{})}})
        return
    if output.exists():
        if not resume:
            raise FileExistsError('Suite output exists; use --resume or a new --output')
        if json.loads((output / 'suite_config.json').read_text()) != config:
            raise ValueError('Resume requires exactly the saved suite config')
    else:
        output.mkdir(parents=True)
        write_json(output / 'suite_config.json',config)
    status = {}
    try:
        for model in config['models']:
            name=model['name']; target=output/name
            if resume and (target/'completed.json').exists():
                status[name]='already completed'; continue
            if target.exists():
                target.rename(output/(name+'.incomplete-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
            server={**config['server'],**model.get('server',{})}
            server_process=experiment_process=None
            try:
                # Never connect to or kill an unrelated server on an occupied port.
                wait_for_server_port(server['host'], server['port'], server['shutdown_timeout'])
                with (output/f'{name}.server.log').open('a') as log:
                    server_process=subprocess.Popen(server_command(config,model),cwd=ROOT,env=server_environment(config,model),
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    deadline=time.monotonic()+server['startup_timeout']
                    while True:
                        if server_process.poll() is not None:
                            raise RuntimeError(f'{name} server exited: ' + server_log_error(output / f'{name}.server.log'))
                        try:
                            health(config,model);break
                        except Exception:
                            if time.monotonic()>=deadline:
                                raise TimeoutError(f'{name} readiness timed out; see server log')
                            time.sleep(2)
                    print(f'{name}: ready; starting experiment',flush=True)
                    with (output/f'{name}.experiment.log').open('a') as experiment_log:
                        experiment_process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'model',
                            '--config',str(output/'suite_config.json'),'--model',name,'--output',str(target)],
                            cwd=ROOT,stdout=experiment_log,stderr=subprocess.STDOUT,start_new_session=True)
                        if experiment_process.wait()!=0:
                            raise RuntimeError(f'{name} experiment failed; see {name}.experiment.log')
                status[name]='completed'
            except Exception as exc:
                status[name]=f'failed: {exc}'
                print(f'{name}: {status[name]}',file=sys.stderr,flush=True)
            finally:
                stop_process(experiment_process,server['shutdown_timeout'])
                stop_process(server_process,server['shutdown_timeout'])
                write_json(output/'suite_status.json',status)
    finally:
        # Retain a useful partial comparison if interrupted after any completed model.
        if any((output/m['name']/'completed.json').exists() for m in config['models']):
            compare_suite(output)
    if any(v.startswith('failed') for v in status.values()):
        raise RuntimeError('Some models failed; inspect suite_status.json and logs, then --resume')


def collect_datasets(output):
    """Combine latest per-dataset comparison CSVs without pooling task metrics.

    Args:
        output: Root containing dataset suite directories. Writes all_datasets.csv
            and collection.json, identifying absent suites and exact source files.
    """
    import pandas as pd
    output=Path(output)
    tables=[];sources={};missing=[]
    for name in ('deepset','wildguard','toxicchat','toxicchat_jailbreaking','xstest','moderation'):
        candidates=sorted((output/name).glob('comparison-*/all_results.csv'))
        if not candidates:missing.append(name);continue
        path=candidates[-1];table=pd.read_csv(path);table.insert(0,'dataset',name)
        tables.append(table);sources[name]=str(path.resolve())
    if not tables:raise ValueError('No dataset comparisons to collect')
    pd.concat(tables,ignore_index=True).to_csv(output/'all_datasets.csv',index=False)
    write_json(output/'collection.json',{'sources':sources,'missing_datasets':missing,
        'note':'Per-dataset results only; no pooled or cross-task averaged accuracy.'})
    print('Combined results:',output/'all_datasets.csv',flush=True)


def main(argv=None):
    """Parse argv (None reads sys.argv) and dispatch suite/model/compare commands."""
    from jev_guardrails.suite_options import add_sample_arguments, apply_sample_arguments
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('suite','model'):
        p=commands.add_parser(name)
        p.add_argument('--config',type=Path,default=ROOT/'config/suites/deepset.json')
        p.add_argument('--output',type=Path,required=name=='model',help='Suite root, or one model root in model mode')
        add_sample_arguments(p)
        if name=='suite':
            p.add_argument('--resume',action='store_true')
            p.add_argument('--dry-run',action='store_true')
        else:p.add_argument('--model',required=True)
    p=commands.add_parser('compare');p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('collect');p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    # Convert TERM to an exception so owned server/experiment groups are cleaned up.
    def interrupted(signum, frame):
        """Handle signum/frame from OS signals by unwinding through cleanup."""
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    if args.command=='collect':return collect_datasets(args.output)
    if args.command=='compare':return compare_suite(args.output)
    config=apply_sample_arguments(load_config(args.config), args)
    if args.command=='model':
        model=next(m for m in config['models'] if m['name']==args.model)
        return run_model(config,model,args.output)
    output=args.output or ROOT/'runs'/('openjev-'+config['dataset']['name']+'-suite-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run_suite(config,output,resume=args.resume,dry_run=args.dry_run)


if __name__=='__main__':
    main()
