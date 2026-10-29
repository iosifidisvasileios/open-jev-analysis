#!/usr/bin/env python3
"""Plot saved per-query retrieval + Open-Jev latency on matched collections."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

if __package__:
    from .plot_rag_spiders import LABELS, ROOT, SERIES, load_results
else:
    from plot_rag_spiders import LABELS, ROOT, SERIES, load_results


def load_latency(suite, collection_level=False):
    import pandas as pd

    _, _, axes, provenance = load_results(suite, collection_level=collection_level)
    records = []
    sources = {}
    for collection in provenance['included_collections']:
        for retriever, model in SERIES:
            folder = Path(suite) / collection / retriever / model
            plan = json.loads((folder / 'plan.json').read_text())
            path = folder / 'per_query.csv'
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            sources[str(path.relative_to(suite))] = hashlib.sha256(path.read_bytes()).hexdigest()
            by_method = {}
            for method in (retriever, retriever + '+jev'):
                selected = [r for r in rows if r['method'] == method]
                values = {}
                for row in selected:
                    value = float(row['latency_ms'])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f'Invalid latency: {path}')
                    if row['query_id'] in values:
                        raise ValueError(f'Duplicate query: {path}')
                    values[row['query_id']] = value
                if set(values) != set(plan['query_ids']):
                    raise ValueError(f'Latency query IDs differ from plan: {path}')
                by_method[method] = values
            for qid in plan['query_ids']:
                retrieval = by_method[retriever][qid]
                total = by_method[retriever + '+jev'][qid]
                if total < retrieval:
                    raise ValueError(f'Total latency is below retrieval latency: {path}')
                records.append({
                    'collection': collection,
                    'dataset': collection if collection_level else collection.split('/')[0],
                    'retriever': retriever, 'model': model, 'query_id': qid,
                    'retrieval_ms': retrieval,
                    'reranking_ms': total - retrieval,
                    'total_ms': total,
                })
    raw = pd.DataFrame(records)
    keys = ['dataset', 'retriever', 'model']
    columns = ['retrieval_ms', 'reranking_ms', 'total_ms']
    collection_means = raw.groupby(['collection', *keys], sort=False)[columns].mean().reset_index()
    table = collection_means.groupby(keys, sort=False)[columns].mean().reset_index()
    counts = raw.groupby(keys).agg(queries=('query_id', 'size'), collections=('collection', 'nunique')).reset_index()
    table = table.merge(counts, on=keys, validate='one_to_one')
    provenance['latency_sources_sha256'] = sources
    provenance['latency_definition'] = (
        'Per-query retrieval plus all candidate-scoring HTTP requests; excludes server startup and '
        'index construction. Dense query encoding is amortized; cached embeddings affect retrieval timings.'
    )
    return table, raw, axes, provenance


def plot_latency(table, axes, output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = ['#2563EB', '#EA580C', '#0891B2', '#BE123C']
    fig, panels = plt.subplots(2, 1, figsize=(max(13, len(axes) * 1.9), 10))
    x = np.arange(len(axes))
    width = .19
    sizes = table.groupby('dataset').first()
    labels = []
    for name in axes:
        row = sizes.loc[name]
        label = LABELS.get(name, name.replace('cqadupstack/', 'CQA: '))
        labels.append(f'{label}\nn={int(row.queries)}' +
                      (f' · {int(row.collections)} forums' if row.collections > 1 else ''))
    for ax, column, title in zip(
            panels, ('retrieval_ms', 'reranking_ms'),
            ('Retriever only', 'Open-Jev reranker only — all candidates per query')):
        for i, ((retriever, model), color) in enumerate(zip(SERIES, colors)):
            rows = table[(table.retriever == retriever) & (table.model == model)].set_index('dataset').loc[axes]
            values = rows[column].to_numpy()
            bars = ax.bar(x + (i - 1.5) * width, values, width, color=color)
            ax.bar_label(bars, labels=[f'{v:,.1f}' if column == 'retrieval_ms' else f'{v:,.0f}'
                                      for v in values], fontsize=8, padding=3)
        ax.set_xticks(x, labels)
        ax.set_ylabel('Mean latency per query (ms)')
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_ylim(bottom=0)
        ax.margins(y=.18)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Average per-query latency: retrieval and reranking', fontsize=19, y=.98)
    handles = [Patch(color=color, label=f'{"BM25" if r == "bm25" else "Dense"} + JEV {m.upper()} run')
               for (r, m), color in zip(SERIES, colors)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .075), ncol=4, frameon=False)
    fig.text(.5, .025,
             'Panels use separate scales; lower is better. Reranking is the sum of candidate-scoring HTTP times per query.\n'
             'Excludes startup/indexing. Dense retrieval depends on cache state. CQADupstack uses equal-forum means.',
             ha='center', fontsize=9, color='#475569')
    fig.tight_layout(rect=(0, .14, 1, .96), h_pad=2.5)
    for extension in ('png', 'svg'):
        fig.savefig(output / f'latency_bar.{extension}', dpi=160, facecolor='white')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-root', type=Path, default=ROOT / 'runs/rag-under-60k-latest')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--collection-level', action='store_true')
    args = parser.parse_args(argv)
    suite = args.suite_root.resolve()
    table, raw, axes, provenance = load_latency(suite, args.collection_level)
    output = (args.output or suite / ('latency-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plot_latency(table, axes, output)
    table.to_csv(output / 'dataset_latency.csv', index=False)
    raw.to_csv(output / 'per_query_latency.csv', index=False)
    (output / 'sources.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'excluded': provenance['excluded_collections']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
