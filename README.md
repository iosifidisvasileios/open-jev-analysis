# open-jev-analysis

A research repository for studying the capabilities and limitations of
**Open-Jev 2B and 9B** through two experiments:

- **Classification:** six safety-related targets, with fixed prompts, prompt
  search, and thresholds fitted on training data.
- **Retrieval and reranking:** BM25 or EmbeddingGemma 2 candidates, reordered by
  Open-Jev relevance scores on a resource-limited BEIR subset.

The checkpoints come from the independent
[Open-Jev project](https://github.com/Zefan-Cai/Open-Jev). These results do not
evaluate TypeSafe's hosted Jev model. The RAG experiment measures document
ranking, not generated answers.

Start with the [analysis and blog-post outline](docs/analysis/blogpost_analysis.md)
for the findings, or the [experiment guide](docs/experiment-guide.md) for the
full protocol and saved artifact reference.

## Repository map

| Path | Purpose |
| --- | --- |
| `run_openjev_dataset_experiments.py` | Six-task classification, prompt search, comparison and collection |
| `scripts/run_openjev_dataset_suites.sh` | Run one classification task or all six |
| `scripts/run_rag_analysis.sh` | One retrieval/reranking experiment |
| `scripts/run_all_rag_datasets.sh` | Both retrievers and both judges across eligible BEIR collections |
| `download_datasets.py`, `jev_guardrails/beir_data.py` | Prepare classification and retrieval data |
| `config/suites/` | Per-task labels, splits, prompts and model settings |
| `config/openjev_suite.json` | RAG server settings and separate HateXplain experiment configuration |
| `jev_guardrails/` | Shared data, inference, metric and plotting helpers |
| `scripts/plots/` | Replot saved predictions without inference |
| `docs/analysis/` | Evidence reports, derived tables, figures and analysis scripts |
| `dspy_hatexplain_openjev_client.ipynb` | Additional interactive three-class HateXplain study |

Scripts run directly from the repository root; this repository is not packaged
as an application or Python distribution.

## Before running

Use Python 3.12 and Git. Create the main environment with:

```bash
uv sync
```

For dense retrieval, create the separate encoder environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-dense uv sync --only-group dense
```

Dependencies for both runtimes are in [pyproject.toml](pyproject.toml). They use
separate environments because they require different Transformers versions.
See the [environment notes](docs/environment.md) for recorded versions.
Model checkpoints and datasets are downloaded separately. GPU inference requires
suitable NVIDIA hardware and drivers.

The configurations expect these checkpoint directories:

```text
checkpoints/open-jev-2b/package/checkpoint
checkpoints/open-jev-9b/package/checkpoint
```

Review the checkpoint paths, device, context limit and batch size in the relevant
JSON configuration. The saved 9B setup uses CPU offloading and prefix caching;
its placement reflects the original workstation.

Classification prompt search also needs an Azure deployment configured through
[.env.example](.env.example). The external proposer generates candidate rubrics;
Open-Jev scores them. Retrieval/reranking and plotting do not use Azure.

Dense retrieval runs its encoder in a separate environment because its
Transformers version differs from the Open-Jev runtime. Both runtime versions
are recorded in the [environment notes](docs/environment.md). Pass the encoder
environment's Python with `--embedding-python /path/to/dense/python` if it is
outside `.venv-dense`.

Commands below assume the selected environment's `python` is on `PATH`.
The Bash launchers also detect an optional local `.venv`; set
`JEV_PYTHON=/path/to/python` to select a particular interpreter. Launchers do not
install dependencies or change environments.

## Classification workflow

```bash
# Inspect sources, then prepare data for the six classification tasks.
python download_datasets.py --list
python download_datasets.py --datasets deepset wildguard toxicchat xstest moderation

# Inspect effective selections and workload without model/Azure calls.
bash scripts/run_openjev_dataset_suites.sh deepset --dry-run

# One task, both models. Prompt search uses the configured Azure deployment.
bash scripts/run_openjev_dataset_suites.sh deepset --output runs/deepset-study

# All six tasks; use a new root for a new protocol.
JEV_SUITE_OUTPUT_ROOT=runs/classification-study \
  bash scripts/run_openjev_dataset_suites.sh all

# Continue with the same arguments/configurations.
JEV_SUITE_OUTPUT_ROOT=runs/classification-study \
  bash scripts/run_openjev_dataset_suites.sh all --resume
```

The six tasks are Deepset prompt injection, WildGuard harmful input, ToxicChat
toxicity, ToxicChat jailbreaking, XSTest safety and a moderation-category union.
The ToxicChat tasks share a source. WildGuard, XSTest and moderation use internal
partitions; they are not full official held-out benchmark evaluations.

Each prompt gets an accuracy-maximizing threshold from training predictions.
Validation selects between baseline and search candidate before test evaluation.
The saved reports retain both candidates and native, fixed-0.5 and fitted-threshold
decisions. **Threshold fitting does not recalibrate probabilities**, and a prompt
named `optimized` is not necessarily the validation-selected policy.

## Retrieval and reranking workflow

```bash
# Prepare one official BEIR archive, including its relevance judgments.
python -m jev_guardrails.beir_data --datasets nfcorpus

# Retrieval-only baseline; no Open-Jev server is started.
bash scripts/run_rag_analysis.sh --bm25-only --queries 3 --candidates 10 \
  --output runs/nfcorpus-bm25

# Inspect, then run a small BM25 + Open-Jev comparison.
bash scripts/run_rag_analysis.sh --model 2b --queries 3 --candidates 10 --dry-run
bash scripts/run_rag_analysis.sh --model 2b --queries 3 --candidates 10 \
  --output runs/nfcorpus-bm25-2b

# Dense candidates, using the separate encoder environment.
bash scripts/run_rag_analysis.sh --retriever dense --model 2b \
  --queries 3 --candidates 10 --output runs/nfcorpus-dense-2b

# Both retrievers and both judges across prepared collections <=60,000 documents.
bash scripts/run_all_rag_datasets.sh --dry-run --output-root runs/rag-study
bash scripts/run_all_rag_datasets.sh --output-root runs/rag-study \
  --queries 100 --candidates 20 --cutoffs 3 5 10 --quiet
```

Use `--model 9b` and a separate output for the larger judge. The runner starts and
stops its own server; `--existing-server` reuses an independently managed server.
Reuse the original arguments with `--resume` after an interruption. The encoder
exits before the judge loads, and cached vectors are retained for later runs.

Open-Jev scores each query/document pair, then sorts the frozen candidates by
P(relevant). It does not retrieve additional documents or fit a threshold.
Reports include nDCG, MRR, recall and hit rate at 3, 5 and 10. Three-query runs
check the workflow; they do not estimate benchmark performance.

**ArguAna needs a protocol correction:** the current runner retains the query's
same-ID document. Exclude ArguAna from benchmark aggregate claims until the
eligible candidate pool is rebuilt and reranked. The analysis includes a
posthoc sensitivity, which is not a corrected top-20 run.

## Read and plot the saved study

The analyzed local suites are `runs/realistic-datasets` and
`runs/rag-under-60k-latest`. Generated runs and downloaded data are kept outside
the public source tree through Git ignore rules, so these folders may be absent
in a fresh checkout. The curated reports and derived tables live in
[docs/analysis](docs/analysis/blogpost_analysis.md).

```bash
MPLBACKEND=Agg python scripts/plots/plot_dataset_spiders.py --suite-root runs/realistic-datasets
MPLBACKEND=Agg python scripts/plots/plot_guardrail_aggregate.py --suite-root runs/realistic-datasets
MPLBACKEND=Agg python scripts/plots/plot_rag_spiders.py --suite-root runs/rag-under-60k-latest
MPLBACKEND=Agg python scripts/plots/plot_rag_latency.py --suite-root runs/rag-under-60k-latest
```

Classification plots compare baseline and search-winner prompts with their
training-fitted thresholds. They do not automatically restrict results to the
validation-selected prompt. RAG plots require all four configurations to complete
for a collection, average shared CQADupstack forums, and retain saved ArguAna
results. Use the [analysis report](docs/analysis/blogpost_analysis.md) for
selected-policy figures, the ArguAna-excluded sensitivity and interpretation.

In the saved study, all 24 classification test prediction sets complete, while
59 of 64 RAG jobs complete. The remaining jobs fail on memory or input length.
The findings support a reusable decision interface with task-dependent strengths
and failures; they do not establish generic reliability, portable confidence or
better generated RAG answers.

Datasets, model weights, vectors, environments, credentials and full run outputs
remain local under `data/`, `checkpoints/`, `.venv*`, `.env` and `runs/`. Source
manifests record revisions and selections; dataset and model access/redistribution
conditions still apply. Published tables should identify the exact suite,
decision rule, completion cohort and aggregation weights.
