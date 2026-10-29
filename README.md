<div align="center">

# 🎯 open-jev-analysis

### *How well does a general decision model work across different tasks?*

**A comprehensive research study of Open-Jev 2B and 9B models** for classification, guardrails, and retrieval reranking.

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/Dependencies-uv-DE5FE9)
![Models](https://img.shields.io/badge/Open--Jev-2B%20%7C%209B-2563EB)

[🚀 Quick start](#quick-start) · [🧪 Experiments](#experiments) · [📊 Plot results](#plot-results) · [📁 Repository guide](#repository-guide)

</div>

---

## 📋 The Study

**One decision interface, two research tracks:**

| Track | Research Question | Metrics |
|:-----:|:------------------|:--------|
| **🔍 Classification & Guardrails** | Can a shared decision model recognize different safety-related targets? | Accuracy, class F1, recall, false-positive rates, calibration, and latency |
| **🔎 Retrieval & Reranking** | Does a relevance judge improve the order produced by a retriever? | nDCG, MRR, recall and hit rate at **3, 5, and 10**, plus retrieval and reranking latency |

### Scope

- **Classification** covers six targets: prompt injection, harmful requests, toxicity, jailbreaking, XSTest safety, and a moderation-category union
- **Retrieval** compares BM25 and EmbeddingGemma 2, before and after Open-Jev reranking, on BEIR collections (max 60,000 documents)

> The models come from the independent [Open-Jev project](https://github.com/Zefan-Cai/Open-Jev). This study evaluates those checkpoints; retrieval experiments measure document rankings rather than generated answers.

## 📈 A Look at the Results

The experiments show task-dependent strengths and limitations. Adding a judge can improve some rankings and weaken others; high classification accuracy can also hide poor detection of rare positive examples.

This checkout contains the experiment code and configurations. Study reports, publication figures and derived analysis tables under `docs/`, along with the original datasets and run outputs, are local artifacts excluded from Git.

### Study Status

- ✅ **24/24** completed classification test configurations
- ⚠️ **59/64** completed retrieval jobs (5 failed on memory/input length)

> Results show task-dependent strengths and limitations; they do not establish universal decision reliability or calibrated confidence on new tasks.

<a id="quick-start"></a>

## 🚀 Quick Start

### Prerequisites

You need **Python 3.12+, uv, and Git**.

### Installation

From the repository root:

```bash
uv sync
```

For dense retrieval, also create the separate encoder environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-dense uv sync --only-group dense
```

> Both environments are declared in [pyproject.toml](pyproject.toml) and resolved in [uv.lock](uv.lock). They use different Transformers versions. The repository runs directly from the checkout.

### 🤖 Models and Credentials

Download the Open-Jev checkpoints separately using the [upstream instructions](https://github.com/Zefan-Cai/Open-Jev). The configurations expect:

```text
checkpoints/
├── open-jev-2b/package/checkpoint/
└── open-jev-9b/package/checkpoint/
```

⚠️ **GPU inference** requires suitable NVIDIA hardware and drivers. Review device placement, context length and batch size in the relevant configuration; the saved 9B settings reflect the original workstation.

**Classification prompt search** also uses an Azure proposer configured through [.env.example](.env.example). Retrieval and plotting do not need Azure. Dependency versions are pinned in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock). Record the checkpoint revision you download when reporting a new run.

<a id="experiments"></a>

## 🧪 Experiments

### 1️⃣ Classification and Guardrails

Prepare the data and preview a single task:

```bash
uv run python download_datasets.py --datasets deepset wildguard toxicchat xstest moderation
bash scripts/run_openjev_dataset_suites.sh deepset --dry-run
```

Run both model sizes on that task:

```bash
bash scripts/run_openjev_dataset_suites.sh deepset --output runs/deepset-study
```

<details>
<summary><strong>📋 Run all six tasks or resume a study</strong></summary>

```bash
JEV_SUITE_OUTPUT_ROOT=runs/classification-study \
  bash scripts/run_openjev_dataset_suites.sh all

JEV_SUITE_OUTPUT_ROOT=runs/classification-study \
  bash scripts/run_openjev_dataset_suites.sh all --resume
```

> Use a new output root for changed settings. To inspect available sources, run `uv run python download_datasets.py --list`.

</details>

> 💡 **Note:** Each prompt receives a threshold fitted on training predictions. Validation selects between the baseline and searched prompt before testing. Reports retain both prompts and native, fixed-0.5 and training-threshold decisions.

> ⚠️ **Important:** Threshold fitting changes decisions; it does not recalibrate probabilities. A prompt saved as `optimized` is a search candidate and may be rejected by validation. WildGuard, XSTest and moderation use internal partitions, while the two ToxicChat targets share a source.

### 2️⃣ Retrieval and Reranking

Prepare a small BEIR collection:

```bash
uv run python -m jev_guardrails.beir_data --datasets nfcorpus
```

Start with a retrieval-only baseline, then rerank with Open-Jev:

```bash
# BM25 only
bash scripts/run_rag_analysis.sh --bm25-only --queries 3 --candidates 10 \
  --output runs/nfcorpus-bm25

# BM25 + Open-Jev 2B
bash scripts/run_rag_analysis.sh --model 2b --queries 3 --candidates 10 \
  --output runs/nfcorpus-bm25-2b

# Dense retrieval + Open-Jev 2B
bash scripts/run_rag_analysis.sh --retriever dense --model 2b \
  --queries 3 --candidates 10 --output runs/nfcorpus-dense-2b
```

> Use `--model 9b` for the larger judge, `--dry-run` to preview a run, and `--resume` with the original arguments to continue it. Three-query runs check the workflow; they are not benchmark estimates.

<details>
<summary><strong>🔍 Run both retrievers and judges across prepared collections</strong></summary>

```bash
bash scripts/run_all_rag_datasets.sh --dry-run --output-root runs/rag-study

bash scripts/run_all_rag_datasets.sh --output-root runs/rag-study \
  --queries 100 --candidates 20 --cutoffs 3 5 10 --quiet
```

> The launcher selects prepared collections with at most 60,000 corpus documents. Servers run sequentially. The encoder exits before Open-Jev loads, and its vectors remain cached for reuse.

</details>

> 💡 **How it works:** The judge scores each query/document pair independently and sorts the frozen candidates by **P(relevant)**. It cannot recover documents outside that candidate pool. The runner manages its own server; `--existing-server` connects to a separately managed server. For an encoder environment outside `.venv-dense`, pass `--embedding-python /path/to/python`.

> 🚨 **Critical:** **ArguAna has a known protocol issue** - the current runner retains the query's same-ID document. The main analysis excludes ArguAna; its posthoc sensitivity is not a corrected top-20 experiment. Rebuild eligible candidates before making standard ArguAna benchmark claims.

<a id="plot-results"></a>

## 📊 Plot Your Results

After running the experiments above, generate plots from your local output folders:

```bash
# Classification: use the output root from the all-task example
uv run python scripts/plots/plot_dataset_spiders.py --suite-root runs/classification-study
uv run python scripts/plots/plot_guardrail_aggregate.py --suite-root runs/classification-study

# Retrieval quality and latency: use the output root from the collection sweep
uv run python scripts/plots/plot_rag_spiders.py --suite-root runs/rag-study
uv run python scripts/plots/plot_rag_latency.py --suite-root runs/rag-study
```

> Classification plotters compare baseline and search-winner prompts under their own training thresholds; these are not necessarily the validation-selected deployment prompts. RAG plotters use matched completed collections and retain any saved ArguAna results. Apply the protocol caveat above when interpreting them.

These commands require completed local runs. The original study outputs and publication-figure builder are not included in this checkout.

<a id="repository-guide"></a>

## 📁 Repository Guide

| Location | Contents |
|:---------|:---------|
| [config/](config/) | Task definitions, split settings, prompts and model configuration |
| [jev_guardrails/](jev_guardrails/) | Shared data, inference, scoring and comparison helpers |
| [scripts/](scripts/) | Experiment launchers and plotting entry points |
| [pyproject.toml](pyproject.toml) / [uv.lock](uv.lock) | Dependency groups and locked environment resolution |
| [.env.example](.env.example) | Azure proposer configuration template |

### Execution Notes

- The root Python runners handle classification execution and comparison
- Bash launchers automatically use `.venv`; set `JEV_PYTHON=/path/to/python` to choose another interpreter
- Local reports (`docs/`), downloaded data, checkpoints, vector caches, environments, credentials and full run outputs are Git-ignored; they are created or supplied locally
- Source manifests record revisions and selections

> 📝 **When reporting results:** Identify the suite, decision rule, completed cohort and aggregation weights; source access and redistribution conditions still apply.

---

<div align="center">

**[Open-Jev Project](https://github.com/Zefan-Cai/Open-Jev)** · **[Quick Start](#quick-start)** · **[Experiments](#experiments)**

</div>
