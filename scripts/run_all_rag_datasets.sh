#!/usr/bin/env bash
# Run prepared BEIR collections with <=60,000 documents, sequentially for both retrievers/models.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
DATA_ROOT="$PROJECT_ROOT/data/downloads/beir"
OUTPUT_ROOT="${JEV_RAG_OUTPUT_ROOT:-$PROJECT_ROOT/runs/rag-all-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASETS=""
DRY_RUN=0
QUIET=0
EXTRA=()
usage() {
  cat <<'HELP'
Usage: scripts/run_all_rag_datasets.sh [launcher options] [experiment options]

Launcher options:
  --data-root PATH       Prepared BEIR root (default: data/downloads/beir).
  --output-root PATH     Root for separate collection/retriever/model outputs.
  --datasets NAMES       Optional comma-separated dataset or collection names,
                         e.g. nfcorpus,fiqa or cqadupstack/android.
  --dry-run              Print all commands; perform no indexing or inference.
  --quiet                Show dataset/job status only; save detailed output to logs.
  --help                 Show this help.

Other options pass to run_rag_analysis.sh, e.g. --queries 20 --candidates 20
--cutoffs 3 5 10 --seed 42 --embedding-device auto --resume.

Order per collection: BM25+2B, BM25+9B, dense+2B, dense+9B.
Collections with more than 60,000 corpus documents are excluded, including
individual CQADupstack forums. Query count is not the size filter.
Every job finishes and releases its owned model server before the next starts.
Failed jobs are logged; the remaining jobs continue. Any failure gives exit 1.
HELP
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root|--output-root|--datasets)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --data-root) DATA_ROOT="$2" ;;
        --output-root) OUTPUT_ROOT="$2" ;;
        --datasets) DATASETS="$2" ;;
      esac
      shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --quiet) QUIET=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --data|--data=*|--output|--output=*|--retriever|--retriever=*|--model|--model=*|--bm25-only|--retrieval-only|--existing-server)
      echo "This launcher manages $1; use --data-root/--output-root and its four configured pipelines." >&2
      exit 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
JOBS="$(mktemp)"
trap 'rm -f -- "$JOBS"' EXIT
# Inspect metadata only. In particular, --dry-run does not load million-document corpora.
"$PYTHON_BIN" - "$DATA_ROOT" "$OUTPUT_ROOT" "$DATASETS" "$PROJECT_ROOT/jev_guardrails/beir_catalog.json" > "$JOBS" <<'PY'
import json
from pathlib import Path
import sys
root, output = (Path(p).resolve() for p in sys.argv[1:3])
selected = {s.strip() for s in sys.argv[3].split(',') if s.strip()}
catalog = json.loads(Path(sys.argv[4]).read_text())
rows, matched = [], set()
max_documents = 60_000
for manifest_path in sorted(root.glob('*/manifest.json')):
    dataset = manifest_path.parent.name
    if dataset not in catalog:
        continue
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'complete':
        continue
    for relative, collection in sorted(manifest['collections'].items()):
        key = dataset if relative == '.' else dataset + '/' + relative
        if selected and dataset not in selected and key not in selected:
            continue
        matched.update(selected & {dataset, key})
        documents = collection.get('documents')
        if type(documents) is not int or documents < 1:
            raise SystemExit(f'Missing or invalid corpus document count for {key}')
        if documents > max_documents:
            print(f'Skipping {key}: {documents:,} documents exceeds {max_documents:,}.', file=sys.stderr)
            continue
        path = (manifest_path.parent / relative).resolve()
        if not path.is_relative_to(root) or any(v in key for v in ('\t','\n','..')):
            raise SystemExit(f'Unsafe collection path: {key}')
        # Runner defaults to test; an explicitly forwarded --split is validated by it.
        rows.append((key, str(path), str(output/key)))
if selected - matched:
    raise SystemExit('No prepared collections for: ' + ', '.join(sorted(selected-matched)))
if not rows:
    raise SystemExit(f'No completed BEIR collections with at most {max_documents:,} documents found in {root}')
for name, spec in catalog.items():
    if spec['url'] is None and (not selected or name in selected):
        print(f'Skipping {name}: separate corpus access required.', file=sys.stderr)
for row in rows:
    print('\t'.join(row))
PY
COUNT="$(wc -l < "$JOBS")"
printf 'Collections (<=60,000 documents): %s | Jobs: %s | Output root: %s\n' "$COUNT" "$((COUNT * 4))" "$OUTPUT_ROOT"
if [[ "$DRY_RUN" == 0 ]]; then
  mkdir -p -- "$OUTPUT_ROOT"
  STATUS="$OUTPUT_ROOT/job_status.tsv"
  if [[ ! -f "$STATUS" ]]; then
    printf 'utc_time\tcollection\tretriever\tmodel\tstatus\texit_code\n' > "$STATUS"
  fi
fi
FAILED=0
NUMBER=0
while IFS=$'\t' read -r COLLECTION DATA OUTPUT; do
  for RETRIEVER in bm25 dense; do
    for MODEL in 2b 9b; do
      NUMBER=$((NUMBER + 1))
      DESTINATION="$OUTPUT/$RETRIEVER/$MODEL"
      COMMAND=("$PROJECT_ROOT/scripts/run_rag_analysis.sh" --data "$DATA" --retriever "$RETRIEVER" --model "$MODEL" --output "$DESTINATION" "${EXTRA[@]}")
      printf '\n[%s/%s] %s | %s | JEV %s\n' "$NUMBER" "$((COUNT * 4))" "$COLLECTION" "$RETRIEVER" "$MODEL"
      if [[ "$DRY_RUN" == 1 ]]; then
        printf '%q ' "${COMMAND[@]}"
        printf '\n'
        continue
      fi
      LOG="$OUTPUT_ROOT/logs/$COLLECTION/$RETRIEVER-$MODEL.log"
      mkdir -p -- "$(dirname -- "$LOG")"
      CODE=0
      if [[ "$QUIET" == 1 ]]; then
        # Capture prints, Python logging, third-party progress bars and tracebacks.
        "${COMMAND[@]}" >> "$LOG" 2>&1 || CODE=$?
      else
        "${COMMAND[@]}" 2>&1 | tee -a "$LOG" || CODE=$?
      fi
      if [[ "$CODE" == 0 ]]; then
        STATE=complete
        printf 'Completed: %s | %s | JEV %s\n' "$COLLECTION" "$RETRIEVER" "$MODEL"
      else
        STATE=failed; FAILED=1
        printf 'Failed: %s/%s/%s (exit %s). Log: %s\n' "$COLLECTION" "$RETRIEVER" "$MODEL" "$CODE" "$LOG" >&2
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$COLLECTION" "$RETRIEVER" "$MODEL" "$STATE" "$CODE" >> "$STATUS"
    done
  done
done < "$JOBS"
if [[ "$DRY_RUN" == 0 ]]; then
  # Collect completed results without pooling queries from different datasets.
  "$PYTHON_BIN" - "$JOBS" "$OUTPUT_ROOT" <<'PY'
import csv
import json
from pathlib import Path
import sys
rows = []
for line in Path(sys.argv[1]).read_text().splitlines():
    collection, _, output = line.split('\t')
    for retriever in ('bm25','dense'):
        for model in ('2b','9b'):
            run = Path(output)/retriever/model
            if not (run/'status.json').exists() or json.loads((run/'status.json').read_text()).get('status') != 'complete':
                continue
            for summary in json.loads((run/'results.json').read_text())['summary']:
                rows.append({'collection':collection,'retriever':retriever,'model':model,**summary})
if rows:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (Path(sys.argv[2])/'all_results.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    print(f'Saved {len(rows)} result rows to {Path(sys.argv[2])/"all_results.csv"}')
PY
fi
exit "$FAILED"
