#!/usr/bin/env bash
# Usage: run_openjev_dataset_suites.sh [all|deepset|wildguard|toxicchat|toxicchat_jailbreaking|xstest|moderation] [suite flags]
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Usage: bash scripts/run_openjev_dataset_suites.sh [TASK|all] [experiment options]

Tasks: deepset, wildguard, toxicchat, toxicchat_jailbreaking, xstest, moderation.
Default: all. Both model sizes run sequentially for each task.

Examples:
  bash scripts/run_openjev_dataset_suites.sh deepset --dry-run
  bash scripts/run_openjev_dataset_suites.sh deepset --output runs/deepset-study
  JEV_SUITE_OUTPUT_ROOT=runs/classification-study bash scripts/run_openjev_dataset_suites.sh all

Options pass to run_openjev_dataset_experiments.py suite, including --resume,
--train-samples, --validation-samples, --test-samples, --breadth and --depth.
Use --output for a single task; use JEV_SUITE_OUTPUT_ROOT for all tasks.
HELP
  exit 0
fi
DATASET="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$DATASET" in
  all) DATASETS=(deepset wildguard toxicchat toxicchat_jailbreaking xstest moderation) ;;
  deepset|wildguard|toxicchat|toxicchat_jailbreaking|xstest|moderation) DATASETS=("$DATASET") ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac
# --output is reserved for a single dataset; all uses a shared output root.
if [[ "$DATASET" == all ]]; then
  for arg in "$@"; do
    if [[ "$arg" == --output || "$arg" == --output=* || "$arg" == --config || "$arg" == --config=* ]]; then
      echo 'For all datasets, set JEV_SUITE_OUTPUT_ROOT instead of --output; configs are dataset-specific.' >&2
      exit 2
    fi
  done
  OUTPUT_ROOT="${JEV_SUITE_OUTPUT_ROOT:-$PROJECT_ROOT/runs/openjev-datasets-$(date -u +%Y%m%dT%H%M%SZ)}"
fi
DRY_RUN=0
for arg in "$@"; do if [[ "$arg" == --dry-run ]]; then DRY_RUN=1; fi; done
FAILED=0
for name in "${DATASETS[@]}"; do
  echo "=== Dataset: $name ==="
  EXTRA=()
  if [[ "$DATASET" == all ]]; then EXTRA=(--output "$OUTPUT_ROOT/$name"); fi
  if ! "$PYTHON_BIN" "$PROJECT_ROOT/run_openjev_dataset_experiments.py" suite \
      --config "$PROJECT_ROOT/config/suites/$name.json" "${EXTRA[@]}" "$@"; then
    FAILED=1
    echo "Dataset $name failed; retaining results and continuing." >&2
  fi
done
if [[ "$DATASET" == all && "$DRY_RUN" == 0 ]]; then
  if ! "$PYTHON_BIN" "$PROJECT_ROOT/run_openjev_dataset_experiments.py" collect --output "$OUTPUT_ROOT"; then
    FAILED=1
  fi
fi
exit "$FAILED"
