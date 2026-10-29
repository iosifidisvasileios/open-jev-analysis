#!/usr/bin/env bash
# Shared paths and Python selection for the experiment launchers.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${JEV_PYTHON:-}" ]]; then
  PYTHON_BIN="$JEV_PYTHON"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || command -v python)" || {
    echo 'Select your research environment with JEV_PYTHON=/path/to/python.' >&2
    exit 2
  }
fi
export JEV_PYTHON="$PYTHON_BIN"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
