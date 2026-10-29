#!/usr/bin/env bash
# Sequential server + experiment processes; all arguments go to the Python runner.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/run_openjev_experiments.py" suite "$@"
