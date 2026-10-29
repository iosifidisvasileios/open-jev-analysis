#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
exec "$PYTHON_BIN" -m jev_guardrails.rag_analysis "$@"
