#!/usr/bin/env bash
set -euo pipefail
HIM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HIM_ROOT"
export CYCLONEDDS_HOME="$HIM_ROOT/.deps/cyclonedds-install"
export PYTORCH_ENABLE_MPS_FALLBACK=0
exec .venv-sim2sim/bin/python -m sim2sim.run "$@"
