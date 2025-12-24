#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python3}

mkdir -p "$ROOT/outputs"{,/logs,/models,/figures,/tables,/summaries}

# Make matplotlib safe in headless environments
export MPLBACKEND=Agg

cd "$ROOT"
