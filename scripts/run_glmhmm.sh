#!/usr/bin/env bash
# Full GLM-HMM protocol (behaviour only). Two heavy processes at most.
# Usage: caffeinate -i bash scripts/run_glmhmm.sh > results/glmhmm/run.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
PY="nice -n 10 .venv/bin/python -m src.glmhmm"
$PY build --design A90
$PY build --design P
( $PY global --design A90 && $PY cv --design A90 && $PY mouse --design A90 && $PY select --design A90 \
  && $PY label --design A90 --k "$(.venv/bin/python -c "import json;print(json.load(open('results/glmhmm/A90/k_selection.json'))['K_chosen_1se'])")" \
  && $PY cv --design P ) &
A=$!
$PY global --design P &
B=$!
wait $A; wait $B
$PY mouse --design P
$PY select --design P
K=$(.venv/bin/python -c "import json;print(json.load(open('results/glmhmm/P/k_selection.json'))['K_chosen_1se'])")
$PY label --design P --k "$K"
echo "DONE: chosen K = $K"
