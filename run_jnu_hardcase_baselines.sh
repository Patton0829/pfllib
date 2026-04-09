#!/usr/bin/env bash

set -euo pipefail

# Suggested workflow:
#   1. Generate the hardcase JNU split first
#      python dataset/generate_jnu.py noniid - - 42 severe 0.25
#   2. Run this script from repo root on the server
#      bash ./run_jnu_hardcase_baselines.sh
#
# This script complements the experiments you already ran for:
#   - FedAvg
#   - FedSimNorm (FedAvgSimNormNoSize)
# and adds three strong external baselines:
#   - FedProx
#   - SCAFFOLD
#   - FedDyn

PYTHON_BIN="python"
MAIN_SCRIPT="main.py"
SYSTEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/system" && pwd)"

COMMON_ARGS=(
  -data jnu
  -ncl 4
  -m CNN1D
  -lbs 64
  -dev cuda
  -did 0
  -lr 0.001
  -gr 3000
  -ls 5
  -nc 20
  -jr 0.2
  -eg 1
  -pg 1
  -t 1
)

cd "$SYSTEM_DIR"

echo "Running JNU hardcase external baseline experiments ..."
echo "Expected dataset split: noniid / severe / size_jitter_ratio=0.25"

echo "Running FedProx ..."
"$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  -algo FedProx \
  -go jnu_hardcase_fedprox \
  -mu 0.01

echo "Running SCAFFOLD ..."
"$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  -algo SCAFFOLD \
  -go jnu_hardcase_scaffold

echo "Running FedDyn ..."
"$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  -algo FedDyn \
  -go jnu_hardcase_feddyn \
  -lam 1.0
