#!/usr/bin/env bash

set -euo pipefail

# Suggested workflow:
#   1. Generate the hardcase JNU split first
#      python dataset/generate_jnu.py noniid - - 42 severe 0.25
#   2. Run this script from repo root on the server
#      bash ./run_jnu_hardcase_similarity_baselines.sh
#
# This script compares similarity-weighted variants of strong baselines.

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
  -stau 4.0
)

cd "$SYSTEM_DIR"

echo "Running JNU hardcase similarity-weighted baseline experiments ..."
echo "Expected dataset split: noniid / severe / size_jitter_ratio=0.25"

echo "Running FedProxSimNormNoSize ..."
"$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  -algo FedProxSimNormNoSize \
  -go jnu_hardcase_fedprox_simnorm_nosize \
  -mu 0.01

echo "Running SCAFFOLDSimNormNoSize ..."
"$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  -algo SCAFFOLDSimNormNoSize \
  -go jnu_hardcase_scaffold_simnorm_nosize
