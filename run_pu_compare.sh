#!/usr/bin/env bash

set -euo pipefail

# Suggested workflow:
#   1. Generate the PU split first
#      python dataset/generate_pu.py noniid - - 42 severe 0.25 balanced
#   2. Run this script from repo root
#      bash ./run_pu_compare.sh
#
# Purpose:
#   - compare FedAvg and FedSimNormNoSize(stau=4.0) on PU

PYTHON_BIN="python"
MAIN_SCRIPT="main.py"
SYSTEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/system" && pwd)"

COMMON_ARGS=(
  -data pu
  -ncl 4
  -m CNN1D
  -lbs 64
  -dev cuda
  -did 0
  -lr 0.001
  -gr 3000
  -ls 5
  -nc 20
  -jr 0.3
  -eg 1
  -pg 1
  -t 1
)

cd "$SYSTEM_DIR"

echo "Running PU comparison experiments ..."
echo "Expected dataset split: noniid / severe / size_jitter_ratio=0.25 / test_split_mode=balanced"
echo "Training setup: join_ratio=0.30, local_epochs=5, global_rounds=3000"

echo "Running FedAvg baseline ..."
"$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  -algo FedAvg \
  -go pu_baseline_fedavg

echo "Running FedSimNormNoSize(stau=4.0) ..."
"$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  -algo FedAvgSimNormNoSize \
  -go pu_simnorm_nosize_stau_4p0 \
  -stau 4.0
