#!/usr/bin/env bash

set -euo pipefail

# Suggested workflow:
#   1. Generate the noniid JNU split
#      python dataset/generate_jnu.py noniid - - 42 balanced 0.12
#   2. Run this script from repo root on the server
#      bash ./run_jnu_noniid_simnorm_3000.sh
#
# This script only runs the key 3000-round comparisons:
#   - FedAvg baseline
#   - FedAvgSimNormNoSize (stau=2.0)
#   - FedAvgSimNormNoSize (stau=4.0)

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
  -ls 2
  -nc 20
  -jr 0.5
  -eg 1
  -pg 1
  -t 1
)

cd "$SYSTEM_DIR"

echo "Running key 3000-round JNU noniid SimNorm comparisons ..."

echo "Running FedAvg baseline ..."
"$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  -algo FedAvg \
  -go jnu_noniid_baseline_fedavg_3000

echo "Running FedAvgSimNormNoSize with sim_tau=2.0 ..."
"$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  -algo FedAvgSimNormNoSize \
  -go jnu_noniid_simnorm_nosize_stau_2p0_3000 \
  -stau 2.0

echo "Running FedAvgSimNormNoSize with sim_tau=4.0 ..."
"$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  -algo FedAvgSimNormNoSize \
  -go jnu_noniid_simnorm_nosize_stau_4p0_3000 \
  -stau 4.0
