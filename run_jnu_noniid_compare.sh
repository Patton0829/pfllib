#!/usr/bin/env bash

set -euo pipefail

# Suggested workflow:
#   1. Generate the noniid JNU split
#      python dataset/generate_jnu.py noniid - - 42 balanced 0.12
#   2. Run this script from repo root
#      bash ./run_jnu_noniid_compare.sh

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
  -gr 2000
  -ls 2
  -nc 20
  -jr 0.5
  -eg 1
  -pg 1
  -t 1
)

SIM_TAU_LIST=(0.5 1.0 2.0 4.0)
ACC_TAU_LIST=(0.5 1.0 2.0 4.0)

format_tag() {
  local value="$1"
  value="${value//-/'neg'}"
  value="${value//./'p'}"
  printf '%s\n' "$value"
}

cd "$SYSTEM_DIR"

echo "Running JNU noniid comparison experiments ..."

echo "Running FedAvg baseline ..."
"$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  -algo FedAvg \
  -go jnu_noniid_baseline_fedavg

for stau in "${SIM_TAU_LIST[@]}"; do
  stau_tag="$(format_tag "$stau")"
  echo "Running FedAvgSimNormNoSize with sim_tau=${stau} ..."
  "$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
    -algo FedAvgSimNormNoSize \
    -go "jnu_noniid_simnorm_nosize_stau_${stau_tag}" \
    -stau "$stau"
done

for atau in "${ACC_TAU_LIST[@]}"; do
  atau_tag="$(format_tag "$atau")"
  echo "Running FedAvgAcc with acc_tau=${atau} ..."
  "$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
    -algo FedAvgAcc \
    -go "jnu_noniid_acc_atau_${atau_tag}" \
    -atau "$atau"
done
