#!/usr/bin/env bash

set -euo pipefail

# Suggested workflow:
#   1. Generate the mixed-domain split first
#      python dataset/generate_jnu_cwru_mix.py noniid 42 severe 0.50 balanced
#   2. Run this script from repo root on the server
#      bash ./run_jnu_cwru_mix_compare.sh
#
# Mixed-domain setup:
#   - 10 clients from JNU + 10 clients from CWRU
#   - each client belongs to exactly one domain and one condition
#   - CWRU labels are collapsed to 4 superclasses to align with JNU
#   - balanced test split across clients

PYTHON_BIN="python"
MAIN_SCRIPT="main.py"
SYSTEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/system" && pwd)"

COMMON_ARGS=(
  -data jnu_cwru_mix
  -ncl 4
  -m CNN1D
  -lbs 64
  -dev cuda
  -did 0
  -lr 0.001
  -gr 3000
  -ls 5
  -nc 20
  -jr 0.1
  -eg 1
  -pg 1
  -t 1
)

SIM_TAU_LIST=(2.0 4.0 6.0)

format_tag() {
  local value="$1"
  value="${value//-/'neg'}"
  value="${value//./'p'}"
  printf '%s\n' "$value"
}

cd "$SYSTEM_DIR"

echo "Running JNU + CWRU mixed-domain comparison experiments ..."
echo "Expected dataset split: noniid / severe / size_jitter_ratio=0.50 / test_split_mode=balanced"
echo "Training setup: join_ratio=0.10, local_epochs=5, global_rounds=3000"

echo "Running FedAvg baseline ..."
"$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  -algo FedAvg \
  -go jnu_cwru_mix_baseline_fedavg

for stau in "${SIM_TAU_LIST[@]}"; do
  stau_tag="$(format_tag "$stau")"
  echo "Running FedAvgSimNormNoSize with sim_tau=${stau} ..."
  "$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
    -algo FedAvgSimNormNoSize \
    -go "jnu_cwru_mix_simnorm_nosize_stau_${stau_tag}" \
    -stau "$stau"
done
