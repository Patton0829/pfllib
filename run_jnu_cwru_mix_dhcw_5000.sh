#!/usr/bin/env bash

set -euo pipefail

# Suggested workflow:
#   1. Generate the mixed-domain split first
#      python dataset/generate_jnu_cwru_mix.py noniid 42 severe 0.50 balanced
#   2. Run this script from repo root on the server
#      bash ./run_jnu_cwru_mix_dhcw_5000.sh
#
# Purpose:
#   - run only DHCW-FL for 5000 rounds
#   - compare with previous 5000-round FedAvg/FedSimNorm results

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
  -gr 5000
  -ls 1
  -nc 20
  -jr 0.3
  -eg 1
  -pg 1
  -t 1
)

cd "$SYSTEM_DIR"

echo "Running JNU + CWRU mixed-domain DHCW-FL only ..."
echo "Expected dataset split: noniid / severe / size_jitter_ratio=0.50 / test_split_mode=balanced"
echo "Training setup: join_ratio=0.30, local_epochs=1, global_rounds=5000"
echo "DHCW-FL setup: lambda=0.8, tau=3.0, eta=3.0"

"$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  -algo DHCWFL \
  -go jnu_cwru_mix_5000_seed42_dhcw_full_lam0p8_tau3_eta3 \
  --dhcw_history_lambda 0.8 \
  --dhcw_group_tau 3.0 \
  --dhcw_domain_eta 3.0
