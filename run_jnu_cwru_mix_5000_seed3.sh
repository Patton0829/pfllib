#!/usr/bin/env bash

set -euo pipefail

# Suggested workflow:
#   1. Generate the mixed-domain split for each seed
#      python dataset/generate_jnu_cwru_mix.py noniid 42 severe 0.50 balanced
#      python dataset/generate_jnu_cwru_mix.py noniid 43 severe 0.50 balanced
#      python dataset/generate_jnu_cwru_mix.py noniid 44 severe 0.50 balanced
#   2. Run this script from repo root on the server
#      bash ./run_jnu_cwru_mix_5000_seed3.sh
#
# Purpose:
#   - answer whether FedAvg only drops temporarily at the tail
#   - compare FedAvg vs FedSimNorm under a longer training budget
#   - use 3 seeds for a more robust conclusion

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
  -ls 5
  -nc 20
  -jr 0.3
  -eg 1
  -pg 1
  -t 1
)

SEEDS=(42 43 44)

cd "$SYSTEM_DIR"

echo "Running JNU + CWRU mixed-domain 5000-round seed-3 comparison ..."
echo "Expected dataset split per seed: noniid / severe / size_jitter_ratio=0.50 / test_split_mode=balanced"
echo "Training setup: join_ratio=0.30, local_epochs=5, global_rounds=5000"
echo "Methods: FedAvg baseline vs FedSimNormNoSize(stau=4.0)"

for seed in "${SEEDS[@]}"; do
  echo "=================================================="
  echo "Preparing dataset for seed=${seed}"
  cd ..
  "$PYTHON_BIN" dataset/generate_jnu_cwru_mix.py noniid "$seed" severe 0.50 balanced
  cd system

  echo "Running FedAvg baseline for seed=${seed} ..."
  "$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
    -algo FedAvg \
    -go "jnu_cwru_mix_5000_seed${seed}_baseline_fedavg"

  echo "Running FedSimNormNoSize(stau=4.0) for seed=${seed} ..."
  "$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
    -algo FedAvgSimNormNoSize \
    -go "jnu_cwru_mix_5000_seed${seed}_simnorm_nosize_stau_4p0" \
    -stau 4.0
done
