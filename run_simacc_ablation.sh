#!/usr/bin/env bash

set -euo pipefail

# Run from repo root:
#   bash ./run_simacc_ablation.sh
#   bash ./run_simacc_ablation.sh jnu
#
# This script runs three families of experiments:
# 1. FedAvgSimAcc           : original sample-size x sim x acc
# 2. FedAvgSimAccNoSize     : remove sample-size effect
# 3. FedAvgSimAccSizeAlpha  : weaken sample-size effect with size_alpha

PYTHON_BIN="python"
MAIN_SCRIPT="main.py"
SYSTEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/system" && pwd)"

DATASET="${1:-cwru}"

case "$DATASET" in
  cwru)
    NUM_CLASSES=10
    ;;
  jnu)
    NUM_CLASSES=4
    ;;
  *)
    echo "Unsupported dataset: $DATASET"
    echo "Usage: bash ./run_simacc_ablation.sh [cwru|jnu]"
    exit 1
    ;;
esac

COMMON_ARGS=(
  -data "$DATASET"
  -ncl "$NUM_CLASSES"
  -m CNN1D
  -dev cuda
  -did 0
  -lr 0.005
  -gr 3000
  -ls 3
  -nc 20
  -jr 0.2
  -eg 1
  -pg 10
  -t 1
)

# Tau ablation lists.
SIM_TAU_LIST=(0.5 1.0 2.0 4.0)
ACC_TAU_LIST=(0.5 1.0 2.0 4.0)

# Weakened sample-size effect: alpha=1.0 means original, alpha=0.0 means no size effect.
# Use values in (0,1) to weaken.
SIZE_ALPHA_LIST=(0.5)

format_tau_tag() {
  local value="$1"
  value="${value//-/'neg'}"
  value="${value//./'p'}"
  printf '%s\n' "$value"
}

cd "$SYSTEM_DIR"

echo "Dataset: ${DATASET} (num_classes=${NUM_CLASSES})"
echo "Running SimAcc ablations ..."

for stau in "${SIM_TAU_LIST[@]}"; do
  stau_tag="$(format_tau_tag "$stau")"
  for atau in "${ACC_TAU_LIST[@]}"; do
    atau_tag="$(format_tau_tag "$atau")"

    echo "Running FedAvgSimAcc with sim_tau=${stau}, acc_tau=${atau} ..."
    "$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
      -algo FedAvgSimAcc \
      -go "simacc_orig_stau_${stau_tag}_atau_${atau_tag}" \
      -stau "$stau" \
      -atau "$atau"

    echo "Running FedAvgSimAccNoSize with sim_tau=${stau}, acc_tau=${atau} ..."
    "$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
      -algo FedAvgSimAccNoSize \
      -go "simacc_nosize_stau_${stau_tag}_atau_${atau_tag}" \
      -stau "$stau" \
      -atau "$atau"

    for alpha in "${SIZE_ALPHA_LIST[@]}"; do
      alpha_tag="$(format_tau_tag "$alpha")"
      echo "Running FedAvgSimAccSizeAlpha with sim_tau=${stau}, acc_tau=${atau}, size_alpha=${alpha} ..."
      "$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
        -algo FedAvgSimAccSizeAlpha \
        -go "simacc_alpha_${alpha_tag}_stau_${stau_tag}_atau_${atau_tag}" \
        -stau "$stau" \
        -atau "$atau" \
        -sa "$alpha"
    done
  done
done
