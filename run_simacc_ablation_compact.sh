#!/usr/bin/env bash

set -euo pipefail

# Run from repo root:
#   bash ./run_simacc_ablation_compact.sh
#   bash ./run_simacc_ablation_compact.sh jnu
#
# Compact ablation:
# - Focus on the stronger late-stage tau region observed on JNU
# - Compare original / no-size / weakened-size weighting

PYTHON_BIN="python"
MAIN_SCRIPT="main.py"
SYSTEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/system" && pwd)"

DATASET="${1:-jnu}"

case "$DATASET" in
  cwru)
    NUM_CLASSES=10
    ;;
  jnu)
    NUM_CLASSES=4
    ;;
  *)
    echo "Unsupported dataset: $DATASET"
    echo "Usage: bash ./run_simacc_ablation_compact.sh [cwru|jnu]"
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

# Compact search space: 1 baseline + 4 tau pairs x 3 aggregation variants = 13 runs
SIM_TAU_LIST=(2.0 4.0)
ACC_TAU_LIST=(2.0 4.0)
SIZE_ALPHA_LIST=(0.5)

format_tag() {
  local value="$1"
  value="${value//-/'neg'}"
  value="${value//./'p'}"
  printf '%s\n' "$value"
}

cd "$SYSTEM_DIR"

echo "Dataset: ${DATASET} (num_classes=${NUM_CLASSES})"
echo "Running compact SimAcc ablations ..."

echo "Running FedAvg baseline ..."
"$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  -algo FedAvg \
  -go compact_baseline_fedavg

for stau in "${SIM_TAU_LIST[@]}"; do
  stau_tag="$(format_tag "$stau")"
  for atau in "${ACC_TAU_LIST[@]}"; do
    atau_tag="$(format_tag "$atau")"

    echo "Running FedAvgSimAcc with sim_tau=${stau}, acc_tau=${atau} ..."
    "$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
      -algo FedAvgSimAcc \
      -go "compact_orig_stau_${stau_tag}_atau_${atau_tag}" \
      -stau "$stau" \
      -atau "$atau"

    echo "Running FedAvgSimAccNoSize with sim_tau=${stau}, acc_tau=${atau} ..."
    "$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
      -algo FedAvgSimAccNoSize \
      -go "compact_nosize_stau_${stau_tag}_atau_${atau_tag}" \
      -stau "$stau" \
      -atau "$atau"

    for alpha in "${SIZE_ALPHA_LIST[@]}"; do
      alpha_tag="$(format_tag "$alpha")"
      echo "Running FedAvgSimAccSizeAlpha with sim_tau=${stau}, acc_tau=${atau}, size_alpha=${alpha} ..."
      "$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
        -algo FedAvgSimAccSizeAlpha \
        -go "compact_alpha_${alpha_tag}_stau_${stau_tag}_atau_${atau_tag}" \
        -stau "$stau" \
        -atau "$atau" \
        -sa "$alpha"
    done
  done
done
