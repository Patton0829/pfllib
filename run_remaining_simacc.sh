#!/usr/bin/env bash

set -euo pipefail

# Run from repo root:
#   bash ./run_remaining_simacc.sh
#   bash ./run_remaining_simacc.sh cwru
#   bash ./run_remaining_simacc.sh jnu

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
    echo "Usage: bash ./run_remaining_simacc.sh [cwru|jnu]"
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

SIMACC_ALGORITHM="FedAvgSimAcc"

# Only the remaining combinations that were not completed yet.
SIMACC_STAU_LIST=(1.0 2.0 4.0)

format_tau_tag() {
  local value="$1"
  value="${value//-/'neg'}"
  value="${value//./'p'}"
  printf '%s\n' "$value"
}

cd "$SYSTEM_DIR"

echo "Dataset: ${DATASET} (num_classes=${NUM_CLASSES})"
echo "Running remaining ${SIMACC_ALGORITHM} experiments ..."

for stau in "${SIMACC_STAU_LIST[@]}"; do
  stau_tag="$(format_tau_tag "$stau")"

  if [[ "$stau" == "1.0" ]]; then
    atau_list=(2.0 4.0)
  else
    atau_list=(0.5 1.0 2.0 4.0)
  fi

  for atau in "${atau_list[@]}"; do
    atau_tag="$(format_tau_tag "$atau")"
    goal="simacc_stau_${stau_tag}_atau_${atau_tag}"
    echo "Running ${SIMACC_ALGORITHM} with sim_tau=${stau}, acc_tau=${atau} ..."
    "$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
      -algo "$SIMACC_ALGORITHM" \
      -go "$goal" \
      -stau "$stau" \
      -atau "$atau"
  done
done
