#!/usr/bin/env bash

set -euo pipefail

# Run from repo root:
#   bash ./run_tau_experiments.sh
#   bash ./run_tau_experiments.sh jnu
#
# Update the shared args below to match your experiment settings.

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
    echo "Usage: bash ./run_tau_experiments.sh [cwru|jnu]"
    exit 1
    ;;
esac

COMMON_ARGS=(
  -data "$DATASET"
  -ncl "$NUM_CLASSES"
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
  -pg 10
  -t 1
)

# FedAvg baseline: run once without tau sweep.
BASELINE_ALGORITHM="FedAvg"

# FedAvgSim / FedAvgSimNorm: vary sim_tau here.
SIM_ALGORITHM="FedAvgSim"
SIM_TAU_LIST=(0.5 1.0 2.0 4.0)

# FedAvgAcc: vary acc_tau here.
ACC_ALGORITHM="FedAvgAcc"
ACC_TAU_LIST=(0.5 1.0 2.0 4.0)

# FedAvgSimAcc: vary both sim_tau and acc_tau here.
SIMACC_ALGORITHM="FedAvgSimAcc"
SIMACC_SIM_TAU_LIST=(0.5 1.0 2.0 4.0)
SIMACC_ACC_TAU_LIST=(0.5 1.0 2.0 4.0)

format_tau_tag() {
  local value="$1"
  value="${value//-/'neg'}"
  value="${value//./'p'}"
  printf '%s\n' "$value"
}

cd "$SYSTEM_DIR"

echo "Dataset: ${DATASET} (num_classes=${NUM_CLASSES})"

echo "Running ${BASELINE_ALGORITHM} baseline ..."
"$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  -algo "$BASELINE_ALGORITHM" \
  -go baseline_fedavg

for tau in "${SIM_TAU_LIST[@]}"; do
  tau_tag="$(format_tau_tag "$tau")"
  goal="sim_stau_${tau_tag}"
  echo "Running ${SIM_ALGORITHM} with sim_tau=${tau} ..."
  "$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
    -algo "$SIM_ALGORITHM" \
    -go "$goal" \
    -stau "$tau"
done

for tau in "${ACC_TAU_LIST[@]}"; do
  tau_tag="$(format_tau_tag "$tau")"
  goal="acc_atau_${tau_tag}"
  echo "Running ${ACC_ALGORITHM} with acc_tau=${tau} ..."
  "$PYTHON_BIN" "$MAIN_SCRIPT" "${COMMON_ARGS[@]}" \
    -algo "$ACC_ALGORITHM" \
    -go "$goal" \
    -atau "$tau"
done

for stau in "${SIMACC_SIM_TAU_LIST[@]}"; do
  stau_tag="$(format_tau_tag "$stau")"
  for atau in "${SIMACC_ACC_TAU_LIST[@]}"; do
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
