#!/usr/bin/env bash

set -euo pipefail

echo "Running STRICT PU baselines on server:"
echo "  Dataset split: by_source_file"
echo "  1) MOON"
echo "  2) FedProto"
echo "Settings: bs=512, rounds=5000, local_epochs=1, join_ratio=0.2, eval_gap=10, print_gap=10"
echo ""

cd system

python main.py \
  -data pu \
  -ncl 4 \
  -m CNN1D \
  -lbs 512 \
  -dev cuda \
  -did 0 \
  -lr 0.001 \
  -gr 5000 \
  -ls 1 \
  -nc 20 \
  -jr 0.2 \
  -eg 10 \
  -pg 10 \
  -t 1 \
  -algo MOON \
  -tau 1.0 \
  -mu 1.0 \
  -go pu_strict_5000_ls1_bs512_jr02_eval10_moon

python main.py \
  -data pu \
  -ncl 4 \
  -m CNN1D \
  -lbs 512 \
  -dev cuda \
  -did 0 \
  -lr 0.001 \
  -gr 5000 \
  -ls 1 \
  -nc 20 \
  -jr 0.2 \
  -eg 10 \
  -pg 10 \
  -t 1 \
  -algo FedProto \
  -go pu_strict_5000_ls1_bs512_jr02_eval10_fedproto
