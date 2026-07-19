#!/usr/bin/env bash
# Same as gen_all.sh, but distributes each scenario across CPU-only shards
# (scripts/launch_cpu.sh) instead of GPU/MIG slices. Intended for a
# CPU-only partition (e.g. sapphire: 112 cores/node, no GPU).
#
# Usage:
#   scripts/gen_all_cpu.sh                  # 250 videos each, default seed
#   scripts/gen_all_cpu.sh --num_videos 250
#   SHARDS=14 THREADS=8 scripts/gen_all_cpu.sh --num_videos 250
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
NUM_VIDEOS=${1:-250}
EXTRA_ARGS=("${@:2}")   # forward any extra flags to generate_dataset.py

SCENARIOS=(
  one_ball_nocollide
  two_balls_nocollide
  three_balls_nocollide
  four_balls_nocollide
  five_balls_nocollide
)

for scenario in "${SCENARIOS[@]}"; do
  echo ""
  echo "===== $scenario  (${NUM_VIDEOS} videos, CPU) ====="
  bash "$HERE/scripts/launch_cpu.sh" \
    --scenario "$HERE/configs/scenarios/${scenario}.yaml" \
    --num_videos "$NUM_VIDEOS" \
    --name "$scenario" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
done

echo ""
echo "All scenarios complete."
