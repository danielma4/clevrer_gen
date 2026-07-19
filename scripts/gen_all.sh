#!/usr/bin/env bash
# Generate 1000 videos each of 3-, 4-, and 5-ball no-collision scenarios,
# distributed across all available MIG slices. Runs sequentially so all 8
# MIG slices are dedicated to one scenario at a time.
#
# Usage:
#   scripts/gen_all.sh                  # 1000 videos each, default seed
#   scripts/gen_all.sh --num_videos 50  # quick test run
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
  echo "===== $scenario  (${NUM_VIDEOS} videos) ====="
  bash "$HERE/scripts/launch_local.sh" \
    --scenario "$HERE/configs/scenarios/${scenario}.yaml" \
    --num_videos "$NUM_VIDEOS" \
    --name "$scenario" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
done

echo ""
echo "All scenarios complete."
