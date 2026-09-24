#!/usr/bin/env bash
# Generate the six second-wave OOD scenarios on a CPU-only node, then build the
# derived stages (motion, tracks) in place. Everything here is CPU-bound; the GPU
# path is not worth it at 480x320 / 16 samples, where the bottleneck is scene
# updates, PNG+EXR writes and the x264 mux rather than path tracing.
#
# Ordered by value: if the allocation ends early, the tail is what's lost.
#
# Usage (on a `cpu` partition node, 128 cores):
#   scripts/gen_ood_cpu.sh
#   THREADS=2 scripts/gen_ood_cpu.sh          # more shards, fewer threads each
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
export PY=${PY:-/work/hu/alpha_hu/akiruga_hu/sw/conda/envs/clevrergen/bin/python}
VWM_PY=${VWM_PY:-/work/hu/alpha_hu/akiruga_hu/sw/conda/envs/test2/bin/python}
VWM=${VWM:-/home/akiruga_hu/dma/video_world_model}
OUTBASE=${OUTBASE:-/scratch/akiruga_hu/datasets/clevrer_gen_new}
THREADS=${THREADS:-4}
export PYTHONNOUSERSITE=1

SCENARIOS=(
  tiny_fast_nocollide     # hero: both blindness mechanisms stacked
  tiny_nocollide          # one-factor control for the above
  fast_nocollide          # one-factor control for the above
  obstacle_bounce         # discrete event with a visible cause
  accel_nocollide         # breaks the const-velocity baseline
  spin_nocollide          # rotation; control is the existing new_shapes set
)

for s in "${SCENARIOS[@]}"; do
  echo ""
  echo "===== $s  $(date +%H:%M:%S)"
  THREADS="$THREADS" bash "$HERE/scripts/launch_cpu.sh" \
    --scenario "$HERE/configs/scenarios/${s}.yaml" \
    --name "$s" --output_base "$OUTBASE"

  # Derived stages. build_motion needs mathutils (ships in bpy) and the
  # derender_proposals RLE, so it must run here, before those get cleaned up.
  echo "----- motion $s $(date +%H:%M:%S)"
  "$PY" -s "$HERE/scripts/build_motion.py" "$OUTBASE/$s" --workers "$(nproc)"

  echo "----- tracks $s $(date +%H:%M:%S)"
  "$VWM_PY" -s "$VWM/scripts/gen_clevrer_tracks.py" --data-root "$OUTBASE/$s"
done

echo ""
echo "All six scenarios complete $(date +%H:%M:%S)"
